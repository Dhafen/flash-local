import Foundation
import GGUFCore
import MLX
import MLXBridge
import MinirunKit
import StorageCore

/// A bounded weight source for a fully verified GGUF. The only file opener is
/// the verification capability; no model pathname is reopened by this class.
/// Keep this instance confined to the runner's serial inference executor.
public final class GGUFWeightSource {
    public let artifact: VerifiedGGUFArtifact
    public let rawCacheBytes: Int
    public let dequantizationScratchBytes: Int
    private let pool: BufferPool
    private let reader: TileReader
    private let scratch: GGUFLinearScratch
    private let smallScratch: AlignedBuffer
    private let sourceIDs: [String: UInt64]
    private let tileBytes: Int
    private var closed = false

    public init(artifact: VerifiedGGUFArtifact, rawCacheBytes: Int,
                tileBytes: Int = 1 << 20, noCache: Bool = true) throws {
        guard (4096...(4 << 20)).contains(tileBytes), rawCacheBytes >= tileBytes * 2,
              rawCacheBytes <= 1_000_000_000 else { throw GGUFError.limit("raw weight cache") }
        self.artifact = artifact
        self.tileBytes = tileBytes
        let slots = rawCacheBytes / tileBytes
        guard slots <= 4096 else { throw GGUFError.limit("raw cache slot metadata") }
        pool = try BufferPool(configuration: .init(slotCount: slots, slotBytes: tileBytes,
                                                   cachePolicy: .leastRecentlyUsed))
        self.rawCacheBytes = pool.configuration.budgetBytes
        scratch = try GGUFLinearScratch(elements: 2 * 1024 * 1024)
        smallScratch = try AlignedBuffer(count: 65536 * 4)
        dequantizationScratchBytes = scratch.byteCount + smallScratch.count
        sourceIDs = Dictionary(uniqueKeysWithValues: artifact.index.tensors.keys.sorted().enumerated().map {
            ($0.element, UInt64($0.offset + 1))
        })
        reader = try TileReader(owningFileDescriptor: artifact.duplicateForReading(),
            path: artifact.manifest.fileName, pool: pool,
            configuration: .init(queueDepth: 2, readAhead: 1, noCache: noCache, slotAcquireTimeout: 120))
    }

    deinit { shutdown() }

    public func shutdown() {
        guard !closed else { return }
        closed = true; reader.shutdown()
    }

    public var readStatistics: TileReaderStatistics { reader.statistics }
    public var poolStatistics: BufferPoolStatistics { pool.statistics }

    public func linear(_ x: MLXArray, tensor name: String, expert: UInt64? = nil) throws -> MLXArray {
        try artifact.assertUnchanged()
        guard !closed, let tensor = artifact.index.tensors[name],
              tensor.dimensions.count == 2 || tensor.dimensions.count == 3,
              let columns = Int(exactly: tensor.dimensions[0]),
              let bytes = Int(exactly: tensor.rowBytes), bytes <= tileBytes,
              expert == nil || expert! <= UInt64(UInt32.max) else {
            throw GGUFError.invalid("linear tensor or expert \(name)")
        }
        let rows = min(256, tileBytes / bytes, scratch.elements / columns)
        guard rows > 0 else { throw GGUFError.limit("tensor reduction row exceeds scratch") }
        // Each tensor has its own stable source ID; low 32 bits name tiles,
        // high 32 bits name an expert. Cache keys do not grow with token count.
        let firstRegion = (expert ?? 0) << 32
        let result = try GGUFStreamedLinear.run(x, tensor: tensor, expert: expert,
            payloadOffset: artifact.index.payloadOffset, sourceID: sourceIDs[name]!,
            firstRegionID: firstRegion, tileRows: rows, reader: reader, scratch: scratch)
        try artifact.assertUnchanged()
        return result.output
    }

    /// Norms and convolution kernels only; routers use the streamed linear.
    /// Never a large matrix or
    /// PLE table: total decoded size is capped at 256 KiB before any read.
    public func smallTensor(_ name: String) throws -> MLXArray {
        guard let tensor = artifact.index.tensors[name] else { throw GGUFError.invalid("missing \(name)") }
        var elements = UInt64(1)
        for dimension in tensor.dimensions {
            let next = elements.multipliedReportingOverflow(by: dimension)
            guard !next.overflow, next.partialValue <= 65536 else { throw GGUFError.limit("small tensor \(name)") }
            elements = next.partialValue
        }
        let bytes = try decodeRegion(tensor: tensor, offset: 0, bytes: tensor.byteCount,
            elements: Int(elements), regionID: UInt64.max)
        return bytes.reshaped(tensor.dimensions.reversed().map { Int($0) })
    }

    /// Token embeddings and PLE rows share the same bounded lookup path.
    public func row(_ row: UInt64, tensor name: String) throws -> MLXArray {
        guard let tensor = artifact.index.tensors[name], tensor.dimensions.count == 2,
              row < tensor.dimensions[1], tensor.dimensions[0] <= 65536 else {
            throw GGUFError.invalid("lookup row in \(name)")
        }
        let offset = row.multipliedReportingOverflow(by: tensor.rowBytes)
        guard !offset.overflow else { throw GGUFError.invalid("lookup row offset") }
        // Separate row identity from linear tiles for the same tensor.
        guard row < UInt64.max / 2 else { throw GGUFError.invalid("lookup identity") }
        return try decodeRegion(tensor: tensor, offset: offset.partialValue, bytes: tensor.rowBytes,
            elements: Int(tensor.dimensions[0]), regionID: (UInt64(1) << 63) + row)
    }

    private func decodeRegion(tensor: GGUFTensor, offset: UInt64, bytes: UInt64,
                              elements: Int, regionID: UInt64) throws -> MLXArray {
        try artifact.assertUnchanged()
        guard !closed, bytes <= UInt64(tileBytes), elements <= 65536 else {
            throw GGUFError.limit("small read")
        }
        let start = artifact.index.payloadOffset.addingReportingOverflow(tensor.relativeOffset)
        let absolute = start.partialValue.addingReportingOverflow(offset)
        guard !start.overflow, !absolute.overflow else { throw GGUFError.invalid("read offset") }
        let region = RegionDescriptor(identity: .init(sourceID: sourceIDs[tensor.name]!, index: regionID),
                                      offset: absolute.partialValue, length: Int(bytes))
        let lease = try reader.acquire(region)
        defer { lease.release() }
        let output = UnsafeMutableBufferPointer(start: smallScratch.pointer.assumingMemoryBound(to: Float.self), count: elements)
        try GGUFDequantizer.decode(lease.contents, type: tensor.quantization, elements: elements, into: output)
        let result = MLXArray(UnsafeBufferPointer(output), [elements])
        result.eval()
        try artifact.assertUnchanged()
        return result
    }
}
