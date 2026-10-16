import Foundation
import GGUFCore
import MLX
import MLXBridge
import MinirunKit
import StorageCore
import LlamaTokenizerRuntime

public typealias GGUFResidentWeightCacheStatistics = GGUFPackedTileCache.Statistics

/// Projection-only cumulative work on the weight source's serial executor.
/// Logical weight bytes include cache hits. I/O wait includes cache lookup and
/// lease acquisition; these figures do not claim physical flash-controller I/O.
public struct GGUFProjectionStatistics: Codable, Equatable, Sendable {
    public private(set) var calls: UInt64 = 0
    public private(set) var tiles: UInt64 = 0
    public private(set) var logicalWeightBytes: UInt64 = 0
    public private(set) var wallSeconds = 0.0
    public private(set) var waitOnIOSeconds = 0.0
    public private(set) var dequantizeAndCopySeconds = 0.0
    public private(set) var computeSeconds = 0.0
    public private(set) var activationRoundingCalls: UInt64 = 0
    public private(set) var activationRoundingSeconds = 0.0
    public private(set) var packedMetalCalls: UInt64? = 0
    public private(set) var packedWeightCopyBytes: UInt64? = 0
    public private(set) var residentWeightBytesReused: UInt64? = 0

    mutating func record(_ report: GGUFLinearReport, roundingSeconds: Double = 0, rounded: Bool = false) throws {
        func add(_ a: UInt64, _ b: UInt64) throws -> UInt64 {
            let value = a.addingReportingOverflow(b)
            guard !value.overflow else { throw GGUFError.limit("projection counter overflow") }
            return value.partialValue
        }
        calls = try add(calls, 1); tiles = try add(tiles, UInt64(report.tiles))
        logicalWeightBytes = try add(logicalWeightBytes, report.logicalWeightBytes)
        packedMetalCalls = try add(packedMetalCalls ?? 0, report.execution == .packedMetal ? 1 : 0)
        packedWeightCopyBytes = try add(packedWeightCopyBytes ?? 0, report.packedWeightCopyBytes ?? 0)
        residentWeightBytesReused = try add(residentWeightBytesReused ?? 0, report.residentWeightBytesReused ?? 0)
        if rounded { activationRoundingCalls = try add(activationRoundingCalls, 1) }
        activationRoundingSeconds += roundingSeconds
        wallSeconds += report.wallSeconds + roundingSeconds; waitOnIOSeconds += report.waitOnIOSeconds
        dequantizeAndCopySeconds += report.dequantizeAndCopySeconds; computeSeconds += report.computeSeconds
        guard [wallSeconds, waitOnIOSeconds, dequantizeAndCopySeconds, computeSeconds, activationRoundingSeconds].allSatisfy({ $0.isFinite && $0 >= 0 }) else {
            throw GGUFError.limit("projection timing overflow")
        }
    }

    public func subtracting(_ earlier: Self) throws -> Self {
        guard calls >= earlier.calls, tiles >= earlier.tiles, logicalWeightBytes >= earlier.logicalWeightBytes,
              wallSeconds >= earlier.wallSeconds, waitOnIOSeconds >= earlier.waitOnIOSeconds,
              dequantizeAndCopySeconds >= earlier.dequantizeAndCopySeconds, computeSeconds >= earlier.computeSeconds,
              activationRoundingCalls >= earlier.activationRoundingCalls,
              activationRoundingSeconds >= earlier.activationRoundingSeconds else {
            throw GGUFError.invalid("projection counters went backwards")
        }
        var value = Self()
        value.calls = calls - earlier.calls; value.tiles = tiles - earlier.tiles
        value.logicalWeightBytes = logicalWeightBytes - earlier.logicalWeightBytes
        value.wallSeconds = wallSeconds - earlier.wallSeconds; value.waitOnIOSeconds = waitOnIOSeconds - earlier.waitOnIOSeconds
        value.dequantizeAndCopySeconds = dequantizeAndCopySeconds - earlier.dequantizeAndCopySeconds
        value.computeSeconds = computeSeconds - earlier.computeSeconds
        value.activationRoundingCalls = activationRoundingCalls - earlier.activationRoundingCalls
        value.activationRoundingSeconds = activationRoundingSeconds - earlier.activationRoundingSeconds
        for (a, b) in [(packedMetalCalls, earlier.packedMetalCalls), (packedWeightCopyBytes, earlier.packedWeightCopyBytes),
                       (residentWeightBytesReused, earlier.residentWeightBytesReused)] {
            guard (a ?? 0) >= (b ?? 0) else { throw GGUFError.invalid("packed projection counters went backwards") }
        }
        value.packedMetalCalls = (packedMetalCalls ?? 0) - (earlier.packedMetalCalls ?? 0)
        value.packedWeightCopyBytes = (packedWeightCopyBytes ?? 0) - (earlier.packedWeightCopyBytes ?? 0)
        value.residentWeightBytesReused = (residentWeightBytesReused ?? 0) - (earlier.residentWeightBytesReused ?? 0)
        return value
    }
}

/// A bounded weight source for a fully verified GGUF. The only file opener is
/// the verification capability; no model pathname is reopened by this class.
/// Keep this instance confined to the runner's serial inference executor.
public final class GGUFWeightSource {
    public let artifact: VerifiedGGUFArtifact
    public let rawCacheBytes: Int
    public let dequantizationScratchBytes: Int
    public private(set) var projectionStatistics = GGUFProjectionStatistics()
    private let pool: BufferPool
    private let reader: TileReader
    private let scratch: GGUFLinearScratch
    private let smallScratch: AlignedBuffer
    private let activationScratch: AlignedBuffer
    private let activationPackingScratch: AlignedBuffer
    private let sourceIDs: [String: UInt64]
    private let tileBytes: Int
    private let execution: GGUFLinearExecution
    private let packedCache: GGUFPackedTileCache
    private var closed = false

    public init(artifact: VerifiedGGUFArtifact, rawCacheBytes: Int,
                tileBytes: Int = 1 << 20, noCache: Bool = true,
                execution: GGUFLinearExecution = .packedMetal, residentCacheBytes: Int = 0) throws {
        guard (4096...(4 << 20)).contains(tileBytes), rawCacheBytes >= tileBytes * 2,
              rawCacheBytes <= 1_000_000_000, residentCacheBytes >= 0,
              residentCacheBytes <= 4_500_000_000 else { throw GGUFError.limit("weight cache") }
        self.artifact = artifact
        self.tileBytes = tileBytes
        self.execution = execution
        let slots = rawCacheBytes / tileBytes
        guard slots <= 4096 else { throw GGUFError.limit("raw cache slot metadata") }
        pool = try BufferPool(configuration: .init(slotCount: slots, slotBytes: tileBytes,
                                                   cachePolicy: .leastRecentlyUsed))
        self.rawCacheBytes = pool.configuration.budgetBytes
        scratch = try GGUFLinearScratch(elements: 2 * 1024 * 1024)
        smallScratch = try AlignedBuffer(count: 65536 * 4)
        activationScratch = try AlignedBuffer(count: 65536 * 4)
        activationPackingScratch = try AlignedBuffer(count: 65536 * 2)
        dequantizationScratchBytes = scratch.byteCount + smallScratch.count
            + activationScratch.count + activationPackingScratch.count
        sourceIDs = Dictionary(uniqueKeysWithValues: artifact.index.tensors.keys.sorted().enumerated().map {
            ($0.element, UInt64($0.offset + 1))
        })
        // Admit whole matrices in decreasing size, deterministically. Rows of
        // the large token/PLE tables and all routed experts remain streamed.
        // Reserving dense entries separately avoids a cyclic LRU with no hits.
        var remaining = residentCacheBytes, admitted = Set<UInt64>()
        for tensor in artifact.index.tensors.values.sorted(by: {
            $0.byteCount == $1.byteCount ? $0.name < $1.name : $0.byteCount > $1.byteCount
        }) where execution == .packedMetal && tensor.dimensions.count == 2
            && tensor.name != "token_embd.weight" && tensor.name != "per_layer_token_embd.weight" {
            if tensor.byteCount <= UInt64(max(0, remaining)) {
                admitted.insert(sourceIDs[tensor.name]!); remaining -= Int(tensor.byteCount)
            }
        }
        packedCache = try GGUFPackedTileCache(capacityBytes: residentCacheBytes, admittedSources: admitted)
        reader = try TileReader(owningFileDescriptor: artifact.duplicateForReading(),
            path: artifact.manifest.fileName, pool: pool,
            configuration: .init(queueDepth: 2, readAhead: 1, noCache: noCache, slotAcquireTimeout: 120))
    }

    deinit { shutdown() }

    public func shutdown() {
        guard !closed else { return }
        closed = true; reader.shutdown(); packedCache.clear()
    }

    public var readStatistics: TileReaderStatistics { reader.statistics }
    public var poolStatistics: BufferPoolStatistics { pool.statistics }
    public var residentCacheStatistics: GGUFPackedTileCache.Statistics { packedCache.statistics }

    public func linear(_ x: MLXArray, tensor name: String, expert: UInt64? = nil) throws -> MLXArray {
        try artifact.assertUnchanged()
        guard !closed, let tensor = artifact.index.tensors[name],
              tensor.dimensions.count == 2 || tensor.dimensions.count == 3,
              let columns = Int(exactly: tensor.dimensions[0]),
              let bytes = Int(exactly: tensor.rowBytes), bytes <= tileBytes,
              expert == nil || expert! <= UInt64(UInt32.max) else {
            throw GGUFError.invalid("linear tensor or expert \(name)")
        }
        let rows = execution == .packedMetal ? min(4096, tileBytes / bytes)
            : min(256, tileBytes / bytes, scratch.elements / columns)
        guard rows > 0 else { throw GGUFError.limit("tensor reduction row exceeds scratch") }
        // Each tensor has its own stable source ID; low 32 bits name tiles,
        // high 32 bits name an expert. Cache keys do not grow with token count.
        let firstRegion = (expert ?? 0) << 32
        let roundingStart = MonotonicClock.now()
        let activations = try canonicalActivations(x, type: tensor.quantization, columns: columns)
        let roundingSeconds = tensor.quantization == .f32 ? 0 : MonotonicClock.seconds(since: roundingStart)
        let result = try GGUFStreamedLinear.run(activations, tensor: tensor, expert: expert,
            payloadOffset: artifact.index.payloadOffset, sourceID: sourceIDs[name]!,
            firstRegionID: firstRegion, tileRows: rows, reader: reader, scratch: scratch,
            execution: execution, packedCache: expert == nil ? packedCache : nil)
        try artifact.assertUnchanged()
        try projectionStatistics.record(result.report, roundingSeconds: roundingSeconds, rounded: tensor.quantization != .f32)
        return result.output
    }

    public func linearExperts(_ x: MLXArray, tensor name: String, experts: [UInt64]) throws -> MLXArray {
        try artifact.assertUnchanged()
        guard !closed, let tensor = artifact.index.tensors[name], tensor.dimensions.count == 3,
              !experts.isEmpty, experts.count <= 64, x.ndim == 2,
              (x.shape[0] == 1 || x.shape[0] == experts.count),
              let columns = Int(exactly: tensor.dimensions[0]), x.shape[1] == columns,
              experts.allSatisfy({ $0 < tensor.dimensions[2] && $0 <= UInt64(UInt32.max) }) else {
            throw GGUFError.invalid("grouped expert projection")
        }
        if execution == .dequantized {
            return try concatenated(experts.enumerated().map { slot, expert in
                try linear(x[x.shape[0] == 1 ? 0..<1 : slot..<(slot+1)], tensor: name, expert: expert)
            }, axis: 0)
        }
        let rows = min(4096, (4 << 20) / experts.count / Int(tensor.rowBytes), tileBytes / Int(tensor.rowBytes))
        guard rows > 0 else { throw GGUFError.limit("grouped expert row") }
        let roundingStart = MonotonicClock.now()
        let activations = try canonicalActivations(x, type: tensor.quantization, columns: columns)
        let roundingSeconds = tensor.quantization == .f32 ? 0 : MonotonicClock.seconds(since: roundingStart)
        let result = try GGUFStreamedLinear.runExperts(broadcast(activations, to: [experts.count, columns]),
            tensor: tensor, experts: experts, payloadOffset: artifact.index.payloadOffset,
            sourceID: sourceIDs[name]!, tileRows: rows, reader: reader)
        try artifact.assertUnchanged()
        try projectionStatistics.record(result.report, roundingSeconds: roundingSeconds, rounded: tensor.quantization != .f32)
        return result.output
    }

    // Canonical CPU dot kernels round their RHS to Q8_K/Q8_0/F16/BF16 before
    // multiplication. Preserve that arithmetic while MLX still computes the
    // row-streamed FP32 product. Never quantize or rewrite the model weights.
    func canonicalActivations(_ x: MLXArray, type: GGUFQuantization, columns: Int) throws -> MLXArray {
        if type == .f32 { return x }
        guard x.ndim == 2, x.shape[1] == columns, columns > 0, columns <= 65536,
              x.shape[0] > 0, x.shape[0] <= 65536 / columns else {
            throw GGUFError.limit("canonical activation matrix bound")
        }
        let values = x.asType(.float32).asArray(Float.self)
        let output = activationScratch.pointer.assumingMemoryBound(to: Float.self)
        try values.withUnsafeBufferPointer { input in
            for row in 0..<x.shape[0] {
                let code = flr_round_linear_activations(type.rawValue, input.baseAddress! + row * columns,
                    Int32(columns), output + row * columns, Int32(columns),
                    activationPackingScratch.pointer, Int32(activationPackingScratch.count))
                guard code == 0 else {
                    throw GGUFError.invalid("canonical activation rounding: \(String(cString: flr_last_error()))")
                }
            }
        }
        let result = MLXArray(UnsafeBufferPointer(start: output, count: values.count), x.shape)
        result.eval() // own the copied values before reusing either scratch buffer
        return result
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
