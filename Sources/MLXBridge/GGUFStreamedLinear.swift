import Foundation
import GGUFCore
import MLX
import StorageCore

public enum GGUFLinearEvaluation: Sendable { case perTile, boundedBatch }

/// A single reusable FP32 tile. Serialises access across callers; the eager
/// MLX copy is charged separately in the run report.
public final class GGUFLinearScratch {
    fileprivate let buffer: AlignedBuffer
    fileprivate let lock = NSLock()
    public let elements: Int

    public init(elements: Int) throws {
        guard elements > 0, elements <= GGUFDequantizer.maximumTileElements else {
            throw GGUFError.limit("linear scratch")
        }
        self.elements = elements
        buffer = try AlignedBuffer(count: elements * MemoryLayout<Float>.size)
    }
    public var byteCount: Int { buffer.count }
}

public struct GGUFLinearReport: Sendable, Codable {
    public let tiles: Int
    public let logicalWeightBytes: UInt64
    public let scratchBytes: Int
    public let peakCopiedWeightBytes: Int
    /// Evaluated row outputs plus final concatenation can coexist.
    public let boundedOutputBytes: Int
    public let wallSeconds: Double
    public let waitOnIOSeconds: Double
    public let dequantizeAndCopySeconds: Double
    public let computeSeconds: Double
    /// Optional additions keep saved reports from the first build readable.
    public let execution: GGUFLinearExecution?
    public let packedWeightCopyBytes: UInt64?
    public let residentWeightBytesReused: UInt64?
    public var evaluationBatches: Int? = nil
}

public enum GGUFStreamedLinear {
    /// Selected experts share a dispatch, not weights or activations. A packed
    /// tile contains only the router's choices; at most 16 MiB is copied and
    /// evaluated together, irrespective of model size or expert count.
    public static func runExperts(
        _ x: MLXArray, tensor: GGUFTensor, experts: [UInt64], payloadOffset: UInt64,
        sourceID: UInt64, tileRows: Int, reader: TileReader,
        evaluation: GGUFLinearEvaluation = .boundedBatch
    ) throws -> (output: MLXArray, report: GGUFLinearReport) {
        guard tensor.dimensions.count == 3, !experts.isEmpty, experts.count <= 64,
              let columns = Int(exactly: tensor.dimensions[0]),
              let rows = Int(exactly: tensor.dimensions[1]),
              x.ndim == 2, x.shape[0] == experts.count, x.shape[1] == columns,
              tileRows > 0, tileRows <= 4096, columns > 0, rows > 0,
              rows <= (16 << 20) / 4 / experts.count,
              experts.allSatisfy({ $0 < tensor.dimensions[2] && $0 <= UInt64(UInt32.max) }),
              tensor.rowBytes <= UInt64((16 << 20) / experts.count / min(rows, tileRows)) else {
            throw GGUFError.limit("grouped expert tile/shape")
        }
        let started = MonotonicClock.now(), count = (rows - 1) / tileRows + 1
        guard count <= 4096 else { throw GGUFError.limit("expert tile count") }
        let activations = x.asType(.float32); activations.eval()
        var outputs: [MLXArray] = []
        var logicalBytes: UInt64 = 0
        var ioSeconds = 0.0, copySeconds = 0.0, computeSeconds = 0.0
        var pending: [MLXArray] = [], pendingBytes = 0, peakBatchBytes = 0, batches = 0
        func flush() {
            guard !pending.isEmpty else { return }
            let begin = MonotonicClock.now()
            MLX.eval(pending)
            batches += 1
            computeSeconds += MonotonicClock.seconds(since: begin)
            pending.removeAll(keepingCapacity: true); pendingBytes = 0
        }
        for tile in 0..<count {
            let first = tile * tileRows, n = min(tileRows, rows - first)
            let packedBytes = n * Int(tensor.rowBytes) * experts.count
            if pendingBytes + packedBytes > 16 << 20 { flush() }
            let regions = try experts.map { expert in
                try tensor.region(sourceID: sourceID,
                    regionID: (expert << 32) | (UInt64(1) << 31) | UInt64(tile),
                    payloadOffset: payloadOffset, rows: UInt64(first)..<UInt64(first+n), expert: expert)
            }
            // Schedule only the selected experts, in consumption order. The
            // reader's fixed read-ahead window bounds filled/in-flight slots.
            try reader.prefetch(regions)
            var packed: [MLXArray] = []
            for region in regions {
                let ioStart = MonotonicClock.now()
                let lease = try reader.acquire(region)
                ioSeconds += MonotonicClock.seconds(since: ioStart)
                let copyStart = MonotonicClock.now()
                do {
                    defer { lease.release() }
                    packed.append(MLXArray(lease.contents.bindMemory(to: UInt8.self), [region.length]))
                }
                copySeconds += MonotonicClock.seconds(since: copyStart)
                logicalBytes += UInt64(region.length)
            }
            let computeStart = MonotonicClock.now()
            let output = try GGUFPackedLinear.run(activations, packed: concatenated(packed),
                type: tensor.quantization, rows: n, columns: columns, stackedWeights: true)
            computeSeconds += MonotonicClock.seconds(since: computeStart)
            outputs.append(output); pending.append(output); pendingBytes += packedBytes
            peakBatchBytes = max(peakBatchBytes, pendingBytes)
            if evaluation == .perTile { flush() }
        }
        flush()
        let output = outputs.count == 1 ? outputs[0] : concatenated(outputs, axis: -1)
        output.eval()
        return (output, GGUFLinearReport(tiles: count * experts.count, logicalWeightBytes: logicalBytes,
            scratchBytes: 0, peakCopiedWeightBytes: peakBatchBytes * 2,
            boundedOutputBytes: rows * experts.count * 4 * 2,
            wallSeconds: MonotonicClock.seconds(since: started), waitOnIOSeconds: ioSeconds,
            dequantizeAndCopySeconds: copySeconds, computeSeconds: computeSeconds,
            execution: .packedMetal, packedWeightCopyBytes: logicalBytes, residentWeightBytesReused: 0,
            evaluationBatches: batches))
    }

    /// Compute x W^T for a matrix or one selected expert. The reader must own
    /// an authorised descriptor and a pool sized for a packed row tile. IDs
    /// must be stable and unique per (tensor, expert, tile geometry) within
    /// that reader's lifetime. No pathname is opened here.
    public static func run(
        _ x: MLXArray, tensor: GGUFTensor, expert: UInt64? = nil,
        payloadOffset: UInt64, sourceID: UInt64, firstRegionID: UInt64,
        tileRows: Int, reader: TileReader, scratch: GGUFLinearScratch,
        maximumOutputBytes: Int = 16 * 1024 * 1024,
        execution: GGUFLinearExecution = .packedMetal,
        packedCache: GGUFPackedTileCache? = nil,
        evaluation: GGUFLinearEvaluation = .boundedBatch
    ) throws -> (output: MLXArray, report: GGUFLinearReport) {
        guard tensor.dimensions.count == 2 || tensor.dimensions.count == 3,
              let columns = Int(exactly: tensor.dimensions[0]),
              let rows = Int(exactly: tensor.dimensions[1]),
              x.ndim == 2, x.shape[0] > 0, x.shape[1] == columns,
              tileRows > 0, maximumOutputBytes > 0, maximumOutputBytes <= 64 * 1024 * 1024 else {
            throw GGUFError.invalid("streamed linear shape or output budget")
        }
        let tileElements = min(rows, tileRows).multipliedReportingOverflow(by: columns)
        let outputElements = rows.multipliedReportingOverflow(by: x.shape[0])
        guard !tileElements.overflow,
              execution == .packedMetal || tileElements.partialValue <= scratch.elements,
              execution != .packedMetal || tensor.rowBytes <= UInt64((4 << 20) / min(rows, tileRows)),
              !outputElements.overflow, outputElements.partialValue <= maximumOutputBytes / 4 else {
            throw GGUFError.limit("linear scratch or output")
        }
        let tileCount = (rows - 1) / tileRows + 1
        guard tileCount <= 4096, firstRegionID <= UInt64.max - UInt64(tileCount - 1) else {
            throw GGUFError.limit("linear tile count/identity")
        }
        // Validate expert selection and range before scheduling I/O.
        _ = try tensor.region(sourceID: sourceID, regionID: firstRegionID,
                              payloadOffset: payloadOffset, rows: 0..<UInt64(min(rows, tileRows)), expert: expert)
        scratch.lock.lock()
        defer { scratch.lock.unlock() }
        let started = MonotonicClock.now()
        let activations = x.asType(.float32)
        activations.eval()
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(tileCount)
        var logicalBytes: UInt64 = 0
        var copiedBytes: UInt64 = 0, reusedBytes: UInt64 = 0
        var waitSeconds = 0.0, decodeSeconds = 0.0, computeSeconds = 0.0
        var pending: [MLXArray] = [], pendingBytes = 0, peakBatchBytes = 0, batches = 0
        func flush() {
            guard !pending.isEmpty else { return }
            let begin = MonotonicClock.now()
            MLX.eval(pending)
            batches += 1
            computeSeconds += MonotonicClock.seconds(since: begin)
            pending.removeAll(keepingCapacity: true); pendingBytes = 0
        }
        for tile in 0..<tileCount {
            let batchTileBytes = min(tileRows, rows - tile * tileRows) * Int(tensor.rowBytes)
            if execution == .packedMetal, pendingBytes + batchTileBytes > 16 << 20 { flush() }
            // Overlap cold reads with the current GPU tile. Resident weights
            // need no raw-slot prefetch and must not generate duplicate I/O.
            if execution == .packedMetal {
                var upcoming: [RegionDescriptor] = []
                for ahead in tile..<min(tileCount, tile + reader.configuration.readAhead) {
                    let first = ahead * tileRows, n = min(tileRows, rows - first)
                    let region = try tensor.region(sourceID: sourceID, regionID: firstRegionID + UInt64(ahead),
                        payloadOffset: payloadOffset, rows: UInt64(first)..<UInt64(first+n), expert: expert)
                    if try packedCache?.contains(region) != true { upcoming.append(region) }
                }
                try reader.prefetch(upcoming)
            }
            let start = tile * tileRows, count = min(tileRows, rows - start)
            let region = try tensor.region(sourceID: sourceID, regionID: firstRegionID + UInt64(tile),
                payloadOffset: payloadOffset, rows: UInt64(start)..<UInt64(start + count), expert: expert)
            let weights: MLXArray
            if execution == .packedMetal, let cached = try packedCache?.lookup(region) {
                weights = cached; reusedBytes += UInt64(region.length)
            } else {
                let ioStart = MonotonicClock.now()
                let lease = try reader.acquire(region)
                waitSeconds += MonotonicClock.seconds(since: ioStart)
                let decodeStart = MonotonicClock.now()
                do {
                    defer { lease.release() }
                    if execution == .packedMetal {
                        weights = MLXArray(lease.contents.bindMemory(to: UInt8.self), [region.length])
                        try packedCache?.insert(weights, region: region)
                        copiedBytes += UInt64(region.length)
                    } else {
                        let pointer = scratch.buffer.pointer.assumingMemoryBound(to: Float.self)
                        let decoded = UnsafeMutableBufferPointer(start: pointer, count: count * columns)
                        try GGUFDequantizer.decode(lease.contents, type: tensor.quantization,
                                                   elements: decoded.count, into: decoded)
                        weights = MLXArray(UnsafeBufferPointer(decoded), [count, columns])
                    }
                }
                decodeSeconds += MonotonicClock.seconds(since: decodeStart)
            }
            let computeStart = MonotonicClock.now()
            let output: MLXArray
            if execution == .packedMetal {
                output = try GGUFPackedLinear.run(activations, packed: weights, type: tensor.quantization,
                    rows: count, columns: columns)
            } else { output = matmul(activations, weights.T) }
            if execution == .dequantized { output.eval(); batches += 1 }
            computeSeconds += MonotonicClock.seconds(since: computeStart)
            outputs.append(output)
            if execution == .packedMetal {
                pending.append(output); pendingBytes += region.length
                peakBatchBytes = max(peakBatchBytes, pendingBytes)
                if evaluation == .perTile { flush() }
            }
            logicalBytes += UInt64(region.length)
        }
        flush() // at most 16 MiB of packed tile operands await evaluation
        let output = outputs.count == 1 ? outputs[0] : concatenated(outputs, axis: -1)
        output.eval()
        return (output, GGUFLinearReport(tiles: tileCount, logicalWeightBytes: logicalBytes,
            scratchBytes: scratch.byteCount,
            peakCopiedWeightBytes: execution == .packedMetal
                ? peakBatchBytes
                : tileElements.partialValue * 4,
            boundedOutputBytes: outputElements.partialValue * 4 * 2,
            wallSeconds: MonotonicClock.seconds(since: started), waitOnIOSeconds: waitSeconds,
            dequantizeAndCopySeconds: decodeSeconds, computeSeconds: computeSeconds,
            execution: execution, packedWeightCopyBytes: copiedBytes, residentWeightBytesReused: reusedBytes,
            evaluationBatches: batches))
    }
}
