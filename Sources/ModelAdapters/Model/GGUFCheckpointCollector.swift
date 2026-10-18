import Foundation
import GGUFCore

/// Ephemeral diagnostic values, deliberately not Codable. The exported LES
/// report contains only scalar comparisons of the fixed numerical test.
public struct GGUFCheckpoint: Sendable {
    public let stage: UInt32
    public let layer: Int32
    public let values: [Float]
    public init(stage: UInt32, layer: Int32, values: [Float]) {
        self.stage = stage; self.layer = layer; self.values = values
    }
    public var key: UInt64 { UInt64(stage) << 32 | UInt64(UInt32(bitPattern: layer)) }
    public var order: Int {
        if stage == 1 { return -1 }
        if stage == 9 { return 10000 }
        let rank = stage == 10 ? 5 : stage >= 5 ? Int(stage) + 1 : Int(stage)
        return (Int(layer) + 1) * 16 + rank
    }
}

/// One serial forward at a time; the lock also protects the C callback boundary.
/// At most 256 points / 4 MiB of payload per forward; the replay owner separately
/// caps retained payload at 64 MiB. Failure never permits a partial trace to pass.
public final class GGUFCheckpointCollector: @unchecked Sendable {
    private let lock = NSLock()
    private var points: [UInt64: GGUFCheckpoint] = [:]
    private var bytes = 0
    private var failed = false
    public init() {}
    public func capture(stage: UInt32, layer: Int32, values: UnsafePointer<Float>?, count: Int32) -> Bool {
        lock.lock(); defer { lock.unlock() }
        guard !failed, (1...10).contains(stage),
              (stage == 10 ? layer == 0 : ((stage == 1 || stage == 9) ? layer == -1 : (0..<40).contains(layer))),
              count > 0, count <= 16384, let values,
              points.count < 256, bytes <= (4 << 20) - Int(count) * 4 else {
            failed = true; return false
        }
        let point = GGUFCheckpoint(stage: stage, layer: layer, values: Array(UnsafeBufferPointer(start: values, count: Int(count))))
        guard points[point.key] == nil, point.values.allSatisfy(\.isFinite) else { failed = true; return false }
        points[point.key] = point; bytes += point.values.count * 4
        return true
    }
    public func take() throws -> [GGUFCheckpoint] {
        lock.lock(); defer { lock.unlock() }
        guard !failed, !points.isEmpty else { throw GGUFError.invalid("Incomplete bounded layer checkpoint trace") }
        let result = points.values.sorted { $0.order < $1.order }
        points.removeAll(keepingCapacity: true); bytes = 0
        return result
    }
}
