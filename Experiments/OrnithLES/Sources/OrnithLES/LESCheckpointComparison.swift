import Foundation
import GGUFCore
import ModelAdapters

public struct LESCheckpointDifference: Codable, Sendable, Equatable {
    public let stage: UInt32
    public let layer: Int32
    public let positions: Int
    public let maximumAbsoluteError: Double
    public let relativeL2Error: Double
}
public struct LESCheckpointDivergence: Codable, Sendable, Equatable {
    public let position: Int
    public let stage: UInt32
    public let layer: Int32
    public let maximumAbsoluteError: Double
    public let relativeL2Error: Double
}
public struct LESCheckpointComparison: Codable, Sendable, Equatable {
    public let comparedPoints: Int
    public let missingPoints: Int
    public let shapeMismatches: Int
    public let maximumAbsoluteError: Double
    public let relativeL2Error: Double
    public let firstDivergence: LESCheckpointDivergence?
    public let layers: [LESCheckpointDifference]
    public var complete: Bool { comparedPoints > 0 && missingPoints == 0 && shapeMismatches == 0 }
}

// Diagnostic only: never weakens or replaces the full-vocabulary/routing gate.
struct LESCheckpointAccumulator {
    private struct Sum {
        var stage: UInt32
        var layer: Int32
        var positions = 0
        var absolute = 0.0, squared = 0.0, reference = 0.0
    }
    private var sums: [UInt64: Sum] = [:]
    private var compared = 0, missing = 0, mismatched = 0
    private var first: LESCheckpointDivergence?
    mutating func compare(_ expected: [GGUFCheckpoint], _ actual: [GGUFCheckpoint], position: Int) throws {
        var candidate: [UInt64: GGUFCheckpoint] = [:]
        for point in actual {
            guard candidate.updateValue(point, forKey: point.key) == nil else { throw GGUFError.invalid("Duplicate layer checkpoint") }
        }
        var seen = Set<UInt64>()
        for point in expected.sorted(by: { $0.order < $1.order }) {
            guard seen.insert(point.key).inserted else { throw GGUFError.invalid("Duplicate reference checkpoint") }
            guard let other = candidate.removeValue(forKey: point.key) else { missing += 1; continue }
            guard !point.values.isEmpty, point.values.count == other.values.count else { mismatched += 1; continue }
            var absolute = 0.0, squared = 0.0, reference = 0.0
            for (a, b) in zip(point.values, other.values) {
                guard a.isFinite, b.isFinite else { throw GGUFError.invalid("Non-finite layer checkpoint") }
                let d = Double(a) - Double(b)
                absolute = max(absolute, abs(d)); squared += d * d; reference += Double(a) * Double(a)
            }
            let relative = sqrt(squared / max(reference, 1e-30))
            if first == nil && (absolute > 0.01 || relative > 0.001) {
                first = LESCheckpointDivergence(position: position, stage: point.stage, layer: point.layer,
                    maximumAbsoluteError: absolute, relativeL2Error: relative)
            }
            var sum = sums[point.key] ?? Sum(stage: point.stage, layer: point.layer)
            sum.positions += 1; sum.absolute = max(sum.absolute, absolute)
            sum.squared += squared; sum.reference += reference
            sums[point.key] = sum; compared += 1
        }
        missing += candidate.count
    }
    var result: LESCheckpointComparison {
        let layers = sums.values.map {
            LESCheckpointDifference(stage: $0.stage, layer: $0.layer, positions: $0.positions,
                maximumAbsoluteError: $0.absolute, relativeL2Error: sqrt($0.squared / max($0.reference, 1e-30)))
        }.sorted { $0.layer == $1.layer ? $0.stage < $1.stage : $0.layer < $1.layer }
        return LESCheckpointComparison(comparedPoints: compared, missingPoints: missing, shapeMismatches: mismatched,
            maximumAbsoluteError: sums.values.map(\.absolute).max() ?? 0,
            relativeL2Error: sqrt(sums.values.reduce(0) { $0 + $1.squared } / max(sums.values.reduce(0) { $0 + $1.reference }, 1e-30)),
            firstDivergence: first, layers: layers)
    }
}
