import Foundation

public enum PerformanceKernelMode: String, Codable, Sendable {
    case original, groupedExperts
}

/// A bounded comparison at four positions, not a full model-quality claim.
/// No prompt, token IDs or logit vectors are exported.
public struct PerformanceKernelCheck: Codable, Sendable {
    public let schemaVersion = 1
    public let runtimeRevision = OrnithPerformanceEngine.swiftletRuntimeRevision
    public let fullModelNumericsVerified = false
    public let sourceIdentity: String
    public let promptTokenCount: Int
    public let decodeSteps: Int
    public let comparedLogitVectors: Int
    public let mismatchedLogitValues: Int
    public let maximumAbsoluteLogitDifference: Double
    public let greedyChoicesMatch: Bool
    public let passed: Bool
    public let elapsedSeconds: Double
    public let maximumObservedFootprintBytes: UInt64
}

/// Synchronous, nonescaping callbacks keep the engine's actor ownership for the
/// entire check. One model is replayed twice; no second dense core/cache exists.
enum PerformanceKernelProbe {
    static func run(
        prompt: [Int], sourceIdentity: String, decodeSteps: Int = 3,
        shouldCancel: () -> Bool,
        resetMode: (PerformanceKernelMode) -> Void,
        forward: ([Int], PerformanceForwardMetrics.Phase) throws -> PerformanceForwardResult
    ) throws -> PerformanceKernelCheck {
        resetMode(.original)
        defer { resetMode(.original) }
        guard (1...32).contains(prompt.count), (1...3).contains(decodeSteps) else {
            throw OrnithPerformanceError.invalidSource("kernel check exceeds its bounded input")
        }
        func checkCancellation() throws {
            if shouldCancel() { throw OrnithPerformanceError.cancelled }
        }
        let started = ProcessInfo.processInfo.systemUptime
        var peak: UInt64 = 0
        func step(_ tokens: [Int], _ phase: PerformanceForwardMetrics.Phase) throws -> [Float] {
            try checkCancellation()
            let value = try forward(tokens, phase)
            guard value.metrics.completed, !value.logits.isEmpty,
                  value.logits.count <= 1_000_000, value.logits.allSatisfy(\.isFinite) else {
                throw OrnithPerformanceError.integrity("invalid kernel-check output")
            }
            peak = max(peak, value.metrics.sampledPeak.maximumObservedFootprintBytes)
            return value.logits
        }
        // At most four vocabulary vectors (~4 MB for Ornith), plus the current
        // vector. They fit inside the existing runtime/scratch reservation.
        var references = [try step(prompt, .prefill)]
        var teacherTokens: [Int] = []
        for index in 0..<decodeSteps {
            let token = try OrnithPerformanceSession.greedy(references[index])
            teacherTokens.append(token)
            references.append(try step([token], .decode))
        }
        resetMode(.groupedExperts)
        var mismatches = 0
        var maximumDifference = 0.0
        var choicesMatch = true
        for index in references.indices {
            let actual = try step(index == 0 ? prompt : [teacherTokens[index - 1]],
                                  index == 0 ? .prefill : .decode)
            let expected = references[index]
            guard actual.count == expected.count else {
                throw OrnithPerformanceError.integrity("kernel-check vocabulary changed")
            }
            let actualChoice = try OrnithPerformanceSession.greedy(actual)
            let expectedChoice = try OrnithPerformanceSession.greedy(expected)
            choicesMatch = choicesMatch && actualChoice == expectedChoice
            for row in expected.indices {
                if actual[row].bitPattern != expected[row].bitPattern { mismatches += 1 }
                maximumDifference = max(maximumDifference, abs(Double(actual[row]) - Double(expected[row])))
            }
        }
        try checkCancellation()
        return .init(sourceIdentity: sourceIdentity, promptTokenCount: prompt.count,
            decodeSteps: decodeSteps, comparedLogitVectors: references.count,
            mismatchedLogitValues: mismatches, maximumAbsoluteLogitDifference: maximumDifference,
            greedyChoicesMatch: choicesMatch, passed: mismatches == 0 && choicesMatch,
            elapsedSeconds: max(0, ProcessInfo.processInfo.systemUptime - started),
            maximumObservedFootprintBytes: peak)
    }
}
