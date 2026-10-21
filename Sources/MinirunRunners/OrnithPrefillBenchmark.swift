#if canImport(OrnithLES)
import Foundation
import OrnithLES
import MinirunKit
import ModelAdapters

public enum OrnithPrefillBenchmark {
    public struct Measurement: Codable, Sendable {
        public let configuration: LESPrefillConfiguration
        public let contextTokens: Int
        public let comparable: Bool
        public let promptTokensPerSecond, decodeTokensPerSecond: Double?
        public let peakMemoryBytes: UInt64?
        public let maximumPhysicalBatchActuallyExecuted: Int?
        public let maximumEvaluationBatchActuallySubmitted: Int?
        public let minimumPromptTokensActuallyEvaluated: Int?
        public let failure: String?
        public init(configuration: LESPrefillConfiguration, contextTokens: Int, report: Report? = nil, failure: String? = nil) {
            self.configuration = configuration; self.contextTokens = contextTokens
            comparable = report?.comparable == true; self.failure = failure
            let rows = report?.runs.filter { $0.mode == "candidate" }.map(\.generation) ?? []
            promptTokensPerSecond = rows.isEmpty ? nil : Double(rows.reduce(0) { $0 + $1.evaluatedPromptTokens }) / max(rows.reduce(0) { $0 + $1.prefillSeconds }, 1e-9)
            decodeTokensPerSecond = rows.isEmpty ? nil : Double(rows.reduce(0) { $0 + $1.decodeForwardPasses }) / max(rows.reduce(0) { $0 + $1.decodeForwardSeconds }, 1e-9)
            peakMemoryBytes = rows.map(\.maximumObservedFootprintBytes).max()
            maximumPhysicalBatchActuallyExecuted = rows.compactMap { $0.metricsAfter.maximumPhysicalBatchTokens }.max()
            maximumEvaluationBatchActuallySubmitted = rows.map(\.maximumEvaluationBatchActuallySubmitted).max()
            minimumPromptTokensActuallyEvaluated = rows.map(\.evaluatedPromptTokens).min()
        }
    }
    public struct Run: Codable, Sendable {
        public let mode: String
        public let generation: OrnithLESChatGeneration.Summary
    }
    public struct Report: Codable, Sendable {
        public let promptID = "ornith-prefill-abba-v1"
        public let osCacheControlled = false
        public let fullModelNumericsVerified = false
        public let candidate: LESPrefillConfiguration
        public let check: LESPrefillCheck?
        public let runs: [Run]
        public var comparable: Bool {
            guard let first = runs.first else { return false }
            return runs.map(\.mode) == ["baseline", "candidate", "candidate", "baseline"] &&
                (candidate == LESPrefillConfiguration() || check?.passed == true) &&
                runs.allSatisfy { $0.generation.promptTokenIDs == first.generation.promptTokenIDs } &&
                Self.same128TokenOutputs(runs.map { $0.generation.generatedTokenIDs }) &&
                runs.allSatisfy { $0.generation.decodeForwardPasses >= 128 && !$0.generation.lowPowerModeAtStart &&
                    $0.generation.maximumThermalState == $0.generation.thermalStateAtStart &&
                    $0.generation.maximumObservedFootprintBytes <= $0.generation.memoryPlan.processCeilingBytes } &&
                Set(runs.map { $0.generation.thermalStateAtStart }).count == 1
        }
        public static func same128TokenOutputs(_ values: [[Int]]) -> Bool {
            guard values.count == 4, let first = values.first else { return false }
            return values.allSatisfy { $0.count >= 128 && $0 == first }
        }
        /// Promote only a measured, output-equivalent improvement without a
        /// decode regression. This is a profile for this device/context, not a
        /// claim that the largest number is best.
        public var preferredConfiguration: LESPrefillConfiguration? {
            guard comparable else { return nil }
            let base = runs.filter { $0.mode == "baseline" }.map(\.generation)
            let tuned = runs.filter { $0.mode == "candidate" }.map(\.generation)
            func prefill(_ values: [OrnithLESChatGeneration.Summary]) -> Double {
                Double(values.reduce(0) { $0 + $1.evaluatedPromptTokens }) / max(values.reduce(0) { $0 + $1.prefillSeconds }, 1e-9)
            }
            func decode(_ values: [OrnithLESChatGeneration.Summary]) -> Double {
                Double(values.reduce(0) { $0 + $1.decodeForwardPasses }) / max(values.reduce(0) { $0 + $1.decodeForwardSeconds }, 1e-9)
            }
            return Self.improves(promptBaseline: prefill(base), promptCandidate: prefill(tuned),
                decodeBaseline: decode(base), decodeCandidate: decode(tuned)) ? candidate : .init()
        }
        public static func improves(promptBaseline: Double, promptCandidate: Double,
                                    decodeBaseline: Double, decodeCandidate: Double) -> Bool {
            guard [promptBaseline, promptCandidate, decodeBaseline, decodeCandidate].allSatisfy({ $0.isFinite && $0 > 0 }) else { return false }
            return promptCandidate >= promptBaseline * 1.05 && decodeCandidate >= decodeBaseline * 0.95
        }
    }
    public static func execute(artifact: VerifiedGGUFArtifact, prepared: URL, baseline: LESNumericalCheck,
                               candidate: LESPrefillConfiguration, check: LESPrefillCheck?, control: LESControl,
                               contextProfile: LocalContextProfile = .standard,
                               prefixCheckpointAdmitted: Bool = false,
                               minimumPromptTokens: Int = 0,
                               progress: @Sendable (String) -> Void,
                               onAdmission: (LESMemoryAdmission) -> Void) throws -> Report {
        var results: [Run] = []
        let prompt: String = try autoreleasepool {
            var text = LESBenchmark.prompt
            guard minimumPromptTokens > 0 else { return text }
            guard minimumPromptTokens <= 4096 else { throw LocalAPIError.invalid("benchmark", "Benchmark prompt bound exceeded") }
            let tokenizer = try CanonicalQwen35Tokenizer(artifact: artifact)
            var line = 0
            while try tokenizer.chat([.init(role: .user, content: text)], maximumTokens: contextProfile.rawValue - 129,
                enableThinking: true).tokenIDs.count < minimumPromptTokens {
                line += 1
                guard line <= 256 else { throw LocalAPIError.invalid("benchmark", "Benchmark token target could not be reached") }
                text += "\nSynthetic cache test \(line): check exact prefix equality, changed system instructions, token accounting, bounded memory and cancellation."
            }
            return text
        }
        for (index, mode) in ["baseline", "candidate", "candidate", "baseline"].enumerated() {
            let summary = try autoreleasepool {
                let engine = try OrnithLESChatGeneration(artifact: artifact, prepared: prepared, check: baseline,
                    control: control, prefillConfiguration: mode == "baseline" ? .init() : candidate,
                    prefillCheck: mode == "baseline" ? nil : check, contextProfile: contextProfile,
                    prefixCheckpointAdmitted: prefixCheckpointAdmitted, onAdmission: onAdmission)
                defer { engine.close() }
                var options = LocalGenerationOptions(); options.maximumNewTokens = 129
                return try engine.generate(messages: [.init(role: .user, content: prompt)], options: options) { event in
                    switch event {
                    case .prefill(let value): progress("Batchtest \(index + 1)/4 · \(mode) · prompt \(value.promptTokensCompleted)/\(value.promptTokensTotal)")
                    case .token(_, _, let value): progress("Batchtest \(index + 1)/4 · \(mode) · \(value.generatedTokens)/129 token")
                    case .trailingText: break
                    }
                }
            }
            results.append(.init(mode: mode, generation: summary))
        }
        return .init(candidate: candidate, check: check, runs: results)
    }
}
#endif
