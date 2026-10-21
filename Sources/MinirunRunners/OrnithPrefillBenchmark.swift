#if canImport(OrnithLES)
import Foundation
import OrnithLES
import MinirunKit
import ModelAdapters

public enum OrnithPrefillBenchmark {
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
            return prefill(tuned) >= prefill(base) * 1.05 && decode(tuned) >= decode(base) * 0.95 ? candidate : .init()
        }
    }
    public static func execute(artifact: VerifiedGGUFArtifact, prepared: URL, baseline: LESNumericalCheck,
                               candidate: LESPrefillConfiguration, check: LESPrefillCheck?, control: LESControl,
                               contextProfile: LocalContextProfile = .standard,
                               progress: @Sendable (String) -> Void,
                               onAdmission: (LESMemoryAdmission) -> Void) throws -> Report {
        var results: [Run] = []
        for (index, mode) in ["baseline", "candidate", "candidate", "baseline"].enumerated() {
            let summary = try autoreleasepool {
                let engine = try OrnithLESChatGeneration(artifact: artifact, prepared: prepared, check: baseline,
                    control: control, prefillConfiguration: mode == "baseline" ? .init() : candidate,
                    prefillCheck: mode == "baseline" ? nil : check, contextProfile: contextProfile, onAdmission: onAdmission)
                defer { engine.close() }
                var options = LocalGenerationOptions(); options.maximumNewTokens = 129
                return try engine.generate(messages: [.init(role: .user, content: LESBenchmark.prompt)], options: options) { event in
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
