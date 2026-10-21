#if canImport(OrnithLES)
import Foundation
import ModelAdapters
import MinirunKit

/// A real on-device, same-prompt warm/fresh comparison. No stored chat is used.
/// Resetting the app's state does not make the OS file cache cold.
public enum OrnithCacheBenchmark {
    public struct Report: Codable, Sendable {
        public let warm, fresh: OrnithLESChatGeneration.Summary
        public let seed: OrnithLESChatGeneration.Summary
        public let osFileCacheControlled = false
        public let fullModelNumericsVerified = false
        public var exactOutputsMatch: Bool {
            warm.promptTokenIDs == fresh.promptTokenIDs && warm.generatedTokenIDs == fresh.generatedTokenIDs
        }
        public var cacheVerified: Bool {
            exactOutputsMatch && warm.reusedPromptTokens > 0 && fresh.reusedPromptTokens == 0 &&
                warm.evaluatedPromptTokens + warm.reusedPromptTokens == warm.promptTokenIDs.count &&
                fresh.evaluatedPromptTokens == fresh.promptTokenIDs.count
        }
        public var observedPrefillSecondsDifference: Double { fresh.prefillSeconds - warm.prefillSeconds }
    }
    public static func execute(engine: OrnithLESChatGeneration, enableThinking: Bool = true,
                               progress: @Sendable (String) -> Void) throws -> Report {
        defer { engine.resetConversation() }
        engine.resetConversation()
        var options = LocalGenerationOptions()
        options.maximumNewTokens = 129; options.enableThinking = enableThinking; options.thinkingBudget = enableThinking ? 64 : nil
        let question = Qwen35ChatMessage(role: .user, content: "Write a numbered checklist of forty software cache correctness tests. Explain each briefly.")
        var text = QwenThinkingText(startsInThinking: enableThinking)
        let seed = try engine.generate(messages: [question], options: options) { event in
            switch event {
            case .prefill(let value): progress("Cachetest 1/3 · prompt \(value.promptTokensCompleted)/\(value.promptTokensTotal)")
            case .token(_, let delta, let value): text.append(delta); progress("Cachetest 1/3 · \(value.generatedTokens) token")
            case .trailingText(let delta): text.append(delta)
            }
        }
        text.finish()
        let history = [question, Qwen35ChatMessage(role: .assistant, content: text.answer, reasoningContent: text.reasoning),
            Qwen35ChatMessage(role: .user, content: "Continue with twenty more numbered tests, avoiding repetition.")]
        func run(_ phase: String) throws -> OrnithLESChatGeneration.Summary {
            try engine.generate(messages: history, options: options) { event in
                switch event {
                case .prefill(let value): progress("Cachetest \(phase) · \(value.reusedPromptTokens) återanvända · \(value.promptTokensCompleted)/\(value.promptTokensTotal)")
                case .token(_, _, let value): progress("Cachetest \(phase) · \(value.generatedTokens) token")
                case .trailingText: break
                }
            }
        }
        let warm = try run("2/3 varm")
        engine.resetConversation()
        let fresh = try run("3/3 ny kontext")
        return .init(warm: warm, fresh: fresh, seed: seed)
    }
}
#endif
