import Foundation

/// Typed text messages consumed by the canonical llama.cpp/Jinja bridge.
/// Optional assistant reasoning stays separate from answer text. Tool calls
/// and images require a separate contract.
public struct Qwen35ChatMessage: Equatable, Sendable, Codable {
    /// An explicit API/system message takes precedence over the app default.
    /// The canonical tokenizer still validates the complete role sequence.
    public static func applyingSystemPrompt(_ prompt: String, to messages: [Self]) -> [Self] {
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              messages.first?.role != .system else { return messages }
        return [.init(role: .system, content: prompt)] + messages
    }
    public enum Role: String, Sendable, Codable { case system, user, assistant }
    public let role: Role
    public let content: String
    public let reasoningContent: String?
    private enum CodingKeys: String, CodingKey { case role, content; case reasoningContent = "reasoning_content" }
    public init(role: Role, content: String, reasoningContent: String? = nil) {
        self.role = role; self.content = content; self.reasoningContent = reasoningContent
    }
}
