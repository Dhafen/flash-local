/// Typed text messages consumed by the canonical llama.cpp/Jinja bridge.
/// Tool calls, images and reasoning history require a separate contract.
public struct Qwen35ChatMessage: Equatable, Sendable, Codable {
    public enum Role: String, Sendable, Codable { case system, user, assistant }
    public let role: Role
    public let content: String
    public init(role: Role, content: String) { self.role = role; self.content = content }
}
