import Foundation
import Tokenizers

public struct OrnithChatMessage: Codable, Equatable, Sendable {
    public enum Role: String, Codable, Sendable { case system, user, assistant }
    public let role: Role
    public let content: String
    public init(role: Role, content: String) { self.role = role; self.content = content }
}

/// Local tokenizer and the publisher's literal Jinja template. No hand-written
/// ChatML rendering, prompt truncation, hidden system message or hub fallback.
public struct OrnithChatCodec: Sendable {
    private let tokenizer: any Tokenizer
    private let template: String
    public let endTokenIDs: Set<Int>

    public init(localDirectory: URL, manifest: MLXSourceManifest) async throws {
        guard let configData = Data(base64Encoded: manifest.configBase64),
              let config = try JSONSerialization.jsonObject(with: configData) as? [String: Any] else {
            throw OrnithPerformanceError.invalidSource("invalid tokenizer config contract")
        }
        for name in ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"] {
            guard let descriptor = manifest.auxiliary.first(where: { $0.name == name }),
                  descriptor.byteCount > 0, descriptor.byteCount <= 32 << 20 else {
                throw OrnithPerformanceError.invalidSource("missing pinned tokenizer metadata")
            }
            let file = try FileHandle(forReadingFrom: localDirectory.appendingPathComponent(name))
            let data = try file.read(upToCount: descriptor.byteCount + 1) ?? Data()
            try file.close()
            guard data.count == descriptor.byteCount, digest(data) == descriptor.sha256 else {
                throw OrnithPerformanceError.integrity("tokenizer metadata differs: \(name)")
            }
        }
        template = try String(contentsOf: localDirectory.appendingPathComponent("chat_template.jinja"), encoding: .utf8)
        tokenizer = try await AutoTokenizer.from(modelFolder: localDirectory)
        if let values = config["eos_token_id"] as? [Int], !values.isEmpty { endTokenIDs = Set(values) }
        else if let value = config["eos_token_id"] as? Int { endTokenIDs = [value] }
        else { throw OrnithPerformanceError.invalidSource("missing published end tokens") }
    }

    public func promptTokens(messages: [OrnithChatMessage], thinking: Bool = false) throws -> [Int] {
        try Self.validate(messages)
        let messages: [Tokenizers.Message] = messages.map { ["role": $0.role.rawValue, "content": $0.content] }
        return try tokenizer.applyChatTemplate(messages: messages, chatTemplate: .literal(template),
            addGenerationPrompt: true, truncation: false, maxLength: nil, tools: nil,
            additionalContext: ["enable_thinking": thinking])
    }

    public func encode(_ text: String) -> [Int] { tokenizer.encode(text: text, addSpecialTokens: false) }
    public func decode(_ tokens: [Int]) -> String { tokenizer.decode(tokens: tokens, skipSpecialTokens: false) }

    static func validate(_ messages: [OrnithChatMessage]) throws {
        guard !messages.isEmpty else { throw OrnithPerformanceError.invalidSource("empty conversation") }
        var expected = OrnithChatMessage.Role.user
        for (index, message) in messages.enumerated() {
            if index == 0 && message.role == .system { continue }
            guard message.role == expected, !message.content.isEmpty else {
                throw OrnithPerformanceError.invalidSource("chat roles must alternate with nonempty turns")
            }
            expected = expected == .user ? .assistant : .user
        }
        guard messages.last?.role == .user else {
            throw OrnithPerformanceError.invalidSource("generation requires a final user message")
        }
    }
}
