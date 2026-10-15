import CryptoKit
import Foundation

public struct Qwen35ChatMessage: Equatable, Sendable, Codable {
    public enum Role: String, Sendable, Codable { case system, user, assistant }
    public let role: Role
    public let content: String
    public init(role: Role, content: String) { self.role = role; self.content = content }
}

/// Experimental parity probe; the product will render through canonical Jinja.
/// Text-only, completed-turn chat with enable_thinking=false and
/// preserve_thinking=false. The primary artifact's template is hash-bound;
/// tool calls, reasoning history and multimodal content need a separate path.
public struct Qwen35NoThinkingChat: Sendable {
    public static let templateSHA256 = "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
    public let messages: [Qwen35ChatMessage]
    private enum Segment { case text(String), control(String) }

    public init(messages: [Qwen35ChatMessage]) throws {
        guard !messages.isEmpty, messages.count <= 4096,
              messages.last?.role == .user else {
            throw Qwen35Vocabulary.Failure.invalid("Chat needs a final user turn and at most 4096 messages")
        }
        var expected = Qwen35ChatMessage.Role.user, bytes = 0
        for (i, message) in messages.enumerated() {
            bytes += message.content.utf8.count
            guard bytes <= 262_144 else { throw Qwen35Vocabulary.Failure.limit("Chat text exceeds 256 KiB") }
            if i == 0, message.role == .system { continue }
            guard message.role == expected else {
                throw Qwen35Vocabulary.Failure.invalid("Chat roles must alternate after the optional initial system message")
            }
            expected = expected == .user ? .assistant : .user
        }
        self.messages = messages
    }

    public func tokens(using vocabulary: Qwen35Vocabulary, maximumTokens: Int) throws -> [Int] {
        guard let template = vocabulary.chatTemplate,
              SHA256.hash(data: Data(template.utf8)).map({ String(format: "%02x", $0) }).joined() == Self.templateSHA256 else {
            throw Qwen35Vocabulary.Failure.invalid("The verified GGUF chat template differs from the supported text-chat template")
        }
        guard maximumTokens > 0, maximumTokens <= 131_072 else {
            throw Qwen35Vocabulary.Failure.limit("Invalid chat context token budget")
        }
        var result: [Int] = []
        for segment in segments {
            switch segment {
            case .text(let text): result.append(contentsOf: try vocabulary.encode(text))
            case .control(let text): result.append(try vocabulary.controlID(text))
            }
            guard result.count <= maximumTokens else {
                throw Qwen35Vocabulary.Failure.limit("The chat prompt exceeds the admitted context")
            }
        }
        return result
    }

    /// Diagnostic/reference text; production tokenization uses typed controls.
    public var renderedText: String {
        segments.map { switch $0 { case .text(let s), .control(let s): return s } }.joined()
    }

    private var segments: [Segment] {
        var result: [Segment] = []
        for message in messages {
            let content = Self.trimLikePinnedReference(message.content)
            if message.role == .system, content.isEmpty { continue }
            result += [.control("<|im_start|>"), .text(message.role.rawValue + "\n" + content),
                       .control("<|im_end|>"), .text("\n")]
        }
        result += [.control("<|im_start|>"), .text("assistant\n"),
                   .control("<think>"), .text("\n\n"), .control("</think>"), .text("\n\n")]
        return result
    }

    // Pinned llama.cpp's Jinja `trim` uses C-locale byte whitespace rather
    // than Foundation's broader Unicode whitespace set. Preserve that exact
    // reference behavior, including NBSP/combining characters at the edges.
    private static func trimLikePinnedReference(_ text: String) -> String {
        func whitespace(_ c: Unicode.Scalar) -> Bool { c.value == 32 || (9...13).contains(c.value) }
        let scalars = text.unicodeScalars
        var first = scalars.startIndex, last = scalars.endIndex
        while first < last, whitespace(scalars[first]) { first = scalars.index(after: first) }
        while first < last, whitespace(scalars[scalars.index(before: last)]) { last = scalars.index(before: last) }
        return String(scalars[first..<last])
    }
}
