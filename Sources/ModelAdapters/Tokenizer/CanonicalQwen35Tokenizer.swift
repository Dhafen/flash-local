import CryptoKit
import Darwin
import Foundation
import GGUFCore
import LlamaTokenizerRuntime
import MinirunKit

/// The product tokenizer and chat renderer. All BPE, Unicode splitting, token
/// pieces, stop IDs and Jinja execution delegate to pinned llama.cpp. Confine
/// this object to the runner's single inference executor. This API only loads
/// vocabulary metadata and never creates an inference context. The runtime's
/// separate flr_* CPU diagnostic API is not invoked by this tokenizer.
public final class CanonicalQwen35Tokenizer {
    public static let supportedTemplateSHA256 = "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
    public static let supportedOrnithTemplateSHA256 = "f55f52930aa8bf44ab5cb85f99370fcc3c56e9a85640b812086d5330bce5d86b"
    public struct ChatPrompt: Equatable, Sendable {
        public let renderedText: String
        public let tokenIDs: [Int]
    }
    private let artifact: VerifiedGGUFArtifact
    private let handle: OpaquePointer
    public let vocabularySize: Int
    public let eos: Int
    public let endOfTurn: Int

    public init(artifact: VerifiedGGUFArtifact) throws {
        guard artifact.index.metadata["tokenizer.ggml.pre"]?.text == "qwen35",
              artifact.index.metadata["tokenizer.ggml.model"]?.text == "gpt2" else {
            throw GGUFError.unsupported("canonical Qwen35 tokenizer metadata")
        }
        let descriptor = try artifact.duplicateForReading()
        defer { _ = Darwin.close(descriptor) }
        guard let handle = flt_load_vocabulary(descriptor) else { throw Self.failure() }
        var transferred = false
        defer { if !transferred { flt_free_vocabulary(handle) } }
        let count = Int(flt_vocabulary_size(handle)), eos = Int(flt_eos(handle)), eot = Int(flt_eot(handle))
        guard (256...300000).contains(count), (0..<count).contains(eos), (0..<count).contains(eot),
              let templatePointer = flt_chat_template(handle),
              let template = artifact.index.metadata["tokenizer.chat_template"]?.text,
              String(cString: templatePointer) == template else {
            throw GGUFError.invalid("canonical vocabulary controls/template")
        }
        try artifact.assertUnchanged()
        self.artifact = artifact; self.handle = handle
        self.vocabularySize = count; self.eos = eos; self.endOfTurn = eot
        transferred = true
    }

    deinit { flt_free_vocabulary(handle) }

    public func isEndOfGeneration(_ token: Int) -> Bool {
        guard (0..<vocabularySize).contains(token) else { return false }
        return flt_is_eog(handle, Int32(token))
    }

    /// Exact raw reference tokenization: no implicit BOS/EOS. A caller needing
    /// chat should use `chat`, which validates typed messages and markup.
    public func tokenize(_ text: String, parseSpecial: Bool = false,
                         maximumTokens: Int = 131072) throws -> [Int] {
        guard text.utf8.count <= 262144 else { throw GGUFError.limit("tokenizer input exceeds 256 KiB") }
        return try tokenizeBounded(text, parseSpecial: parseSpecial, maximumTokens: maximumTokens)
    }

    public func bytes(for token: Int, includeControl: Bool = false) throws -> Data {
        try artifact.assertUnchanged()
        guard (0..<vocabularySize).contains(token) else { throw GGUFError.invalid("output token id") }
        let size = flt_token_piece(handle, Int32(token), includeControl, nil, 0)
        guard size != Int32.min else { throw Self.failure() }
        let count = size < 0 ? -Int(size) : Int(size)
        guard count <= 16384 else { throw GGUFError.limit("token piece exceeds 16 KiB") }
        if count == 0 { return Data() }
        var bytes = [CChar](repeating: 0, count: count)
        let written = bytes.withUnsafeMutableBufferPointer {
            flt_token_piece(handle, Int32(token), includeControl, $0.baseAddress, Int32($0.count))
        }
        guard Int(written) == count else { throw Self.failure() }
        try artifact.assertUnchanged()
        return Data(bytes.map { UInt8(bitPattern: $0) })
    }

    public func chat(_ messages: [Qwen35ChatMessage], maximumTokens: Int,
                     enableThinking: Bool = false) throws -> ChatPrompt {
        try artifact.assertUnchanged()
        let expectedTemplate = artifact.index.metadata["general.architecture"]?.text == "qwen35moe"
            ? Self.supportedOrnithTemplateSHA256 : Self.supportedTemplateSHA256
        guard let template = artifact.index.metadata["tokenizer.chat_template"]?.text,
              SHA256.hash(data: Data(template.utf8)).map({ String(format: "%02x", $0) }).joined() == expectedTemplate else {
            throw GGUFError.unsupported("GGUF chat template differs from the pinned text-chat contract")
        }
        guard !messages.isEmpty, messages.count <= 4096 else { throw GGUFError.limit("chat message count") }
        var contentBytes = 0
        for message in messages {
            contentBytes += message.content.utf8.count
            if let reasoning = message.reasoningContent {
                guard message.role == .assistant else { throw GGUFError.invalid("Only assistant messages may contain reasoning") }
                contentBytes += reasoning.utf8.count
            }
            guard contentBytes <= 262144 else { throw GGUFError.limit("chat content exceeds 256 KiB") }
        }
        let json = try JSONEncoder().encode(messages)
        guard json.count <= 1048576 else { throw GGUFError.limit("chat JSON exceeds 1 MiB") }
        let size = json.withUnsafeBytes {
            flt_render_chat_with_thinking(handle, $0.bindMemory(to: CChar.self).baseAddress,
                                         Int32(json.count), enableThinking, nil, 0)
        }
        guard size < 0, size != Int32.min, -Int(size) <= 524288 else { throw Self.failure() }
        var rendered = [CChar](repeating: 0, count: -Int(size))
        let written = json.withUnsafeBytes { input in
            rendered.withUnsafeMutableBufferPointer { output in
                flt_render_chat_with_thinking(handle, input.bindMemory(to: CChar.self).baseAddress,
                    Int32(json.count), enableThinking, output.baseAddress, Int32(output.count))
            }
        }
        guard Int(written) == rendered.count else { throw Self.failure() }
        let text = String(decoding: rendered.map { UInt8(bitPattern: $0) }, as: UTF8.self)
        let ids = try tokenizeBounded(text, parseSpecial: true, maximumTokens: maximumTokens)
        try artifact.assertUnchanged()
        return ChatPrompt(renderedText: text, tokenIDs: ids)
    }

    private func tokenizeBounded(_ text: String, parseSpecial: Bool, maximumTokens: Int) throws -> [Int] {
        try artifact.assertUnchanged()
        guard (1...131072).contains(maximumTokens), text.utf8.count <= 524288 else {
            throw GGUFError.limit("canonical tokenizer request")
        }
        let bytes = Int32(text.utf8.count)
        let size = text.withCString { flt_encode(handle, $0, bytes, parseSpecial, nil, 0) }
        guard size != Int32.min else { throw Self.failure() }
        let count = size < 0 ? -Int(size) : Int(size)
        guard count <= maximumTokens else { throw GGUFError.limit("prompt exceeds admitted context tokens") }
        var tokens = [Int32](repeating: 0, count: count)
        let written = text.withCString { input in
            tokens.withUnsafeMutableBufferPointer { flt_encode(handle, input, bytes, parseSpecial, $0.baseAddress, Int32($0.count)) }
        }
        guard Int(written) == count, tokens.allSatisfy({ $0 >= 0 && Int($0) < vocabularySize }) else { throw Self.failure() }
        try artifact.assertUnchanged()
        return tokens.map(Int.init)
    }

    private static func failure() -> GGUFError {
        let message = flt_last_error().map { String(cString: $0) } ?? "canonical tokenizer failure"
        return .invalid(message.isEmpty ? "canonical tokenizer operation failed" : message)
    }
}
