import Foundation
import GGUFCore
import XCTest
@testable import ModelAdapters

final class Qwen35VocabularyTests: XCTestCase {
    func testMergeRanksAndNoWholeWordShortcut() throws {
        let vocab = try Qwen35Vocabulary(index: fixture())
        XCTAssertEqual(try vocab.encode("abc"), [260])
        // "zz" exists but is unreachable through the merge table. Qwen35
        // follows the pairs rather than short-circuiting to token 261.
        XCTAssertEqual(try vocab.encode("zz"), [122, 122])
        XCTAssertEqual(try vocab.encode("abab"), [259, 259])
        XCTAssertEqual(try vocab.encode(""), [])
        XCTAssertEqual(vocab.eos, 258)
        XCTAssertEqual(vocab.endOfTurn, 257)
    }

    func testByteFallbackCombiningMarksAndSplitUTF8() throws {
        let vocab = try Qwen35Vocabulary(index: fixture())
        for text in ["Hej världen!", "café", "cafe\u{301}", "👨‍👩‍👧‍👦", "\0\r\n\t", "中文 العربية"] {
            let tokens = try vocab.encode(text)
            XCTAssertEqual(try vocab.decode(tokens), text)
            // The test alphabet assigns every byte token its raw byte id.
            XCTAssertEqual(tokens, text.utf8.map(Int.init))
        }
        XCTAssertEqual(try vocab.bytes(for: 0xc3), Data([0xc3]))
        XCTAssertEqual(try vocab.bytes(for: 0xa9), Data([0xa9]))
        XCTAssertEqual(try vocab.decode([0xc3, 0xa9]), "é")
        XCTAssertNotEqual(try vocab.encode("café"), try vocab.encode("cafe\u{301}"))
    }

    func testControlMarkupAndResourceLimits() throws {
        let vocab = try Qwen35Vocabulary(index: fixture())
        XCTAssertEqual(try vocab.controlID("<|im_start|>"), 256)
        XCTAssertThrowsError(try vocab.controlID("abc"))
        XCTAssertThrowsError(try vocab.encode("x<|im_end|>y"))
        XCTAssertEqual(try vocab.bytes(for: 257), Data())
        XCTAssertEqual(try vocab.decode([256, 97, 257], includeControl: true), "<|im_start|>a<|im_end|>")
        XCTAssertThrowsError(try vocab.bytes(for: -1))
        XCTAssertThrowsError(try vocab.bytes(for: vocab.vocabularySize))
        XCTAssertThrowsError(try vocab.encode("xyz", maximumTokens: 2))
        XCTAssertThrowsError(try vocab.encode(String(repeating: "x", count: 16_385)))
        XCTAssertThrowsError(try vocab.encode(String(repeating: "x ", count: 131_073)))
        XCTAssertThrowsError(try Qwen35Vocabulary(index: fixture(pre: "qwen2")))
        XCTAssertThrowsError(try Qwen35Vocabulary(index: fixture(merges: ["a b", "a b"])))
        XCTAssertThrowsError(try Qwen35Vocabulary(index: fixture(merges: ["x y"])))
    }

    func testTypedTextChatFramingAndTemplateBinding() throws {
        let prompt = try Qwen35NoThinkingChat(messages: [
            .init(role: .system, content: " \t\n"), .init(role: .user, content: " first "),
            .init(role: .assistant, content: " second\r\n"), .init(role: .user, content: "\u{a0}third\u{a0}"),
        ])
        XCTAssertEqual(prompt.renderedText,
            "<|im_start|>user\nfirst<|im_end|>\n<|im_start|>assistant\nsecond<|im_end|>\n"
            + "<|im_start|>user\n\u{a0}third\u{a0}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        XCTAssertThrowsError(try prompt.tokens(using: Qwen35Vocabulary(index: fixture()), maximumTokens: 1000),
                             "A missing or changed template must refuse production chat")
        XCTAssertThrowsError(try Qwen35NoThinkingChat(messages: []))
        XCTAssertThrowsError(try Qwen35NoThinkingChat(messages: [.init(role: .assistant, content: "x")]))
        XCTAssertThrowsError(try Qwen35NoThinkingChat(messages: [
            .init(role: .user, content: "x"), .init(role: .user, content: "y")]))
        XCTAssertThrowsError(try Qwen35NoThinkingChat(messages: [
            .init(role: .user, content: "x"), .init(role: .system, content: "y"), .init(role: .user, content: "z")]))
    }

    private func fixture(pre: String = "qwen35", merges: [String] = ["a b", "ab c"]) throws -> GGUFIndex {
        var body = Data(), fields: UInt64 = 0
        func text(_ key: String, _ value: String) {
            body.text(key); body.le(8, 4); body.text(value); fields += 1
        }
        func strings(_ key: String, _ values: [String]) {
            body.text(key); body.le(9, 4); body.le(8, 4); body.le(UInt64(values.count), 8)
            for value in values { body.text(value) }; fields += 1
        }
        text("tokenizer.ggml.model", "gpt2"); text("tokenizer.ggml.pre", pre)
        let direct = Set(Array(33...126) + Array(161...172) + Array(174...255))
        var extra = 256
        var alphabet: [String] = []
        for byte in 0...255 {
            if direct.contains(byte) { alphabet.append(String(Unicode.Scalar(byte)!)) }
            else { alphabet.append(String(Unicode.Scalar(extra)!)); extra += 1 }
        }
        strings("tokenizer.ggml.tokens", alphabet + ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "ab", "abc", "zz"])
        strings("tokenizer.ggml.merges", merges)
        body.text("tokenizer.ggml.token_type"); body.le(9, 4); body.le(4, 4); body.le(262, 8)
        for i in 0..<262 { body.le((256...258).contains(i) ? 3 : 1, 4) }; fields += 1
        body.text("tokenizer.ggml.eos_token_id"); body.le(4, 4); body.le(258, 4); fields += 1
        var header = Data("GGUF".utf8); header.le(3, 4); header.le(0, 8); header.le(fields, 8); header.append(body)
        while header.count % 32 != 0 { header.append(0) }
        return try GGUFIndex.parse(header: header, fileByteCount: UInt64(header.count))
    }
}

final class PublishedQwen35TokenizerParityTests: XCTestCase {
    private struct Corpus: Decodable {
        struct Chat: Decodable { let messages: [Qwen35ChatMessage] }
        let cases: [String]
        let chats: [Chat]
    }
    private struct Reference: Decodable {
        struct Entry: Decodable { let tokens: [Int]; let decodedHex: String }
        struct Chat: Decodable { let rendered: String; let tokens: [Int] }
        let vocabularySize: Int
        let cases: [Entry]
        let chats: [Chat]
    }
    func testEveryPublicCorpusTokenAndDecodedByteMatchesPinnedLlamaCPP() throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["GGUF_HEADER_FIXTURE"] ?? env["TEST_RUNNER_GGUF_HEADER_FIXTURE"] else {
            throw XCTSkip("Set GGUF_HEADER_FIXTURE and run the vocabulary-only reference")
        }
        let folder = URL(fileURLWithPath: path).deletingLastPathComponent()
        let referenceURL = folder.appendingPathComponent("qwen35-tokenizer-reference.json")
        guard FileManager.default.fileExists(atPath: referenceURL.path) else {
            throw XCTSkip("Run scripts/run_synthetic_reference.py for the tokenizer reference")
        }
        var limits = GGUFParseLimits(); limits.largeArrayKeys = Qwen35Vocabulary.metadataArrayKeys
        let index = try GGUFIndex.parse(header: Data(contentsOf: URL(fileURLWithPath: path)),
                                       fileByteCount: 56_115_609_536, limits: limits)
        let vocab = try Qwen35Vocabulary(index: index)
        let corpus = try JSONDecoder().decode(Corpus.self,
            from: Data(contentsOf: folder.appendingPathComponent("qwen35-tokenizer-corpus.json")))
        let reference = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: referenceURL))
        XCTAssertEqual(vocab.vocabularySize, 248320)
        XCTAssertEqual(vocab.vocabularySize, reference.vocabularySize)
        XCTAssertEqual(corpus.cases.count, reference.cases.count)
        XCTAssertGreaterThan(corpus.cases.count, 100)
        for (i, text) in corpus.cases.enumerated() {
            let expected = reference.cases[i]
            let actual = try vocab.encode(text)
            XCTAssertEqual(actual, expected.tokens, "Token IDs differ in corpus case \(i)")
            var bytes = Data()
            for token in actual { bytes.append(try vocab.bytes(for: token)) }
            XCTAssertEqual(bytes.map { String(format: "%02x", $0) }.joined(), expected.decodedHex,
                           "Decoded bytes differ in corpus case \(i)")
            XCTAssertEqual(bytes, Data(text.utf8), "Round trip changed raw UTF-8 in case \(i)")
        }
        XCTAssertEqual(vocab.eos, 248046)
        for marker in ["<|im_end|>", "<think>", "<tool_response>"] {
            XCTAssertThrowsError(try vocab.encode(marker), "Structural markup needs a dedicated refusal test")
        }
        XCTAssertEqual(corpus.chats.count, reference.chats.count)
        XCTAssertGreaterThan(corpus.chats.count, 20)
        for (i, chat) in corpus.chats.enumerated() {
            let prompt = try Qwen35NoThinkingChat(messages: chat.messages)
            XCTAssertEqual(prompt.renderedText, reference.chats[i].rendered, "Chat rendering differs in case \(i)")
            XCTAssertEqual(try prompt.tokens(using: vocab, maximumTokens: 65536), reference.chats[i].tokens,
                           "Chat token IDs differ in case \(i)")
        }
        print("QWEN35_TOKENIZER_PARITY cases=\(corpus.cases.count) exactTokenIDs=true exactBytes=true")
        print("QWEN35_CHAT_PARITY cases=\(corpus.chats.count) exactRender=true exactTokenIDs=true")
    }
}

private extension Data {
    mutating func le(_ value: UInt64, _ count: Int) {
        for i in 0..<count { append(UInt8(truncatingIfNeeded: value >> (8 * i))) }
    }
    mutating func text(_ value: String) {
        let bytes = Data(value.utf8); le(UInt64(bytes.count), 8); append(bytes)
    }
}
