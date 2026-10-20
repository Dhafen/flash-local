import Foundation
import MinirunKit
import XCTest
@testable import ModelAdapters

final class CanonicalQwen35TokenizerParityTests: XCTestCase {
    func testDefaultSystemPromptPreservesEmptyHistoryAndExplicitClientInstruction() {
        let user = [Qwen35ChatMessage(role: .user, content: "Testa denna funktion.")]
        XCTAssertEqual(Qwen35ChatMessage.applyingSystemPrompt("", to: user), user)
        XCTAssertEqual(Qwen35ChatMessage.applyingSystemPrompt(" \n ", to: user), user)
        let withDefault = Qwen35ChatMessage.applyingSystemPrompt("Default instructions", to: user)
        XCTAssertEqual(withDefault.map(\.role), [.system, .user])
        XCTAssertEqual(withDefault.first?.content, "Default instructions")
        let explicit = [Qwen35ChatMessage(role: .system, content: "Client instructions")] + user
        XCTAssertEqual(Qwen35ChatMessage.applyingSystemPrompt("Default instructions", to: explicit), explicit)
        print("ORNITH_SYSTEM_DEFAULT canonicalRoles=true emptyUnchanged=true explicitClientWins=true modelLoaded=false")
    }
    private struct Corpus: Decodable {
        struct Chat: Decodable { let messages: [Qwen35ChatMessage] }
        struct ReasoningChat: Decodable {
            let messages: [Qwen35ChatMessage]
            let enable_thinking: Bool
        }
        let vocabularySHA256: String
        let vocabularyByteCount: UInt64
        let cases: [String]
        let chats: [Chat]
        let reasoningChats: [ReasoningChat]
    }
    private struct Reference: Decodable {
        struct Entry: Decodable { let tokens: [Int]; let decodedHex: String }
        struct Chat: Decodable { let rendered: String; let tokens: [Int] }
        let vocabularySize: Int
        let cases: [Entry]
        let chats: [Chat]
        let thinkingChats: [Chat]
        let reasoningChats: [Chat]
    }
    private func fixture() throws -> (URL, Corpus, Reference, GGUFModelManifest) {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["GGUF_HEADER_FIXTURE"] ?? env["TEST_RUNNER_GGUF_HEADER_FIXTURE"] else {
            throw XCTSkip("Set GGUF_HEADER_FIXTURE and run the canonical vocabulary reference")
        }
        let folder = URL(fileURLWithPath: path).deletingLastPathComponent()
        guard FileManager.default.fileExists(atPath: folder.appendingPathComponent("qwen35-tokenizer-reference.json").path) else {
            throw XCTSkip("Run scripts/run_synthetic_reference.py")
        }
        let corpus = try JSONDecoder().decode(Corpus.self,
            from: Data(contentsOf: folder.appendingPathComponent("qwen35-tokenizer-corpus.json")))
        let reference = try JSONDecoder().decode(Reference.self,
            from: Data(contentsOf: folder.appendingPathComponent("qwen35-tokenizer-reference.json")))
        let manifest = try GGUFModelManifest(fileName: "qwen35-vocabulary-only.gguf",
            fileByteCount: corpus.vocabularyByteCount, sha256: corpus.vocabularySHA256)
        return (folder, corpus, reference, manifest)
    }

    func testSwiftCallsCanonicalRuntimeForEveryTokenByteAndChat() throws {
        let (folder, corpus, reference, manifest) = try fixture()
        let artifact = try VerifiedGGUFArtifact.verify(insideRoot: folder, manifest: manifest)
        XCTAssertTrue(artifact.index.tensors.isEmpty, "This reference contains no trained tensors")
        let tokenizer = try CanonicalQwen35Tokenizer(artifact: artifact)
        XCTAssertEqual(tokenizer.vocabularySize, reference.vocabularySize)
        XCTAssertEqual(tokenizer.eos, 248046)
        XCTAssertEqual(tokenizer.endOfTurn, 248046)
        XCTAssertTrue(tokenizer.isEndOfGeneration(tokenizer.eos))
        XCTAssertFalse(tokenizer.isEndOfGeneration(-1))
        XCTAssertEqual(corpus.cases.count, reference.cases.count)
        XCTAssertGreaterThan(corpus.cases.count, 100)
        for (i, text) in corpus.cases.enumerated() {
            let ids = try tokenizer.tokenize(text)
            XCTAssertEqual(ids, reference.cases[i].tokens, "Canonical Swift token IDs at case \(i)")
            var bytes = Data()
            for id in ids { bytes.append(try tokenizer.bytes(for: id, includeControl: true)) }
            XCTAssertEqual(bytes.map { String(format: "%02x", $0) }.joined(), reference.cases[i].decodedHex)
            XCTAssertEqual(bytes, Data(text.utf8))
        }
        XCTAssertEqual(corpus.chats.count, reference.chats.count)
        XCTAssertEqual(corpus.chats.count, reference.thinkingChats.count)
        for (i, chat) in corpus.chats.enumerated() {
            let prompt = try tokenizer.chat(chat.messages, maximumTokens: 65536)
            XCTAssertEqual(prompt.renderedText, reference.chats[i].rendered)
            XCTAssertEqual(prompt.tokenIDs, reference.chats[i].tokens)
            let thinking = try tokenizer.chat(chat.messages, maximumTokens: 65536, enableThinking: true)
            XCTAssertEqual(thinking.renderedText, reference.thinkingChats[i].rendered)
            XCTAssertEqual(thinking.tokenIDs, reference.thinkingChats[i].tokens)
        }
        XCTAssertEqual(corpus.reasoningChats.count, 2)
        XCTAssertEqual(corpus.reasoningChats.count, reference.reasoningChats.count)
        for (i, chat) in corpus.reasoningChats.enumerated() {
            let prompt = try tokenizer.chat(chat.messages, maximumTokens: 65536,
                                            enableThinking: chat.enable_thinking)
            XCTAssertEqual(prompt.renderedText, reference.reasoningChats[i].rendered)
            XCTAssertEqual(prompt.tokenIDs, reference.reasoningChats[i].tokens)
            XCTAssertTrue(prompt.renderedText.contains("PUBLIC_REASONING_FIXTURE"))
        }
        for marker in ["<|im_end|>", "<think>", "<tool_response>"] {
            XCTAssertThrowsError(try tokenizer.chat([.init(role: .user, content: marker)], maximumTokens: 1024))
            XCTAssertThrowsError(try tokenizer.chat([
                .init(role: .user, content: "One"),
                .init(role: .assistant, content: "Two", reasoningContent: marker),
                .init(role: .user, content: "Three")], maximumTokens: 1024))
        }
        XCTAssertThrowsError(try tokenizer.chat([
            .init(role: .user, content: "One", reasoningContent: "Two")], maximumTokens: 1024))
        XCTAssertThrowsError(try tokenizer.chat([.init(role: .user, content: "Hello")], maximumTokens: 1))
        XCTAssertThrowsError(try tokenizer.tokenize("abc", maximumTokens: 0))
        XCTAssertThrowsError(try tokenizer.tokenize(String(repeating: "x", count: 262145)))
        XCTAssertThrowsError(try tokenizer.bytes(for: tokenizer.vocabularySize))
        print("CANONICAL_SWIFT_TOKENIZER_PARITY strings=\(corpus.cases.count) chats=\(corpus.chats.count) exact=true")
        print("CANONICAL_THINKING_PARITY cases=\(corpus.chats.count) reasoningHistory=2 exactRender=true exactTokenIDs=true bothModes=true")
    }

    func testBudgetClosingMarkerUsesCanonicalTokensAndPreservesAnswerSeparation() throws {
        let (folder, _, _, manifest) = try fixture()
        let artifact = try VerifiedGGUFArtifact.verify(insideRoot: folder, manifest: manifest)
        let tokenizer = try CanonicalQwen35Tokenizer(artifact: artifact)
        let closing = try tokenizer.tokenize("</think>\n\n", parseSpecial: true, maximumTokens: 8)
        XCTAssertFalse(closing.isEmpty); XCTAssertLessThanOrEqual(closing.count, 8)
        var bytes = Data()
        for id in closing { XCTAssertFalse(tokenizer.isEndOfGeneration(id)); bytes.append(try tokenizer.bytes(for: id, includeControl: true)) }
        XCTAssertEqual(bytes, Data("</think>\n\n".utf8))
        var text = QwenThinkingText(startsInThinking: true)
        text.append("Räknar"); text.append(String(decoding: bytes, as: UTF8.self)); text.append("55"); text.finish()
        XCTAssertFalse(text.isThinking); XCTAssertEqual(text.reasoning, "Räknar"); XCTAssertTrue(text.answer.hasSuffix("55"))
        print("ORNITH_THINKING_BUDGET canonicalClosingTokens=true reasoningSeparated=true modelWeightsLoaded=false phoneInference=false")
    }

    func testLoadedCanonicalVocabularyStillRefusesAChangedVerifiedFile() throws {
        let (folder, _, _, manifest) = try fixture()
        let scratch = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: scratch) }
        let file = scratch.appendingPathComponent(manifest.fileName)
        try FileManager.default.copyItem(at: folder.appendingPathComponent(manifest.fileName), to: file)
        let artifact = try VerifiedGGUFArtifact.verify(insideRoot: scratch, manifest: manifest)
        let tokenizer = try CanonicalQwen35Tokenizer(artifact: artifact)
        XCTAssertFalse(try tokenizer.tokenize("before mutation").isEmpty)
        let handle = try FileHandle(forWritingTo: file)
        try handle.seekToEnd(); try handle.write(contentsOf: Data([0])); try handle.close()
        XCTAssertThrowsError(try tokenizer.tokenize("after mutation"))
        XCTAssertThrowsError(try tokenizer.chat([.init(role: .user, content: "Hello")], maximumTokens: 1024))
        XCTAssertThrowsError(try tokenizer.bytes(for: 1))
    }
}
