import Foundation
import XCTest
@testable import OrnithPerformance

final class OrnithTokenizerTests: XCTestCase {
    struct Reference: Decodable {
        struct Case: Decodable {
            let messages: [OrnithChatMessage]
            let thinking: Bool
            let tokenIDs: [Int]
            let decoded: String
        }
        let revision: String
        let trainedInference: Bool
        let cases: [Case]
    }
    func testPinnedTokenizerAndLiteralTemplateMatchPythonInBothThinkingModes() async throws {
        guard let path = ProcessInfo.processInfo.environment["ORNITH_MLX_TOKENIZER_FIXTURE"] else {
            throw XCTSkip("Run scripts/prepare_ornith_mlx_tokenizer.py for the public tokenization oracle")
        }
        let dir = URL(fileURLWithPath: path)
        let manifest = try MLXSourceManifest.ornith()
        let reference = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: dir.appendingPathComponent("reference.json")))
        XCTAssertEqual(reference.revision, manifest.revision)
        XCTAssertFalse(reference.trainedInference)
        XCTAssertEqual(reference.cases.count, 12)
        let codec = try await OrnithChatCodec(localDirectory: dir, manifest: manifest)
        XCTAssertEqual(codec.endTokenIDs, [248046, 248044])
        for (index, item) in reference.cases.enumerated() {
            let ids = try codec.promptTokens(messages: item.messages, thinking: item.thinking)
            XCTAssertEqual(ids, item.tokenIDs, "case \(index)")
            XCTAssertEqual(codec.decode(ids), item.decoded, "case \(index)")
        }
        print("ORNITH_MLX_TOKENIZER_PARITY cases=12 exactTokenIDs=true exactDecodedText=true bothThinkingModes=true trainedInference=false")
    }

    func testInterruptedEmptyAndConsecutiveUserTurnsCannotReachTheModel() throws {
        XCTAssertThrowsError(try OrnithChatCodec.validate([.init(role: .user, content: "A"),
            .init(role: .assistant, content: ""), .init(role: .user, content: "B")]))
        XCTAssertThrowsError(try OrnithChatCodec.validate([.init(role: .user, content: "A"), .init(role: .user, content: "B")]))
        XCTAssertNoThrow(try OrnithChatCodec.validate([.init(role: .system, content: ""),
            .init(role: .user, content: "A"), .init(role: .assistant, content: "B"), .init(role: .user, content: "C")]))
    }
}
