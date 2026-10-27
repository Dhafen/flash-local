import Foundation
import XCTest
@testable import FlintEngine
import MinirunKit

final class FlintStreamTests: XCTestCase {
    func testNumericalAdmissionRejectsWrongTop1NonFiniteOrDistantLogits() throws {
        let exact = try FlintNumericalCheck.compare(position: 3, reference: [1, 3, -2], candidate: [1, 3, -2])
        XCTAssertTrue(exact.passed)
        XCTAssertFalse(try FlintNumericalCheck.compare(position: 3, reference: [1, 3, -2], candidate: [4, 3, -2]).passed)
        XCTAssertFalse(try FlintNumericalCheck.compare(position: 3, reference: [1, 3, -2], candidate: [1, 3.2, -2]).passed)
        XCTAssertThrowsError(try FlintNumericalCheck.compare(position: 3, reference: [1, .nan], candidate: [1, 2]))
        XCTAssertThrowsError(try FlintNumericalCheck.compare(position: 3, reference: [1], candidate: [1, 2]))
    }
    func testNumericalAdmissionRequiresActualCompleteExpertSets() {
        let a = FlintRouteDecision(position: 3, layer: 48, experts: [1, 25, 127, 200, 250, 255])
        let reordered = FlintRouteDecision(position: 3, layer: 48, experts: a.experts.reversed())
        XCTAssertTrue(FlintNumericalCheck.sameRoutes([a], [reordered]))
        XCTAssertFalse(FlintNumericalCheck.sameRoutes([], []))
        XCTAssertFalse(FlintNumericalCheck.sameRoutes([a], [a, a]))
        XCTAssertFalse(FlintNumericalCheck.sameRoutes([a], [.init(position: 3, layer: 48, experts: [1, 25, 127, 200, 250, 254])]))
        XCTAssertFalse(FlintNumericalCheck.sameRoutes([a], [.init(position: 3, layer: 48, experts: [1, 25, 127, 200, 250, 256])]))
    }
    func testReasoningAndUTF8AreIdenticalForEveryBytePartition() throws {
        let text = "<think>Räkna 👋: 50+5.</think>Svaret är 55.\n"
        let bytes = Array(text.utf8)
        for size in 1...bytes.count {
            var stream = FlintTextStream(), answer = "", reasoning = ""
            for start in stride(from: 0, to: bytes.count, by: size) {
                let delta = try stream.append(Array(bytes[start..<min(start + size, bytes.count)]))
                answer += delta.content; reasoning += delta.reasoning
            }
            let end = try stream.finish(); answer += end.content; reasoning += end.reasoning
            XCTAssertEqual(answer, "Svaret är 55.\n"); XCTAssertEqual(reasoning, "Räkna 👋: 50+5.")
        }
    }
    func testIncompleteAndInvalidUTF8NeverCorruptToolArguments() throws {
        for bytes: [UInt8] in [[0x80], [0xc0, 0x80], [0xed, 0xa0, 0x80], [0xf4, 0x90, 0x80, 0x80], [0xe2, 0x28]] {
            var stream = FlintTextStream()
            XCTAssertThrowsError(try stream.append(bytes))
        }
        var stream = FlintTextStream()
        XCTAssertEqual(try stream.append([0xf0, 0x9f]).content, "")
        XCTAssertThrowsError(try stream.finish())
    }
    func testTruncatedDelimiterAndUnfinishedReasoningAreNotFabricated() throws {
        var text = FlintTextStream()
        let delta = try text.append(Array("code: <thi".utf8))
        XCTAssertEqual(delta.content, "code: ")
        XCTAssertEqual(try text.finish().content, "<thi")
        var thinking = FlintTextStream()
        XCTAssertEqual(try thinking.append(Array("<think>still working".utf8)).reasoning, "still working")
        XCTAssertTrue(thinking.isThinking); XCTAssertEqual(try thinking.finish().content, "")
    }
    func testWirePromptReplaysReasoningAndToolHistoryWithoutQwenTags() throws {
        let caps = LocalModelCapabilities(family: .flint)
        let request = try LocalChatRequest(data: JSONSerialization.data(withJSONObject: ["model": caps.modelID,
            "messages": [["role": "user", "content": "50+5"]]]), capabilities: caps)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: FlintWirePrompt.encode(request)) as? [String: Any])
        XCTAssertEqual((json["messages"] as? [[String: String]])?.first?["content"], "50+5")
        XCTAssertEqual(json["add_generation_prompt"] as? Bool, true)
    }
    func testSlidingWindowPrefixReuseRejectsChangedOrShortenedHistory() {
        XCTAssertEqual(FlintPrefixPolicy.reusableCount(cached: [1, 2, 3], prompt: [1, 2, 3, 4]), 3)
        XCTAssertFalse(FlintPrefixPolicy.requiresReset(cached: [1, 2, 3], prompt: [1, 2, 3, 4]))
        XCTAssertTrue(FlintPrefixPolicy.requiresReset(cached: [1, 2, 3], prompt: [1, 2]))
        XCTAssertTrue(FlintPrefixPolicy.requiresReset(cached: [1, 2, 3], prompt: [1, 9, 3]))
    }
}
