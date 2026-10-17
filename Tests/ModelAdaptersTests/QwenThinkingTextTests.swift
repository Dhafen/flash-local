import XCTest
@testable import ModelAdapters

final class QwenThinkingTextTests: XCTestCase {
    func testEverySplitAcrossMarkersKeepsReasoningSeparateFromAnswer() {
        let text = "<think>Räkna 50 + 5. 🐟</think>\n\n55"
        for offset in 0...text.count {
            let boundary = text.index(text.startIndex, offsetBy: offset)
            var stream = QwenThinkingText()
            stream.append(String(text[..<boundary])); stream.append(String(text[boundary...])); stream.finish()
            XCTAssertEqual(stream.reasoning, "Räkna 50 + 5. 🐟")
            XCTAssertEqual(stream.answer, "\n\n55"); XCTAssertFalse(stream.isThinking)
        }
    }
    func testCanonicalPromptCanAlreadyContainOpeningThinkingTag() {
        var stream = QwenThinkingText(startsInThinking: true)
        stream.append("Kontrollera summan.")
        XCTAssertTrue(stream.isThinking); XCTAssertTrue(stream.answer.isEmpty)
        stream.append("</thi"); XCTAssertFalse(stream.reasoning.contains("</thi"))
        stream.append("nk>55")
        XCTAssertEqual(stream.reasoning, "Kontrollera summan.")
        XCTAssertEqual(stream.answer, "55"); XCTAssertFalse(stream.isThinking)
    }
    func testUnicodeScalarsAndRepeatedOpeningMarkerAreNotLost() {
        var stream = QwenThinkingText(startsInThinking: true)
        for scalar in "<think>Å cafe\u{0301} 👨‍👩‍👧‍👦</think>Klart!".unicodeScalars {
            stream.append(String(scalar))
        }
        stream.finish()
        XCTAssertEqual(stream.reasoning, "Å cafe\u{0301} 👨‍👩‍👧‍👦")
        XCTAssertEqual(stream.answer, "Klart!")
    }
    func testNoThinkingOutputAndIncompleteLiteralPrefixRemainUnchanged() {
        var stream = QwenThinkingText()
        stream.append("Hej! x <th"); stream.finish()
        XCTAssertEqual(stream.answer, "Hej! x <th"); XCTAssertTrue(stream.reasoning.isEmpty)
        XCTAssertFalse(stream.isThinking)
    }
    func testStoppingInsideReasoningNeverInventsAnAnswerOrClosingTag() {
        var stream = QwenThinkingText(startsInThinking: true)
        stream.append("Jag räknar</thi"); stream.finish()
        XCTAssertEqual(stream.reasoning, "Jag räknar</thi")
        XCTAssertTrue(stream.answer.isEmpty); XCTAssertTrue(stream.isThinking)
    }
    func testEmptyReasoningDoesNotLeakControlMarkup() {
        var stream = QwenThinkingText(startsInThinking: true)
        stream.append("</think>55"); stream.finish()
        XCTAssertTrue(stream.reasoning.isEmpty); XCTAssertEqual(stream.answer, "55")
    }
}
