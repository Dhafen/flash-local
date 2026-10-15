import Foundation
import XCTest
@testable import MinirunRunners

final class Qwen4ExpGenerationPrimitivesTests: XCTestCase {
    func testGreedyPolicyUsesActualLargestLogitAndStableTies() throws {
        XCTAssertEqual(try Qwen4ExpGreedyToken.select([-10, 7, 6]), 1)
        XCTAssertEqual(try Qwen4ExpGreedyToken.select([4, 4, 3]), 0)
        XCTAssertEqual(try Qwen4ExpGreedyToken.select([-0.0, 0.0]), 0)
        XCTAssertEqual(try Qwen4ExpGreedyToken.select([-Float.greatestFiniteMagnitude]), 0)
        for invalid in [[Float](), [Float.nan], [0, Float.infinity], [-Float.infinity, 1]] {
            XCTAssertThrowsError(try Qwen4ExpGreedyToken.select(invalid))
        }
    }

    func testEveryUTF8ChunkSizePreservesTextAndAtMostThreePendingBytes() {
        for text in ["", "Hej världen!", "cafe\u{301} 中文", "👨‍👩‍👧‍👦 🇸🇪 🚀", "\0\u{feff}\n"] {
            check(Array(text.utf8))
        }
    }

    func testMalformedAndTruncatedUTF8MatchesCompleteDecoderReplacementPolicy() {
        let cases: [[UInt8]] = [[0xc3], [0xf0, 0x9f, 0x9a], [0xe0, 0x80], [0xed, 0xa0, 0x80],
                              [0xf4, 0x90, 0x80, 0x80], [0x80, 0x80, 0xff], [0xc0, 0xaf],
                              [0x41, 0xe2, 0x82, 0x42], [0xf0, 0x80, 0x80, 0x80],
                              [0xef, 0xbb, 0xbf, 0x61, 0xe2, 0x82]]
        for bytes in cases { check(bytes) }
    }

    private func check(_ bytes: [UInt8], file: StaticString = #filePath, line: UInt = #line) {
        for size in 1...7 {
            var stream = Qwen4ExpTokenText(), actual = "", offset = 0
            while offset < bytes.count {
                let end = min(offset + size, bytes.count)
                actual += stream.append(Data(bytes[offset..<end]))
                XCTAssertLessThanOrEqual(stream.pendingByteCount, 3, file: file, line: line)
                offset = end
            }
            actual += stream.finish()
            XCTAssertEqual(actual, String(decoding: bytes, as: UTF8.self), "Chunk size \(size)", file: file, line: line)
            XCTAssertEqual(stream.pendingByteCount, 0, file: file, line: line)
            XCTAssertEqual(stream.finish(), "", file: file, line: line)
        }
    }
}
