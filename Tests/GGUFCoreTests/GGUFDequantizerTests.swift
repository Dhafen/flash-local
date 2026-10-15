import Foundation
import XCTest
import GGUFCore

final class GGUFDequantizerTests: XCTestCase {
    private func decode(_ bytes: [UInt8], type: GGUFQuantization, count: Int) throws -> [Float] {
        let data = Data(bytes)
        var result = Array(repeating: Float.nan, count: count)
        try result.withUnsafeMutableBufferPointer { output in
            try data.withUnsafeBytes { input in
                try GGUFDequantizer.decode(input, type: type, elements: count, into: output)
            }
        }
        return result
    }

    func testQ4ZeroPointAndGGMLHalfBlockOrdering() throws {
        let values = try decode([0, 0x3c] + Array(repeating: 0xf0, count: 16), type: .q4_0, count: 32)
        XCTAssertEqual(values, Array(repeating: -8, count: 16) + Array(repeating: 7, count: 16))
    }

    func testQ5HighBitAndSignedZeroPoint() throws {
        let low = try decode([0, 0x3c] + Array(repeating: 0, count: 4)
            + Array(repeating: 0xf0, count: 16), type: .q5_0, count: 32)
        XCTAssertEqual(low, Array(repeating: -16, count: 16) + Array(repeating: -1, count: 16))
        let high = try decode([0, 0x3c] + Array(repeating: 255, count: 4)
            + Array(repeating: 0xf0, count: 16), type: .q5_0, count: 32)
        XCTAssertEqual(high, Array(repeating: 0, count: 16) + Array(repeating: 15, count: 16))
    }

    func testQ8UsesSignedInt8AndHalfScale() throws {
        let bytes = (-16..<16).map { UInt8(bitPattern: Int8($0)) }
        XCTAssertEqual(try decode([0, 0x38] + bytes, type: .q8_0, count: 32),
                       (-16..<16).map { Float($0) / 2 })
    }

    func testQ4KSubgroupScalesAndMinima() throws {
        // d=.5, dmin=.25; all eight scale/min pairs are one.
        let scales: [UInt8] = Array(repeating: 1, count: 8) + Array(repeating: 0x11, count: 4)
        let bytes: [UInt8] = [0, 0x38, 0, 0x34] + scales + Array(repeating: 0x32, count: 128)
        let values = try decode(bytes, type: .q4_K, count: 256)
        for i in 0..<256 { XCTAssertEqual(values[i], (i / 32) % 2 == 0 ? 0.75 : 1.25) }
    }

    func testQ6KHighBitsAndFourLanes() throws {
        let bytes: [UInt8] = Array(repeating: 0x21, count: 128)
            + Array(repeating: 0b11100100, count: 64)
            + Array(repeating: 1, count: 16) + [0, 0x38]
        let values = try decode(bytes, type: .q6_K, count: 256)
        let expected: [Float] = [-15.5, -7.5, 1, 9]
        for i in 0..<256 { XCTAssertEqual(values[i], expected[(i % 128) / 32]) }
    }

    func testIEEEHalfAndBF16AreDifferentEncodings() throws {
        XCTAssertEqual(try decode([0, 0x3c, 0, 0xc0, 1, 0], type: .f16, count: 3), [1, -2, 1 / 16_777_216])
        XCTAssertEqual(try decode([0x80, 0x3f, 0, 0xc0], type: .bf16, count: 2), [1, -2])
        XCTAssertEqual(try decode([0, 0, 0x80, 0x3f, 0, 0, 0, 0xc0], type: .f32, count: 2), [1, -2])
    }

    func testRejectsTruncationBlockRemainderAndOversizedTile() {
        XCTAssertThrowsError(try decode([0, 0], type: .q4_0, count: 32))
        XCTAssertThrowsError(try decode(Array(repeating: 0, count: 18), type: .q4_0, count: 31))
        var output = [Float](repeating: 0, count: 1)
        output.withUnsafeMutableBufferPointer { buffer in
            Data([0, 0, 0, 0]).withUnsafeBytes { bytes in
                XCTAssertThrowsError(try GGUFDequantizer.decode(bytes, type: .f32,
                    elements: GGUFDequantizer.maximumTileElements + 1, into: buffer))
            }
        }
    }
}
