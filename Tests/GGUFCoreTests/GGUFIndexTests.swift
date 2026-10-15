import Foundation
import XCTest
@testable import GGUFCore

final class GGUFIndexTests: XCTestCase {
    private func fixture(
        type: UInt32 = 12, dimensions: [UInt64] = [256, 4, 3], offset: UInt64 = 0,
        extras: [(String, UInt32, Data)] = []
    ) -> (Data, UInt64) {
        var data = Data("GGUF".utf8)
        data.le(3, 4); data.le(1, 8); data.le(UInt64(extras.count + 1), 8)
        data.string("general.architecture"); data.le(8, 4); data.string("qwen4exp")
        for (key, type, value) in extras { data.string(key); data.le(UInt64(type), 4); data.append(value) }
        data.string("blk.0.ffn_gate_exps.weight"); data.le(UInt64(dimensions.count), 4)
        for dimension in dimensions { data.le(dimension, 8) }
        data.le(UInt64(type), 4); data.le(offset, 8)
        let payload = UInt64((data.count + 31) / 32 * 32)
        // A logical model can exceed memory without allocating its payload.
        return (data, payload + offset + 8_000_000_000)
    }

    func testInspectsExpertStackWithoutAllocatingPayload() throws {
        let (header, size) = fixture()
        let index = try GGUFIndex.parse(header: header, fileByteCount: size)
        XCTAssertLessThan(header.count, 256)
        XCTAssertGreaterThan(index.fileByteCount, 8_000_000_000)
        XCTAssertEqual(index.metadata["general.architecture"], .string("qwen4exp"))
        let tensor = try XCTUnwrap(index.tensors["blk.0.ffn_gate_exps.weight"])
        XCTAssertEqual(tensor.rowBytes, 144)
        XCTAssertEqual(tensor.byteCount, 144 * 4 * 3)
        let region = try tensor.region(sourceID: 7, regionID: 20,
                                      payloadOffset: index.payloadOffset, rows: 1..<3, expert: 2)
        XCTAssertEqual(region.offset, index.payloadOffset + (2 * 4 + 1) * 144)
        XCTAssertEqual(region.length, 288)
        XCTAssertEqual(region.identity.sourceID, 7)
    }

    func testExactGGMLRowSizesForAllPublishedEncodings() throws {
        let cases: [(UInt32, UInt64, UInt64)] = [
            (0, 2560, 10240), (1, 2560, 5120), (30, 2560, 5120),
            (2, 160, 90), (6, 640, 440), (8, 2560, 2720),
            (12, 2560, 1440), (14, 2560, 2100),
        ]
        for (kind, columns, expected) in cases {
            let (header, size) = fixture(type: kind, dimensions: [columns, 1])
            let index = try GGUFIndex.parse(header: header, fileByteCount: size)
            XCTAssertEqual(index.tensors.values.first?.rowBytes, expected)
        }
    }

    func testPublishedExpertStrides() throws {
        let (header, size) = fixture(dimensions: [2560, 640, 308])
        let index = try GGUFIndex.parse(header: header, fileByteCount: size)
        let tensor = try XCTUnwrap(index.tensors.values.first)
        XCTAssertEqual(tensor.byteCount, 283_852_800)
        let expert = try tensor.region(sourceID: 0, regionID: 1,
                                      payloadOffset: index.payloadOffset, rows: 0..<640, expert: 307)
        XCTAssertEqual(expert.length, 921_600)
        XCTAssertEqual(expert.offset, index.payloadOffset + 307 * 921_600)
        XCTAssertThrowsError(try tensor.region(sourceID: 0, regionID: 1,
            payloadOffset: index.payloadOffset, rows: 0..<640, expert: 308))
        XCTAssertThrowsError(try tensor.region(sourceID: 0, regionID: 1,
            payloadOffset: index.payloadOffset, rows: 0..<641, expert: 0))
        XCTAssertThrowsError(try tensor.region(sourceID: 0, regionID: 1,
            payloadOffset: index.payloadOffset, rows: 0..<0, expert: 0))
    }

    func testEveryTruncatedPrefixFails() {
        let (header, size) = fixture()
        for count in 0..<header.count {
            XCTAssertThrowsError(try GGUFIndex.parse(header: Data(header.prefix(count)), fileByteCount: size))
        }
    }

    func testDoesNotAssumeDataStartsAtZeroIndex() throws {
        let (header, size) = fixture()
        var prefixed = Data([0, 0, 0]); prefixed.append(header)
        let slice = prefixed.dropFirst(3)
        let index = try GGUFIndex.parse(header: slice, fileByteCount: size)
        XCTAssertEqual(index.tensors.count, 1)
    }

    func testUnsupportedTypesAndInvalidBlocksFailClosed() {
        for (type, dimensions) in [(UInt32(999), [UInt64(256), 2]), (12, [255, 2]), (0, [0, 2])] {
            let (header, size) = fixture(type: type, dimensions: dimensions)
            XCTAssertThrowsError(try GGUFIndex.parse(header: header, fileByteCount: size))
        }
    }

    func testOffsetsOverflowAndTruncatedPayloadAreRejected() {
        let (header, _) = fixture()
        XCTAssertThrowsError(try GGUFIndex.parse(header: header, fileByteCount: UInt64(header.count)))
        let (overflow, _) = fixture(dimensions: [256, UInt64.max, 308])
        XCTAssertThrowsError(try GGUFIndex.parse(header: overflow, fileByteCount: UInt64.max))
        let (unaligned, size) = fixture(offset: 1)
        XCTAssertThrowsError(try GGUFIndex.parse(header: unaligned, fileByteCount: size))
    }

    func testDuplicateMetadataAndWrongAlignmentTypeAreRejected() {
        var text = Data(); text.string("another")
        let (duplicate, size) = fixture(extras: [("general.architecture", 8, text)])
        XCTAssertThrowsError(try GGUFIndex.parse(header: duplicate, fileByteCount: size))
        let (badAlignment, n) = fixture(extras: [("general.alignment", 8, text)])
        XCTAssertThrowsError(try GGUFIndex.parse(header: badAlignment, fileByteCount: n))
    }

    func testRetainsHashConstantsAsIntegersWithoutFloatRounding() throws {
        var array = Data(); array.le(10, 4); array.le(3, 8)
        for value: UInt64 in [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071] { array.le(value, 8) }
        let (header, size) = fixture(extras: [("qwen4exp.ple.layer_multipliers", 9, array)])
        let index = try GGUFIndex.parse(header: header, fileByteCount: size)
        XCTAssertEqual(index.metadata["qwen4exp.ple.layer_multipliers"]?.elements?.first?.uint, 23_703_573_157_769)
    }

    func testSkipsLargeVocabularyArraysAndStillParsesFollowingTensors() throws {
        var array = Data(); array.le(8, 4); array.le(10, 8)
        for _ in 0..<10 { array.string("token") }
        var limits = GGUFParseLimits(); limits.retainedArrayCount = 4
        let (header, size) = fixture(extras: [("tokenizer.ggml.tokens", 9, array)])
        let index = try GGUFIndex.parse(header: header, fileByteCount: size, limits: limits)
        XCTAssertEqual(index.metadata["tokenizer.ggml.tokens"], .omittedArray(type: 8, count: 10))
        XCTAssertEqual(index.tensors.count, 1)
    }

    func testMetadataLimitsRejectBeforeAllocatingFromUntrustedCounts() {
        let (header, size) = fixture()
        var limits = GGUFParseLimits(); limits.headerBytes = 10
        XCTAssertThrowsError(try GGUFIndex.parse(header: header, fileByteCount: size, limits: limits))
        limits = .init(); limits.tensorCount = 0
        XCTAssertThrowsError(try GGUFIndex.parse(header: header, fileByteCount: size, limits: limits))
        var array = Data(); array.le(8, 4); array.le(UInt64.max, 8)
        let (large, n) = fixture(extras: [("tokens", 9, array)])
        XCTAssertThrowsError(try GGUFIndex.parse(header: large, fileByteCount: n))
    }

    func testMagicVersionBooleanAndNestedArraysAreRejected() {
        let (header, size) = fixture()
        var badMagic = header; badMagic[0] = 0
        XCTAssertThrowsError(try GGUFIndex.parse(header: badMagic, fileByteCount: size))
        var badVersion = header; badVersion[4] = 2
        XCTAssertThrowsError(try GGUFIndex.parse(header: badVersion, fileByteCount: size))
        let (badBool, n) = fixture(extras: [("bool", 7, Data([2]))])
        XCTAssertThrowsError(try GGUFIndex.parse(header: badBool, fileByteCount: n))
        var array = Data(); array.le(9, 4); array.le(0, 8)
        let (nested, m) = fixture(extras: [("nested", 9, array)])
        XCTAssertThrowsError(try GGUFIndex.parse(header: nested, fileByteCount: m))
    }
}

private extension Data {
    mutating func le(_ value: UInt64, _ count: Int) {
        for i in 0..<count { append(UInt8(truncatingIfNeeded: value >> (i * 8))) }
    }
    mutating func string(_ value: String) {
        let bytes = Data(value.utf8); le(UInt64(bytes.count), 8); append(bytes)
    }
}
