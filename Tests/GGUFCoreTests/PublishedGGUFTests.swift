import Foundation
import XCTest
import GGUFCore

final class PublishedGGUFTests: XCTestCase {
    func testPinnedREAMHeaderHasExactMixedQuantizationAndExpertRanges() throws {
        let path = ProcessInfo.processInfo.environment["GGUF_HEADER_FIXTURE"]
            ?? ProcessInfo.processInfo.environment["TEST_RUNNER_GGUF_HEADER_FIXTURE"]
        guard let path else { throw XCTSkip("Run scripts/fetch_ream_metadata.py and set GGUF_HEADER_FIXTURE") }
        let header = try Data(contentsOf: URL(fileURLWithPath: path))
        let index = try GGUFIndex.parse(header: header, fileByteCount: 56_115_609_536)
        XCTAssertEqual(index.tensors.count, 1224)
        XCTAssertEqual(index.payloadOffset, 11_023_104)
        XCTAssertEqual(index.headerByteCount, 11_023_079)
        XCTAssertEqual(index.metadata["general.architecture"]?.text, "qwen4exp")
        var byType: [GGUFQuantization: UInt64] = [:]
        for tensor in index.tensors.values { byType[tensor.quantization, default: 0] += tensor.byteCount }
        XCTAssertEqual(byType, [.q8_0: 3_562_024_960, .f32: 161_943_040, .q4_0: 7_200_018_090,
            .q6_K: 1_238_630_400, .q5_0: 16_652_697_600, .q4_K: 27_249_868_800,
            .f16: 81_920, .bf16: 39_321_600])
        let down = try XCTUnwrap(index.tensors["blk.47.ffn_down_exps.weight"])
        XCTAssertEqual(down.dimensions, [640, 2560, 308])
        let region = try down.region(sourceID: 0, regionID: 0, payloadOffset: index.payloadOffset,
                                     rows: 0..<2560, expert: 307)
        XCTAssertEqual(region.length, 1_126_400)
        XCTAssertLessThanOrEqual(region.offset + UInt64(region.length), index.fileByteCount)
        let ple = try XCTUnwrap(index.tensors["per_layer_token_embd.weight"])
        XCTAssertEqual(ple.dimensions, [160, 80_000_201])
        XCTAssertEqual(ple.rowBytes, 90)
        XCTAssertEqual(index.metadata["qwen4exp.ple.layer_multipliers"]?.elements?.first?.uint,
                       23_703_573_157_769)
    }
}
