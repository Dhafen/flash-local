import CryptoKit
import Foundation
import GGUFCore
import XCTest
@testable import ModelAdapters

final class PublishedOrnithLayoutTests: XCTestCase {
    func testPinnedOrnithHeaderValidatesTheCompleteTextTrunk() throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["GGUF_HEADER_FIXTURE"] ?? env["TEST_RUNNER_GGUF_HEADER_FIXTURE"] else {
            throw XCTSkip("Fetch the pinned headers; no full weights are required")
        }
        let header = try Data(contentsOf: URL(fileURLWithPath: path).deletingLastPathComponent().appendingPathComponent("ornith.header"))
        XCTAssertEqual(header.count, 10_990_848)
        XCTAssertEqual(SHA256.hash(data: header).map { String(format: "%02x", $0) }.joined(),
            "961df6ceaeedc2fbc0c84dfc16d8ba3ebf1d7e67506d10096a4d3a52d8a0836b")
        let index = try GGUFIndex.parse(header: header, fileByteCount: 21_713_463_040)
        let c = try Qwen35MoEConfig(index: index)
        XCTAssertEqual(index.tensors.count, 753)
        XCTAssertEqual(c.layers, 40); XCTAssertEqual(c.mtpLayers, 1)
        XCTAssertEqual(c.denseAttentionLayers, 10); XCTAssertEqual(c.recurrentLayers, 30)
        XCTAssertEqual(c.hiddenSize, 2048); XCTAssertEqual(c.vocabularySize, 248320)
        XCTAssertEqual(c.experts, 256); XCTAssertEqual(c.expertsPerToken, 8)
        XCTAssertEqual(c.expertWidth, 512); XCTAssertEqual(c.sharedWidth, 512)
        XCTAssertEqual(c.recurrentKeyHeads, 16); XCTAssertEqual(c.recurrentHeads, 32)
        XCTAssertEqual(c.keyLength, 256); XCTAssertEqual(c.rotaryDimensions, 64)
        XCTAssertEqual(Set(index.tensors.values.map { $0.quantization.rawValue }), Set([UInt32(0), 12, 14]))
        let plan = try c.memoryPlan(contextTokens: 65536, budgetBytes: 5_500_000_000)
        XCTAssertEqual(plan.kvBytes, 1_342_177_280)
        XCTAssertEqual(plan.recurrentMatrixBytes, 62_914_560)
        XCTAssertEqual(plan.cacheBytes, 3_094_908_160)
        XCTAssertThrowsError(try c.memoryPlan(contextTokens: c.contextLimit + 1, budgetBytes: 5_500_000_000))
        XCTAssertThrowsError(try Qwen4ExpDecoder.Layout(index: index), "A model switch must select its real graph")
        print("ORNITH_PUBLISHED_LAYOUT mainLayers=40 mtpLayers=1 executedMTP=false formats=3 tensorPayload=false")
    }
}
