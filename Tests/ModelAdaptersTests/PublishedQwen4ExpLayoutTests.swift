import Foundation
import GGUFCore
import XCTest
@testable import ModelAdapters

final class PublishedQwen4ExpLayoutTests: XCTestCase {
    func testEveryRequiredDecoderTensorMatchesThePinnedPublishedHeader() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let path = environment["GGUF_HEADER_FIXTURE"] ?? environment["TEST_RUNNER_GGUF_HEADER_FIXTURE"] else {
            throw XCTSkip("Pinned metadata fixture is required; no full weights needed")
        }
        let index = try GGUFIndex.parse(header: Data(contentsOf: URL(fileURLWithPath: path)),
                                        fileByteCount: 56_115_609_536)
        let layout = try Qwen4ExpDecoder.Layout(index: index)
        XCTAssertEqual(layout.config.layers, 48)
        XCTAssertEqual(layout.vocabularySize, 248320)
        XCTAssertEqual(layout.attentionHeads, 24)
        XCTAssertEqual(layout.rotaryDimensions, 64)
        XCTAssertEqual(layout.recurrentKeyHeads, 16)
        XCTAssertEqual(layout.convolutionWidth, 4)
        XCTAssertEqual(layout.pleConvolutionWidth, 4)
        XCTAssertEqual(layout.ropeBase, 10_000_000)
        XCTAssertEqual(layout.epsilon, 1e-6, accuracy: 1e-10)
        XCTAssertFalse(layout.config.usesSparseAttention)
    }
}
