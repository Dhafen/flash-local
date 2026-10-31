import XCTest
@testable import FlintEngine

final class FlintPolicyTests: XCTestCase {
    func testOnlyExactWholePrefixIsReused() {
        XCTAssertEqual(FlintPrefixPolicy.reusableCount(cached: [1, 2, 3], prompt: [1, 2, 3, 4]), 3)
        XCTAssertFalse(FlintPrefixPolicy.requiresReset(cached: [1, 2, 3], prompt: [1, 2, 3, 4]))
        XCTAssertTrue(FlintPrefixPolicy.requiresReset(cached: [1, 2, 3], prompt: [1, 2]))
        XCTAssertTrue(FlintPrefixPolicy.requiresReset(cached: [1, 2, 3], prompt: [1, 9, 3, 4]))
        XCTAssertEqual(FlintPrefixPolicy.reusableCount(cached: [], prompt: [1, 2]), 0)
        XCTAssertFalse(FlintPrefixPolicy.requiresReset(cached: [], prompt: [1, 2]))
    }
    func testExperimentalExpertCacheCapacityIsBounded() throws {
        var config = FlintConfiguration()
        XCTAssertEqual(config.expertCacheSlots, 0)
        try config.validate()
        config.expertCacheSlots = 128
        try config.validate()
        config.expertCacheSlots = 256
        try config.validate()
        config.expertCacheSlots = -1
        XCTAssertThrowsError(try config.validate())
        config.expertCacheSlots = 257
        XCTAssertThrowsError(try config.validate())
    }

    func testPerLayerDecodeCacheHasSafeBounds() throws {
        var config = FlintConfiguration()
        XCTAssertEqual(config.decodeCacheWindow, 0)
        config.decodeCacheWindow = 8
        try config.validate()
        config.decodeCacheWindow = 64
        try config.validate()
        config.decodeCacheWindow = -1
        XCTAssertThrowsError(try config.validate())
        config.decodeCacheWindow = 65
        XCTAssertThrowsError(try config.validate())
    }

    func testInvalidBatchAndContextAreRejected() throws {
        var config = FlintConfiguration()
        try config.validate()
        config.physicalBatch = 32
        XCTAssertThrowsError(try config.validate())
        config.physicalBatch = 4; config.evaluationBatch = 2
        XCTAssertThrowsError(try config.validate())
        config.evaluationBatch = 128; config.contextTokens = 262144
        XCTAssertThrowsError(try config.validate())
    }
}
