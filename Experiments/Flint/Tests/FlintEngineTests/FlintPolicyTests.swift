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
