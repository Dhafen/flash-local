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

    func testMatchedQ4DiagnosticNeverGrantsServingAndOlderReportsStillDecode() throws {
        let baseline = FlintNumericalCheck(positions: [], routedTokenLayers: 0,
            exactExpertSets: false, routeDiagnostics: nil)
        let oldJSON = try JSONEncoder().encode(baseline)
        let restoredOld = try JSONDecoder().decode(FlintNumericalCheck.self, from: oldJSON)
        XCTAssertNil(restoredOld.matchedMetalQ4F32Control)
        XCTAssertFalse(restoredOld.passed)

        let row = try FlintNumericalCheck.compare(position: 6, reference: [0, 3, 1], candidate: [0, 3, 1])
        let cross = FlintNumericalCheck.MatchedMetalQ4F32Control(
            positions: [row], exactExpertSets: true,
            routeDiagnostics: .init(referenceCount: 1, candidateCount: 1, mismatchCount: 0, firstMismatch: nil))
        let report = FlintNumericalCheck(positions: [], routedTokenLayers: 0,
            exactExpertSets: false, routeDiagnostics: nil, matchedMetalQ4F32Control: cross)
        let restored = try JSONDecoder().decode(FlintNumericalCheck.self, from: JSONEncoder().encode(report))
        XCTAssertEqual(restored.matchedMetalQ4F32Control?.positions.count, 1)
        XCTAssertEqual(restored.matchedMetalQ4F32Control?.routeDiagnostics?.mismatchCount, 0)
        XCTAssertFalse(restored.passed, "A diagnostic result must never bypass serving admission")
    }
}
