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
        XCTAssertThrowsError(try config.validate())
        config.decodeCacheWindow = 64
        XCTAssertThrowsError(try config.validate())
        config.decodeCacheWindow = -1
        XCTAssertThrowsError(try config.validate())
        config.decodeCacheWindow = 65
        XCTAssertThrowsError(try config.validate())
    }

    func testServingModeDoesNotSilentlyEnableUnverifiedBatchOrFlashAttention() throws {
        let requested = FlintConfiguration()
        let standard = try FlintConfiguration.forMode(.standard, requested: requested, tracing: false)
        XCTAssertEqual(standard.physicalBatch, 4)
        XCTAssertFalse(standard.conservativeMetal)
        let conservative = try FlintConfiguration.forMode(.singleTokenF32KV, requested: requested, tracing: false)
        XCTAssertEqual(conservative.physicalBatch, 1)
        XCTAssertTrue(conservative.conservativeMetal)
        let diagnostic = try FlintConfiguration.forMode(.singleTokenF32KV, requested: requested, tracing: true)
        XCTAssertEqual(diagnostic.physicalBatch, 1)
        XCTAssertTrue(diagnostic.diagnosticDisableFlash)
    }

    func testReportTracksEffectiveExpertCacheConfiguration() throws {
        let slots = 128
        let report = FlintGenerationReport(modelSHA256: "test", nativeRevision: "test", contextTokens: 8192,
            evaluationBatch: 128, physicalBatch: 1, cpuThreads: 4, servingMode: .singleTokenF32KV,
            f32KV: true, flashAttentionActive: false, processMemoryBudgetBytes: 5_500_000_000,
            platform: "iOSDevice", operatingSystem: "test", sourceRevision: "test", appBuild: "test",
            timestamp: Date(), promptTokenIDs: [1], generatedTokenIDs: [2], reusedPromptTokens: 0,
            prefillSeconds: 1, decodeForwardSeconds: 1, elapsedSeconds: 2, timeToFirstTokenSeconds: 1,
            maximumObservedFootprintBytes: 1, explicitExpertBytesRead: 2, maximumThermalState: 0, finishReason: "length")
        var recorded = report
        recorded.expertCacheSlots = slots
        recorded.expertReadAheadRequested = slots > 0
        let decoded = try JSONDecoder().decode(FlintGenerationReport.self, from: JSONEncoder().encode(recorded))
        XCTAssertEqual(decoded.expertCacheSlots, slots)
        XCTAssertEqual(decoded.expertReadAheadRequested, true)
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
