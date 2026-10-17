import Foundation
import XCTest
import GGUFCore
import StorageCore
import ModelAdapters
import MinirunKit
@testable import OrnithLES

final class LESPolicyTests: XCTestCase {
    private func index() throws -> GGUFIndex {
        let root = try XCTUnwrap(ProcessInfo.processInfo.environment["ORNITH_LES_FIXTURE_ROOT"])
        let data = try Data(contentsOf: URL(fileURLWithPath: root).appendingPathComponent("synthetic-ornith.gguf"))
        return try GGUFIndex.parse(header: data, fileByteCount: UInt64(data.count))
    }
    private var footprint: ProcessFootprint { .init(footprintBytes: 100_000_000, peakFootprintBytes: 100_000_000, availableBytes: nil) }
    func testAdmissionIncludesDensePrefillKVSSMAndBaseline() throws {
        let plan = try LESMemoryPlan(index: index(), contextTokens: 32, footprint: footprint)
        XCTAssertEqual(plan.admittedProcessBytes, plan.baselineFootprintBytes + plan.denseBytes + plan.kvBytes +
            plan.ssmBytes + plan.runtimeReserveBytes + plan.expertAllocationBytes)
        XCTAssertLessThanOrEqual(plan.admittedProcessBytes, 5_500_000_000)
        XCTAssertGreaterThan(plan.denseBytes, 0); XCTAssertGreaterThan(plan.expertAllocationBytes, 0)
    }
    func testLookaheadComparisonKeepsTotalWindowMemoryConstant() throws {
        let index = try index()
        let off = try LESMemoryPlan(index: index, contextTokens: 32, footprint: footprint, lookaheadDepth: 0)
        let on = try LESMemoryPlan(index: index, contextTokens: 32, footprint: footprint, lookaheadDepth: 4)
        XCTAssertEqual(off.expertAllocationBytes, on.expertAllocationBytes)
        XCTAssertEqual(on.lookaheadDepth, 3)
        XCTAssertEqual(off.cacheSlotsPerLayer, on.cacheSlotsPerLayer + on.lookaheadDepth)
    }
    func testPageCacheVariantRetainsBoundedStagingWithoutDecodeCache() throws {
        let index = try index()
        let own = try LESMemoryPlan(index: index, contextTokens: 32, footprint: footprint)
        let os = try LESMemoryPlan(index: index, contextTokens: 32, footprint: footprint, pageCacheOnly: true)
        XCTAssertEqual(os.cacheSlotsPerLayer, 0); XCTAssertEqual(os.lookaheadDepth, 0)
        XCTAssertGreaterThan(os.expertAllocationBytes, 0); XCTAssertLessThan(os.expertAllocationBytes, own.expertAllocationBytes)
    }
    func testCurrentIOSHeadroomCanRejectOtherwiseValidPlan() throws {
        let sample = ProcessFootprint(footprintBytes: 100_000_000, peakFootprintBytes: 100_000_000, availableBytes: 64 << 20)
        XCTAssertThrowsError(try LESMemoryPlan(index: index(), contextTokens: 32, footprint: sample))
    }
    func testPublishedHeaderAdaptsPredictionBeforeChangingTheRealExpertAllowance() throws {
        let root = try XCTUnwrap(ProcessInfo.processInfo.environment["ORNITH_LES_FIXTURE_ROOT"])
        let header = try Data(contentsOf: URL(fileURLWithPath: root).appendingPathComponent("ornith.header"))
        XCTAssertEqual(header.count, 10_990_848)
        let index = try GGUFIndex.parse(header: header, fileByteCount: 21_713_463_040)
        let config = try Qwen35MoEConfig(index: index)
        XCTAssertEqual(config.experts, 256); XCTAssertEqual(config.expertsPerToken, 8)
        let baseline: UInt64 = 350_000_000
        func plan(_ ceiling: UInt64, depth: Int = 4, context: Int = 128) throws -> LESMemoryPlan {
            try LESMemoryPlan(index: index, contextTokens: context,
                footprint: .init(footprintBytes: baseline, peakFootprintBytes: baseline,
                    availableBytes: ceiling - baseline), lookaheadDepth: depth)
        }
        let full = try plan(5_500_000_000, context: 4096)
        XCTAssertEqual(full.lookaheadDepth, 4); XCTAssertEqual(full.requestedLookaheadDepth, 4)
        XCTAssertEqual(full.expertBytesPerSlot, 78_225_408)
        XCTAssertEqual(full.prefillAllocationBytes, 84_148_224)
        XCTAssertLessThanOrEqual(full.admittedProcessBytes, full.processCeilingBytes)
        let narrow = try plan(4_300_000_000), off = try plan(4_300_000_000, depth: 0)
        XCTAssertEqual(narrow.cacheSlotsPerLayer, 8); XCTAssertEqual(narrow.lookaheadDepth, 3)
        XCTAssertEqual(narrow.requestedLookaheadDepth, 4)
        XCTAssertEqual(narrow.expertAllocationBytes, off.expertAllocationBytes)
        let demandOnly = try plan(4_100_000_000)
        XCTAssertEqual(demandOnly.cacheSlotsPerLayer, 8); XCTAssertEqual(demandOnly.lookaheadDepth, 0)
        XCTAssertLessThanOrEqual(demandOnly.admittedProcessBytes, demandOnly.processCeilingBytes)
        XCTAssertThrowsError(try plan(4_000_000_000)) { error in
            let text = error.localizedDescription
            XCTAssertTrue(text.contains("all 8 selected experts"))
            XCTAssertTrue(text.contains("baseline=350000000")); XCTAssertTrue(text.contains("window=7"))
        }
        let sample = LESMemoryAdmission(phase: "synthetic.admission", beforeReclaim: footprint,
            afterReclaim: footprint, plan: narrow, failure: nil)
        let encoded = String(decoding: try JSONEncoder().encode(sample), as: UTF8.self)
        for key in ["promptTokenIDs", "generatedTokenIDs", "logits", "messages", "path"] {
            XCTAssertFalse(encoded.contains("\"" + key + "\""))
        }
        print("ORNITH_REAL_HEADER_MEMORY selectedExperts=8 requestedPrediction=4 adaptivePrediction=3 demandOnlyPrediction=0 insufficientDemandRejected=true metadataOnly=true")
    }
    func testNoExpertCapacityMayBeBorrowedFromOtherReservations() throws {
        let sample = ProcessFootprint(footprintBytes: 5_000_000_000, peakFootprintBytes: 5_000_000_000, availableBytes: nil)
        XCTAssertThrowsError(try LESMemoryPlan(index: index(), contextTokens: 32, footprint: sample))
    }
    func testInvalidContextAndLookaheadAreRejected() throws {
        for context in [0, 31, 65537] {
            XCTAssertThrowsError(try LESMemoryPlan(index: index(), contextTokens: context, footprint: footprint))
        }
        XCTAssertThrowsError(try LESMemoryPlan(index: index(), contextTokens: 32, footprint: footprint, lookaheadDepth: 17))
        XCTAssertEqual(try LESMemoryPlan(index: index(), contextTokens: 32, footprint: footprint,
            lookaheadDepth: 16).lookaheadDepth, 3)
    }
    func testNumericalGateRequiresAllLogitsAndGreedyAgreement() {
        XCTAssertTrue(LESLogitComparison(positions: 4, maximumAbsoluteError: 0.001, relativeL2Error: 0.0001, sameGreedy: true).passed)
        for value in [
            LESLogitComparison(positions: 3, maximumAbsoluteError: 0, relativeL2Error: 0, sameGreedy: true),
            LESLogitComparison(positions: 4, maximumAbsoluteError: 0.02, relativeL2Error: 0, sameGreedy: true),
            LESLogitComparison(positions: 4, maximumAbsoluteError: 0, relativeL2Error: 0.002, sameGreedy: true),
            LESLogitComparison(positions: 4, maximumAbsoluteError: -.infinity, relativeL2Error: 0, sameGreedy: true),
            LESLogitComparison(positions: 4, maximumAbsoluteError: 0, relativeL2Error: -0.001, sameGreedy: true),
            LESLogitComparison(positions: 4, maximumAbsoluteError: .nan, relativeL2Error: 0, sameGreedy: true),
            LESLogitComparison(positions: 4, maximumAbsoluteError: 0, relativeL2Error: 0, sameGreedy: false)] {
            XCTAssertFalse(value.passed)
        }
    }
    func testAggregateNumericalExportDoesNotContainTokenOrLogitPayloads() throws {
        let result = LESLogitComparison(positions: 4, maximumAbsoluteError: 0, relativeL2Error: 0, sameGreedy: true,
            routingDigest: 19, routingDecisions: 48)
        let check = LESNumericalCheck(modelSHA256: String(repeating: "a", count: 64), referenceRevision: "reference",
            nativeRevision: "candidate", noLookahead: result, lookahead: result, batchedPrefill: result)
        let encoded = String(decoding: try JSONEncoder().encode(check), as: UTF8.self)
        for key in ["promptTokenIDs", "generatedTokenIDs", "logits", "chat", "messages"] { XCTAssertFalse(encoded.contains("\"" + key + "\"")) }
        XCTAssertTrue(check.passed); XCTAssertFalse(check.fullModelNumericsVerified)
        var different = result; different.routingDigest = 20
        XCTAssertFalse(LESNumericalCheck(modelSHA256: "same", referenceRevision: "reference", nativeRevision: "candidate",
            noLookahead: result, lookahead: different, batchedPrefill: result).passed)
        var untraced = result; untraced.routingDecisions = 0
        XCTAssertFalse(LESNumericalCheck(modelSHA256: "same", referenceRevision: "reference", nativeRevision: "candidate",
            noLookahead: untraced, lookahead: untraced, batchedPrefill: result).passed)
    }
    func testOldCPUReferenceAndIsolatedLESCoexistAndPreserveSource() throws {
        let fixtures = try XCTUnwrap(ProcessInfo.processInfo.environment["ORNITH_LES_FIXTURE_ROOT"])
        let source = URL(fileURLWithPath: fixtures).appendingPathComponent("synthetic-ornith.gguf")
        let data = try Data(contentsOf: source)
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("ornith-les-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        try data.write(to: root.appendingPathComponent("synthetic.gguf"))
        let manifest = try GGUFModelManifest(fileName: "synthetic.gguf", fileByteCount: UInt64(data.count),
            sha256: FileDigestComputer.sha256(of: data))
        let artifact = try VerifiedGGUFArtifact.verify(insideRoot: root, manifest: manifest)
        let expected: [[Float]] = try autoreleasepool {
            let reference = try Qwen4ExpCPUReference(artifact: artifact, contextTokens: 32, budgetBytes: 5_500_000_000)
            return try [1, 2].map { try reference.logits(for: $0) }
        }
        let control = LESControl()
        var admissions: [LESMemoryAdmission] = []
        let prepared = try LESPreparation.prepare(artifact: artifact, privateRoot: root.appendingPathComponent("prepared"),
            control: control, onAdmission: { admissions.append($0) })
        XCTAssertEqual(admissions.count, 1)
        let admission = try XCTUnwrap(admissions.first)
        XCTAssertEqual(admission.phase, "preparation"); XCTAssertNil(admission.failure)
        XCTAssertNotNil(admission.beforeReclaim); XCTAssertNotNil(admission.afterReclaim)
        XCTAssertEqual(admission.plan?.contextTokens, 32)
        print("ORNITH_PREPARATION_ADMISSION phaseCaptured=true beforeAndAfterCaptured=true syntheticOnly=true")
        let plan = try LESMemoryPlan(index: artifact.index, contextTokens: 32, footprint: XCTUnwrap(ProcessFootprint.current()))
        let candidate = try LESContext(artifact: artifact, prepared: prepared, plan: plan, control: control, syntheticCPUControl: true)
        for (index, token) in [1, 2].enumerated() {
            let actual = try candidate.logits(tokens: [token])
            XCTAssertEqual(actual.count, expected[index].count)
            XCTAssertLessThanOrEqual(zip(actual, expected[index]).map { abs($0 - $1) }.max() ?? .infinity, 0.01)
        }
        try artifact.assertUnchanged()
        XCTAssertEqual(try Data(contentsOf: root.appendingPathComponent("synthetic.gguf")), data)
    }
}
