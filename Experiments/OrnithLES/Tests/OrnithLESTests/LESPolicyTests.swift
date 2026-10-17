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
        XCTAssertEqual(off.cacheSlotsPerLayer, on.cacheSlotsPerLayer + 4)
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
    func testNoExpertCapacityMayBeBorrowedFromOtherReservations() throws {
        let sample = ProcessFootprint(footprintBytes: 5_000_000_000, peakFootprintBytes: 5_000_000_000, availableBytes: nil)
        XCTAssertThrowsError(try LESMemoryPlan(index: index(), contextTokens: 32, footprint: sample))
    }
    func testInvalidContextAndLookaheadAreRejected() throws {
        for context in [0, 31, 65537] {
            XCTAssertThrowsError(try LESMemoryPlan(index: index(), contextTokens: context, footprint: footprint))
        }
        XCTAssertThrowsError(try LESMemoryPlan(index: index(), contextTokens: 32, footprint: footprint, lookaheadDepth: 17))
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
        let result = LESLogitComparison(positions: 4, maximumAbsoluteError: 0, relativeL2Error: 0, sameGreedy: true)
        let check = LESNumericalCheck(modelSHA256: String(repeating: "a", count: 64), referenceRevision: "reference",
            nativeRevision: "candidate", noLookahead: result, lookahead: result, batchedPrefill: result)
        let encoded = String(decoding: try JSONEncoder().encode(check), as: UTF8.self)
        for key in ["promptTokenIDs", "generatedTokenIDs", "logits", "chat", "messages"] { XCTAssertFalse(encoded.contains("\"" + key + "\"")) }
        XCTAssertTrue(check.passed); XCTAssertFalse(check.fullModelNumericsVerified)
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
        let prepared = try LESPreparation.prepare(artifact: artifact, privateRoot: root.appendingPathComponent("prepared"), control: control)
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
