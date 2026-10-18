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
        XCTAssertFalse(check.canonicalBatchedRoutingVerified)
        var traced = result; traced.canonicalRoutingDigest = .max
        var batch = traced; batch.routingDigest = 21 // Layer-major observation order differs.
        func tracedCheck(_ last: LESLogitComparison) -> LESNumericalCheck {
            LESNumericalCheck(modelSHA256: "same", referenceRevision: "reference", nativeRevision: "candidate",
                noLookahead: traced, lookahead: traced, batchedPrefill: last)
        }
        XCTAssertTrue(tracedCheck(batch).passed)
        XCTAssertTrue(tracedCheck(batch).canonicalBatchedRoutingVerified)
        var partial = batch; partial.canonicalRoutingDigest = nil
        var changedRoute = batch; changedRoute.canonicalRoutingDigest = 1
        var changedCount = batch; changedCount.routingDecisions = 47
        for invalid in [partial, changedRoute, changedCount, result] {
            XCTAssertFalse(tracedCheck(invalid).passed)
            XCTAssertFalse(tracedCheck(invalid).canonicalBatchedRoutingVerified)
        }
        print("ORNITH_CANONICAL_ROUTING completeBatchTrace=true partialRejected=true changedExpertRejected=true legacyPreserved=true")
        var different = result; different.routingDigest = 20
        XCTAssertFalse(LESNumericalCheck(modelSHA256: "same", referenceRevision: "reference", nativeRevision: "candidate",
            noLookahead: result, lookahead: different, batchedPrefill: result).passed)
        var untraced = result; untraced.routingDecisions = 0
        XCTAssertFalse(LESNumericalCheck(modelSHA256: "same", referenceRevision: "reference", nativeRevision: "candidate",
            noLookahead: untraced, lookahead: untraced, batchedPrefill: result).passed)
        let failed = LESLogitComparison(positions: 4, maximumAbsoluteError: 3.12, relativeL2Error: 0.066,
            sameGreedy: true, routingDigest: 19, routingDecisions: 48)
        var originalFailure = LESNumericalCheck(modelSHA256: "same", referenceRevision: "reference", nativeRevision: "candidate",
            noLookahead: failed, lookahead: result, batchedPrefill: result)
        originalFailure.diagnostics = LESNumericalDiagnostics(instrumentedComparison: result,
            instrumentationMaximumAbsoluteError: 3.12, instrumentationRelativeL2Error: 0.066,
            instrumentationSameGreedy: true, referenceInstrumentationMaximumAbsoluteError: 0,
            referenceInstrumentationRelativeL2Error: 0, referenceInstrumentationSameGreedy: true,
            referenceCheckpointPayloadBytes: 0)
        XCTAssertFalse(originalFailure.passed)
        originalFailure.diagnostics = nil; originalFailure.diagnosticsUnavailable = true
        XCTAssertFalse(originalFailure.passed)
        print("ORNITH_DIAGNOSTIC_GATE originalFailurePreserved=true observedSuccessCannotAdmit=true syntheticOnly=true")
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
        let referencePoints: [[GGUFCheckpoint]] = try autoreleasepool {
            let collector = GGUFCheckpointCollector()
            let reference = try Qwen4ExpCPUReference(artifact: artifact, contextTokens: 32,
                budgetBytes: 5_500_000_000, checkpoints: collector)
            return try [1, 2].enumerated().map { index, token in
                let observed = try reference.logits(for: token)
                XCTAssertEqual(observed, expected[index])
                return try collector.take()
            }
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
        let collector = GGUFCheckpointCollector()
        let candidate = try LESContext(artifact: artifact, prepared: prepared, plan: plan, control: control,
            syntheticCPUControl: true, checkpoints: collector)
        defer { candidate.close() }
        XCTAssertEqual(LESContext.runtimeMemory.liveContexts, 1)
        var layers = LESCheckpointAccumulator()
        for (index, token) in [1, 2].enumerated() {
            let actual = try candidate.logits(tokens: [token])
            XCTAssertEqual(actual.count, expected[index].count)
            XCTAssertLessThanOrEqual(zip(actual, expected[index]).map { abs($0 - $1) }.max() ?? .infinity, 0.01)
            let actualPoints = try collector.take()
            XCTAssertEqual(Set(actualPoints.map(\.key)), Set(referencePoints[index].map(\.key)),
                "Both runtimes must expose the same semantic checkpoints at position \(index)")
            try layers.compare(referencePoints[index], actualPoints, position: index)
        }
        XCTAssertTrue(layers.result.complete,
            "Missing points: \(layers.result.missingPoints); shape mismatches: \(layers.result.shapeMismatches)")
        XCTAssertGreaterThan(layers.result.comparedPoints, 20)
        XCTAssertLessThan(layers.result.maximumAbsoluteError, 1e-6)
        XCTAssertNil(layers.result.firstDivergence)
        let probe = try XCTUnwrap(candidate.projectionProbe)
        XCTAssertEqual(probe.samples, 16); XCTAssertEqual(probe.columns, 16)
        XCTAssertLessThan(probe.backendRelativeL2Error, 1e-5)
        let diagnostic = LESLogitComparison(positions: 4, maximumAbsoluteError: 0, relativeL2Error: 0,
            sameGreedy: true, layerCheckpoints: layers.result, projectionProbe: probe)
        let encoded = String(decoding: try JSONEncoder().encode(diagnostic), as: UTF8.self)
        for key in ["values", "logits", "promptTokenIDs", "generatedTokenIDs", "messages"] {
            XCTAssertFalse(encoded.contains("\"" + key + "\""))
        }
        try artifact.assertUnchanged()
        XCTAssertEqual(try Data(contentsOf: root.appendingPathComponent("synthetic.gguf")), data)
        candidate.close(); candidate.close()
        XCTAssertThrowsError(try candidate.logits(tokens: [1]))
        // Keep the Swift wrapper alive while admitting its next native owner.
        // ARC scope or an outer autorelease pool must not govern this boundary.
        for _ in 0..<3 {
            XCTAssertEqual(LESContext.runtimeMemory.liveContexts, 0)
            XCTAssertEqual(LESContext.runtimeMemory.liveMetalBuffers, 0)
            let next = try LESContext(artifact: artifact, prepared: prepared, plan: plan,
                control: control, syntheticCPUControl: true)
            defer { next.close() }
            XCTAssertEqual(LESContext.runtimeMemory.liveContexts, 1)
            let logits = try next.logits(tokens: [1])
            XCTAssertLessThanOrEqual(zip(logits, expected[0]).map { abs($0 - $1) }.max() ?? .infinity, 0.01)
            next.close()
        }
        withExtendedLifetime(candidate) {
            XCTAssertEqual(LESContext.runtimeMemory.liveContexts, 0)
            XCTAssertEqual(LESContext.runtimeMemory.liveMetalBuffers, 0)
            XCTAssertEqual(LESContext.runtimeMemory.ownedMetalBytes, 0)
            XCTAssertEqual(LESContext.runtimeMemory.mappedMetalBytes, 0)
        }
        print("ORNITH_NATIVE_LIFETIME explicitClose=true wrapperStillAlive=true repeatedOwners=3 noLiveBuffers=true syntheticOnly=true")
        print("ORNITH_LAYER_CHECKPOINTS exactReplay=true bounded=true failClosed=true noPayload=true syntheticOnly=true")
    }
    func testLayerCheckpointBoundsAndFirstDivergence() throws {
        let values: [Float] = [1, 2, 3]
        for (stage, layer, count) in [(UInt32(0), Int32(0), Int32(3)), (11, 0, 3), (10, 1, 3), (2, 40, 3), (1, 0, 3), (2, 0, 16385)] {
            let collector = GGUFCheckpointCollector()
            XCTAssertFalse(values.withUnsafeBufferPointer { collector.capture(stage: stage, layer: layer, values: $0.baseAddress, count: count) })
            XCTAssertThrowsError(try collector.take())
        }
        let duplicate = GGUFCheckpointCollector()
        XCTAssertTrue(values.withUnsafeBufferPointer { duplicate.capture(stage: 2, layer: 0, values: $0.baseAddress, count: 3) })
        XCTAssertFalse(values.withUnsafeBufferPointer { duplicate.capture(stage: 2, layer: 0, values: $0.baseAddress, count: 3) })
        XCTAssertThrowsError(try duplicate.take())
        let bounded = GGUFCheckpointCollector(), large = [Float](repeating: 1, count: 16384)
        for index in 0..<64 {
            XCTAssertTrue(large.withUnsafeBufferPointer {
                bounded.capture(stage: index < 40 ? 2 : 5, layer: Int32(index % 40), values: $0.baseAddress, count: 16384)
            })
        }
        XCTAssertFalse(values.withUnsafeBufferPointer { bounded.capture(stage: 6, layer: 0, values: $0.baseAddress, count: 3) })
        XCTAssertThrowsError(try bounded.take())
        var difference = LESCheckpointAccumulator()
        let expected = [GGUFCheckpoint(stage: 2, layer: 0, values: values), GGUFCheckpoint(stage: 8, layer: 0, values: values)]
        try difference.compare(expected, [GGUFCheckpoint(stage: 8, layer: 0, values: [2, 2, 3]),
            GGUFCheckpoint(stage: 2, layer: 0, values: [1, 2, 3.1])], position: 7)
        XCTAssertTrue(difference.result.complete)
        XCTAssertEqual(difference.result.firstDivergence?.stage, 2)
        XCTAssertEqual(difference.result.firstDivergence?.position, 7)
        let recurrent = GGUFCheckpoint(stage: 10, layer: 0, values: values)
        let projected = GGUFCheckpoint(stage: 5, layer: 0, values: values)
        XCTAssertLessThan(recurrent.order, projected.order)
        var attention = LESCheckpointAccumulator()
        try attention.compare([projected, recurrent], [
            GGUFCheckpoint(stage: 5, layer: 0, values: [2, 2, 3]),
            GGUFCheckpoint(stage: 10, layer: 0, values: [1, 2, 3.1])], position: 0)
        XCTAssertEqual(attention.result.firstDivergence?.stage, 10)
        var missing = LESCheckpointAccumulator()
        try missing.compare(expected, [GGUFCheckpoint(stage: 2, layer: 0, values: [1])], position: 0)
        XCTAssertFalse(missing.result.complete)
        XCTAssertEqual(missing.result.missingPoints, 1); XCTAssertEqual(missing.result.shapeMismatches, 1)
    }
}
