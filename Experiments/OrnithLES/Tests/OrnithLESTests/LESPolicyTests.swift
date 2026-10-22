import Foundation
import XCTest
import GGUFCore
import StorageCore
import ModelAdapters
import MinirunKit
@testable import OrnithLES

final class LESPolicyTests: XCTestCase {
    private func withPreparationFixture(_ body: (URL, VerifiedGGUFArtifact, Data) throws -> Void) throws {
        let fixtures = try XCTUnwrap(ProcessInfo.processInfo.environment["ORNITH_LES_FIXTURE_ROOT"])
        let data = try Data(contentsOf: URL(fileURLWithPath: fixtures).appendingPathComponent("synthetic-ornith.gguf"))
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("ornith-cache-test-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        try data.write(to: root.appendingPathComponent("synthetic.gguf"))
        let manifest = try GGUFModelManifest(fileName: "synthetic.gguf", fileByteCount: UInt64(data.count),
            sha256: FileDigestComputer.sha256(of: data))
        let artifact = try VerifiedGGUFArtifact.verify(insideRoot: root, manifest: manifest)
        try body(root, artifact, data)
    }
    private func oldRuntimeSeal(_ prepared: URL) throws {
        let url = prepared.appendingPathComponent("seal.json")
        var seal = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        seal["runtimeRevision"] = "previous-installed-engine-revision"
        try JSONSerialization.data(withJSONObject: seal).write(to: url, options: .atomic)
    }
    func testOldRuntimePreparationRebuildsFromOriginalAndKeepsExactLogits() throws {
        try withPreparationFixture { root, artifact, original in
            let cache = root.appendingPathComponent("prepared")
            let prepared = try LESPreparation.prepare(artifact: artifact, privateRoot: cache, control: LESControl())
            let dense = try Data(contentsOf: prepared.appendingPathComponent("dense.gguf"))
            func logits() throws -> [[Float]] {
                let control = LESControl()
                let plan = try LESMemoryPlan(index: artifact.index, contextTokens: 32,
                    footprint: XCTUnwrap(ProcessFootprint.current()))
                let context = try LESContext(artifact: artifact, prepared: prepared, plan: plan,
                    control: control, syntheticCPUControl: true)
                defer { context.close() }
                return try [1, 2].map { try context.logits(tokens: [$0]) }
            }
            let expected = try logits()
            try oldRuntimeSeal(prepared)
            var admissions = 0
            let repaired = try LESPreparation.prepare(artifact: artifact, privateRoot: cache, control: LESControl(),
                onAdmission: { _ in admissions += 1 })
            XCTAssertEqual(repaired, prepared); XCTAssertEqual(admissions, 1, "Regenerate; do not merely re-sign an old seal")
            XCTAssertEqual(try Data(contentsOf: repaired.appendingPathComponent("dense.gguf")), dense)
            XCTAssertEqual(try logits(), expected)
            XCTAssertEqual(try Data(contentsOf: root.appendingPathComponent("synthetic.gguf")), original)
            try artifact.assertUnchanged()
            print("ORNITH_CACHE_UPGRADE oldRuntimeRebuilt=true exactDenseBytes=true exactLogits=true originalUnchanged=true syntheticOnly=true")
        }
    }
    func testPreparationRepairsDamagedMissingAndMalformedDerivedFiles() throws {
        try withPreparationFixture { root, artifact, original in
            let cache = root.appendingPathComponent("prepared")
            let prepared = try LESPreparation.prepare(artifact: artifact, privateRoot: cache, control: LESControl())
            let denseURL = prepared.appendingPathComponent("dense.gguf")
            let manifestURL = prepared.appendingPathComponent("manifest-template.json")
            let sealURL = prepared.appendingPathComponent("seal.json")
            let dense = try Data(contentsOf: denseURL), manifest = try Data(contentsOf: manifestURL)
            for scenario in 0..<5 {
                switch scenario {
                case 0:
                    var damaged = dense; damaged[damaged.count - 1] ^= 1
                    try damaged.write(to: denseURL)
                case 1: try FileManager.default.removeItem(at: manifestURL)
                case 2: try FileManager.default.removeItem(at: sealURL)
                case 3: try Data("interrupted seal write".utf8).write(to: sealURL)
                default:
                    var seal = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: sealURL)) as? [String: Any])
                    seal["sourceBytes"] = 0
                    try JSONSerialization.data(withJSONObject: seal).write(to: sealURL)
                }
                var admissions = 0
                _ = try LESPreparation.prepare(artifact: artifact, privateRoot: cache, control: LESControl(),
                    onAdmission: { _ in admissions += 1 })
                XCTAssertEqual(admissions, 1)
                XCTAssertEqual(try Data(contentsOf: denseURL), dense)
                XCTAssertEqual(try Data(contentsOf: manifestURL), manifest)
                XCTAssertEqual(try Data(contentsOf: root.appendingPathComponent("synthetic.gguf")), original)
            }
            print("ORNITH_CACHE_DAMAGE cases=5 rebuilt=true hashesRetained=true originalUnchanged=true syntheticOnly=true")
        }
    }
    func testPreparationRecoversInterruptedPublicationAndReusesCurrentFiles() throws {
        try withPreparationFixture { root, artifact, original in
            let fm = FileManager.default, cache = root.appendingPathComponent("prepared")
            let prepared = try LESPreparation.prepare(artifact: artifact, privateRoot: cache, control: LESControl())
            let stage = prepared.deletingLastPathComponent().appendingPathComponent("stage-" + artifact.manifest.sha256)
            let previous = prepared.deletingLastPathComponent().appendingPathComponent("previous-" + artifact.manifest.sha256)
            let dense = try Data(contentsOf: prepared.appendingPathComponent("dense.gguf"))
            try fm.copyItem(at: prepared, to: stage)
            try oldRuntimeSeal(prepared)
            try fm.moveItem(at: prepared, to: previous)
            var admissions = 0
            XCTAssertEqual(try LESPreparation.prepare(artifact: artifact, privateRoot: cache, control: LESControl(),
                onAdmission: { _ in admissions += 1 }), prepared)
            XCTAssertEqual(admissions, 0, "A complete, current sealed transaction can be recovered")
            XCTAssertFalse(fm.fileExists(atPath: stage.path)); XCTAssertFalse(fm.fileExists(atPath: previous.path))
            XCTAssertEqual(try Data(contentsOf: prepared.appendingPathComponent("dense.gguf")), dense)
            let seal = try Data(contentsOf: prepared.appendingPathComponent("seal.json"))
            for leftover in [stage, previous] {
                try fm.createDirectory(at: leftover, withIntermediateDirectories: false)
                try Data("incomplete derived file".utf8).write(to: leftover.appendingPathComponent("dense.gguf"))
            }
            _ = try LESPreparation.prepare(artifact: artifact, privateRoot: cache, control: LESControl(),
                onAdmission: { _ in admissions += 1 })
            XCTAssertEqual(admissions, 0)
            XCTAssertEqual(try Data(contentsOf: prepared.appendingPathComponent("seal.json")), seal)
            XCTAssertFalse(fm.fileExists(atPath: stage.path)); XCTAssertFalse(fm.fileExists(atPath: previous.path))
            XCTAssertEqual(try Data(contentsOf: root.appendingPathComponent("synthetic.gguf")), original)
            print("ORNITH_CACHE_INTERRUPTION validatedStageRecovered=true publishedCacheReused=true staleTransactionsCleaned=true originalUnchanged=true syntheticOnly=true")
        }
    }
    func testPreparationCancellationAndDirectorySymlinksCannotDiscardOtherFiles() throws {
        try withPreparationFixture { root, artifact, original in
            let fm = FileManager.default, cache = root.appendingPathComponent("prepared")
            let prepared = try LESPreparation.prepare(artifact: artifact, privateRoot: cache, control: LESControl())
            try oldRuntimeSeal(prepared)
            let seal = try Data(contentsOf: prepared.appendingPathComponent("seal.json"))
            let dense = try Data(contentsOf: prepared.appendingPathComponent("dense.gguf"))
            let cancelled = LESControl()
            XCTAssertThrowsError(try LESPreparation.prepare(artifact: artifact, privateRoot: cache, control: cancelled,
                onAdmission: { _ in cancelled.cancel() }))
            XCTAssertEqual(try Data(contentsOf: prepared.appendingPathComponent("seal.json")), seal)
            XCTAssertEqual(try Data(contentsOf: prepared.appendingPathComponent("dense.gguf")), dense)
            let outside = root.appendingPathComponent("other-files")
            try fm.createDirectory(at: outside, withIntermediateDirectories: false)
            let sentinel = Data("retained unrelated file".utf8)
            try sentinel.write(to: outside.appendingPathComponent("keep.txt"))
            try fm.removeItem(at: prepared)
            try fm.createSymbolicLink(at: prepared, withDestinationURL: outside)
            XCTAssertThrowsError(try LESPreparation.prepare(artifact: artifact, privateRoot: cache, control: LESControl()))
            XCTAssertEqual(try Data(contentsOf: outside.appendingPathComponent("keep.txt")), sentinel)
            XCTAssertEqual(try Data(contentsOf: root.appendingPathComponent("synthetic.gguf")), original)
            print("ORNITH_CACHE_CANCEL oldCacheRetained=true sourceUnchanged=true directorySymlinkRejected=true outsideUntouched=true syntheticOnly=true")
        }
    }
    func testPreparationRejectsChangedOriginalBeforeTouchingPreparedFiles() throws {
        try withPreparationFixture { root, artifact, original in
            let cache = root.appendingPathComponent("prepared")
            let prepared = try LESPreparation.prepare(artifact: artifact, privateRoot: cache, control: LESControl())
            try oldRuntimeSeal(prepared)
            let seal = try Data(contentsOf: prepared.appendingPathComponent("seal.json"))
            let dense = try Data(contentsOf: prepared.appendingPathComponent("dense.gguf"))
            var changed = original; changed.append(1)
            try changed.write(to: root.appendingPathComponent("synthetic.gguf"))
            XCTAssertThrowsError(try LESPreparation.prepare(artifact: artifact, privateRoot: cache, control: LESControl()))
            XCTAssertEqual(try Data(contentsOf: prepared.appendingPathComponent("seal.json")), seal)
            XCTAssertEqual(try Data(contentsOf: prepared.appendingPathComponent("dense.gguf")), dense)
            XCTAssertEqual(try Data(contentsOf: root.appendingPathComponent("synthetic.gguf")), changed)
            print("ORNITH_CACHE_SOURCE changedOriginalRejected=true preparedFilesUntouched=true syntheticOnly=true")
        }
    }
    func testPrefillConfigurationSeparatesLogicalAndPhysicalLimits() throws {
        var options = LESPrefillConfiguration()
        XCTAssertEqual(options.evaluationTokens, 4); XCTAssertEqual(options.physicalTokens, 0)
        XCTAssertEqual(try options.physicalTokens(checkedDefault: 4), 4)
        XCTAssertEqual(try options.physicalTokens(checkedDefault: 1), 1)
        options.evaluationTokens = 32; options.physicalTokens = 16
        try options.validate(); XCTAssertEqual(try options.physicalTokens(checkedDefault: 4), 16)
        options.evaluationTokens = 8; XCTAssertThrowsError(try options.validate())
        options.physicalTokens = 3; XCTAssertThrowsError(try options.validate())
        options.evaluationTokens = 4096; options.physicalTokens = 0; XCTAssertThrowsError(try options.validate())
        for logical in [32, 64, 128, 256, 512, 1024, 2048] {
            options.evaluationTokens = logical; options.physicalTokens = 4
            XCTAssertNoThrow(try options.validate())
        }
        for physical in [128, 256, 512] {
            options.evaluationTokens = 2048; options.physicalTokens = physical
            XCTAssertThrowsError(try options.validate(), "Unsupported physical graphs must not be advertised as optimized")
        }
    }
    func testPublished65KPlanKeepsOriginalRoPEAndIncludesRecurrentCheckpointReserve() throws {
        let root = try XCTUnwrap(ProcessInfo.processInfo.environment["ORNITH_LES_FIXTURE_ROOT"])
        let data = try Data(contentsOf: URL(fileURLWithPath: root).appendingPathComponent("ornith.header"))
        let idx = try GGUFIndex.parse(header: data, fileByteCount: 21_713_463_040)
        let config = try Qwen35MoEConfig(index: idx)
        XCTAssertGreaterThanOrEqual(config.contextLimit, 65536)
        XCTAssertEqual(config.layers, 40); XCTAssertEqual(config.expertsPerToken, 8)
        for context in [32768, 65536] {
            let plan = try LESMemoryPlan(index: idx, contextTokens: context,
                footprint: .init(footprintBytes: 350_000_000, peakFootprintBytes: 350_000_000, availableBytes: nil),
                pageCacheOnly: true, physicalBatchTokens: 4)
            XCTAssertEqual(plan.kvBytes, UInt64(context) * 20480)
            XCTAssertEqual(plan.runtimeReserveBytes, (1 << 30) + (256 << 20))
            XCTAssertLessThanOrEqual(plan.admittedProcessBytes, 5_500_000_000)
            print("ORNITH_LONG_CONTEXT_PLAN context=\(context) dense=\(plan.denseBytes) KV=\(plan.kvBytes) SSM=\(plan.ssmBytes) admitted=\(plan.admittedProcessBytes) ropeBase=\(config.ropeBase) metadataOnly=true physicalQualityVerified=false")
        }
    }
    func testAccelerationDraftHasAnExplicitBoundAndNeverExtendsContext() throws {
        XCTAssertEqual(LESAccelerationBenchmark.draftManifest.fileByteCount, 782_819_680)
        XCTAssertEqual(LESAccelerationBenchmark.draftManifest.sha256,
            "abb048837823d068b71cea4d162076e5e2be62f504dc79bf08ac80d78c8bb732")
        XCTAssertEqual(try LESAccelerationBenchmark.additionalMemory(context: 32768, candidate: .dflash), 1_073_275_232)
        XCTAssertEqual(try LESAccelerationBenchmark.additionalMemory(context: 8192, candidate: .dflash), 972_611_936)
        XCTAssertEqual(try LESAccelerationBenchmark.additionalMemory(context: 65536, candidate: .flashAttention), 0)
        XCTAssertThrowsError(try LESAccelerationBenchmark.additionalMemory(context: 65536, candidate: .dflash))
        XCTAssertThrowsError(try LESAccelerationBenchmark.additionalMemory(context: Int.max, candidate: .dflash))
        let idx = try index()
        let footprint = ProcessFootprint(footprintBytes: 100_000_000, peakFootprintBytes: 100_000_000, availableBytes: nil)
        let ordinary = try LESMemoryPlan(index: idx, contextTokens: 32, footprint: footprint, pageCacheOnly: true)
        let experiment = try LESMemoryPlan(index: idx, contextTokens: 32, footprint: footprint, pageCacheOnly: true,
            experimentReserveBytes: 1_068_032_352)
        XCTAssertEqual(experiment.kvBytes, ordinary.kvBytes)
        XCTAssertEqual(experiment.admittedProcessBytes, ordinary.admittedProcessBytes + 1_068_032_352)
        XCTAssertThrowsError(try LESMemoryPlan(index: idx, contextTokens: 32, footprint: footprint,
            experimentReserveBytes: UInt64.max))
        let report = LESAccelerationBenchmark.Report(candidate: .dflash, contextCapacityTokens: 32768,
            targetSHA256: GGUFModelManifest.primaryOrnith.sha256,
            draftSHA256: LESAccelerationBenchmark.draftManifest.sha256, runs: [], failure: nil)
        XCTAssertFalse(report.comparable)
        XCTAssertFalse(report.fullModelNumericsVerified); XCTAssertFalse(report.fullContextQualityVerified)
    }
    func testLargerPhysicalBatchesBudgetTheCompleteExpertUnion() throws {
        let idx = try index()
        let original = try LESMemoryPlan(index: idx, contextTokens: 32, footprint: footprint, pageCacheOnly: true)
        for physical in [1, 4, 8, 16] {
            let plan = try LESMemoryPlan(index: idx, contextTokens: 32, footprint: footprint,
                pageCacheOnly: true, physicalBatchTokens: physical)
            XCTAssertEqual(plan.prefillSlots, physical * 8)
            XCTAssertEqual(plan.prefillAllocationBytes * 4, original.prefillAllocationBytes * UInt64(physical))
            XCTAssertEqual(plan.expertAllocationBytes, plan.prefillAllocationBytes)
            XCTAssertLessThanOrEqual(plan.admittedProcessBytes, plan.processCeilingBytes)
            if physical > 4 { XCTAssertGreaterThan(plan.runtimeReserveBytes, original.runtimeReserveBytes) }
        }
        XCTAssertThrowsError(try LESMemoryPlan(index: idx, contextTokens: 32, footprint: footprint, physicalBatchTokens: 32))
    }
    func testPublishedHeaderReservesLargeBatchScratchBeforeUsingIOSHeadroom() throws {
        let root = try XCTUnwrap(ProcessInfo.processInfo.environment["ORNITH_LES_FIXTURE_ROOT"])
        let data = try Data(contentsOf: URL(fileURLWithPath: root).appendingPathComponent("ornith.header"))
        let idx = try GGUFIndex.parse(header: data, fileByteCount: 21_713_463_040)
        let baseline: UInt64 = 350_000_000
        func plan(_ physical: Int, available: UInt64? = nil) throws -> LESMemoryPlan {
            try LESMemoryPlan(index: idx, contextTokens: 4096,
                footprint: .init(footprintBytes: baseline, peakFootprintBytes: baseline, availableBytes: available),
                pageCacheOnly: true, physicalBatchTokens: physical)
        }
        let small = try plan(4), large = try plan(16)
        XCTAssertEqual(large.prefillAllocationBytes, small.prefillAllocationBytes * 4)
        XCTAssertGreaterThan(try XCTUnwrap(large.batchScratchReserveBytes), 0)
        let middle = small.admittedProcessBytes + (large.admittedProcessBytes - small.admittedProcessBytes) / 2
        XCTAssertNoThrow(try plan(4, available: middle - baseline))
        XCTAssertThrowsError(try plan(16, available: middle - baseline))
        for physical in [4, 8, 16] {
            let value = try plan(physical)
            print("ORNITH_PREFILL_MEMORY physical=\(physical) staging=\(value.prefillAllocationBytes) extraScratch=\(value.batchScratchReserveBytes ?? 0) admitted=\(value.admittedProcessBytes) metadataOnly=true phonePeak=false")
        }
    }
    func testPrefillProofRequiresActualBatchOriginalExpertSetsAndUnobservedLogits() throws {
        let traced: [String: Any] = ["positions": 4, "maximumAbsoluteError": 0.001, "relativeL2Error": 0.0001,
            "sameGreedy": true, "routingObserved": true, "routingDecisions": 8000, "canonicalExpertSetDigest": 41,
            "configuredPhysicalBatchTokens": 8, "maximumPhysicalBatchTokens": 8]
        let free: [String: Any] = ["positions": 4, "maximumAbsoluteError": 0.001, "relativeL2Error": 0.0001,
            "sameGreedy": true, "routingObserved": false,
            "configuredPhysicalBatchTokens": 8, "maximumPhysicalBatchTokens": 8]
        let raw: [String: Any] = ["modelSHA256": "model", "nativeRevision": "native",
            "referenceRevision": Qwen4ExpCPUReference.sourceRevision,
            "configuration": ["evaluationTokens": 32, "physicalTokens": 8],
            "contextTokens": 4096, "promptTokens": 22, "forwardCount": 25,
            "referenceMetrics": ["denseCalls": 10000, "expertCalls": 3000, "rows": 100000,
                "maximumScratchBytes": 786432, "routingDecisions": 8000, "canonicalRoutingDigest": 23,
                "canonicalExpertSetDigest": 41],
            "observed": traced, "callbackFree": free]
        func proof(_ value: [String: Any]) throws -> LESPrefillCheck {
            try JSONDecoder().decode(LESPrefillCheck.self, from: JSONSerialization.data(withJSONObject: value))
        }
        XCTAssertTrue(try proof(raw).passed)
        for key in ["maximumPhysicalBatchTokens", "configuredPhysicalBatchTokens", "canonicalExpertSetDigest", "routingDecisions"] {
            var changed = raw, observed = traced; observed.removeValue(forKey: key); changed["observed"] = observed
            XCTAssertFalse(try proof(changed).passed)
        }
        var changed = raw, observed = traced; observed["maximumAbsoluteError"] = 0.02; changed["observed"] = observed
        XCTAssertFalse(try proof(changed).passed)
        changed = raw; var callback = free; callback["routingObserved"] = true; changed["callbackFree"] = callback
        XCTAssertFalse(try proof(changed).passed)
        changed = raw; callback = free; callback["maximumPhysicalBatchTokens"] = 4; changed["callbackFree"] = callback
        XCTAssertFalse(try proof(changed).passed)
        print("ORNITH_PREFILL_POLICY actualBatchRequired=true exactExperts=true independentLogits=true callbackFree=true boundedMemory=true syntheticOnly=true")
    }
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
            let result = try [1, 2].map { try reference.logits(for: $0) }
            XCTAssertEqual(reference.dequantizedMetrics.denseCalls, 0)
            XCTAssertEqual(reference.dequantizedMetrics.expertCalls, 0)
            return result
        }
        var referenceExpertSetDigest: UInt64?
        try autoreleasepool {
            let reference = try Qwen4ExpCPUReference(artifact: artifact, contextTokens: 32,
                budgetBytes: 5_500_000_000, arithmetic: .dequantizedWeights)
            for (position, token) in [1, 2].enumerated() {
                let actual = try reference.logits(for: token)
                XCTAssertEqual(actual.count, expected[position].count)
                XCTAssertLessThanOrEqual(zip(actual, expected[position]).map { abs($0 - $1) }.max() ?? .infinity, 0.0001)
            }
            let metrics = reference.dequantizedMetrics
            XCTAssertGreaterThan(metrics.denseCalls, 0); XCTAssertGreaterThanOrEqual(metrics.expertCalls, 24)
            XCTAssertGreaterThan(metrics.rows, 0); XCTAssertGreaterThan(metrics.maximumScratchBytes, 0)
            XCTAssertLessThanOrEqual(metrics.maximumScratchBytes, 1 << 20)
            XCTAssertEqual(metrics.routingDecisions, 16)
            referenceExpertSetDigest = try XCTUnwrap(metrics.canonicalExpertSetDigest)
            let encoded = String(decoding: try JSONEncoder().encode(metrics), as: UTF8.self)
            for key in ["promptTokenIDs", "generatedTokenIDs", "logits", "messages", "path", "values"] {
                XCTAssertFalse(encoded.contains("\"" + key + "\""))
            }
            print("ORNITH_DEQUANTIZED_SWIFT explicitMode=true defaultUnchanged=true boundedScratch=true noPayload=true syntheticOnly=true")
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
        XCTAssertEqual(candidate.metrics.canonicalExpertSetDigest, referenceExpertSetDigest)
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
    func testIndependentArithmeticRequiresCompleteReferenceRoutingAndCallbackFreeTiming() throws {
        let metricsData = Data(#"{"denseCalls":10000,"expertCalls":3000,"rows":1000000,"maximumScratchBytes":786432,"canonicalRoutingDigest":23,"routingDecisions":8000}"#.utf8)
        let metrics = try JSONDecoder().decode(GGUFDequantizedReferenceMetrics.self, from: metricsData)
        let good = LESLogitComparison(positions: 4, maximumAbsoluteError: 0.00001,
            relativeL2Error: 0.000001, sameGreedy: true, routingDigest: 19, routingDecisions: 8000,
            canonicalRoutingDigest: 23, routingObserved: true)
        let free = LESLogitComparison(positions: 4, maximumAbsoluteError: 0.00001,
            relativeL2Error: 0.000001, sameGreedy: true, routingObserved: false)
        let old = LESLogitComparison(positions: 4, maximumAbsoluteError: 3.12,
            relativeL2Error: 0.066, sameGreedy: true, routingDigest: 19, routingDecisions: 8000)
        func proof(batch: Bool) -> LESArithmeticCheck {
            var batched = good
            if !batch { batched.canonicalRoutingDigest = 24 }
            return LESArithmeticCheck(referenceArithmetic: LESArithmeticCheck.arithmetic,
                referenceRevision: Qwen4ExpCPUReference.sourceRevision, forwardCount: 25,
                referenceMetrics: metrics, ordinaryCPUComparison: old,
                noLookahead: good, lookahead: good, batchedPrefill: batched,
                callbackFreeNoLookahead: free, callbackFreeLookahead: free,
                callbackFreeBatchedPrefill: free, callbackFreePageCache: free, pageCachePrefillTokens: batch ? 4 : 1)
        }
        var check = LESNumericalCheck(modelSHA256: "same", referenceRevision: "reference", nativeRevision: "candidate",
            noLookahead: old, lookahead: old, batchedPrefill: old)
        XCTAssertFalse(check.passed); XCTAssertFalse(check.nativeBenchmarkAdmitted)
        for batch in [false, true] {
            let value = proof(batch: batch)
            XCTAssertTrue(value.passed); XCTAssertEqual(value.batchedPrefillPassed, batch)
            XCTAssertEqual(value.prefillTokens, batch ? 4 : 1)
            check.arithmeticCheck = value
            XCTAssertFalse(check.passed, "Keep the original CPU discrepancy visible")
            XCTAssertTrue(check.nativeBenchmarkAdmitted)
            XCTAssertFalse(check.fullModelNumericsVerified)
        }
        let mutations: [(String, String, Any)] = [
            ("", "referenceArithmetic", "unknown"), ("", "referenceRevision", "unknown"),
            ("", "forwardCount", Int.min), ("", "forwardCount", 68), ("", "pageCachePrefillTokens", 4),
            ("referenceMetrics", "denseCalls", 0), ("referenceMetrics", "expertCalls", 0),
            ("referenceMetrics", "maximumScratchBytes", 1048577), ("referenceMetrics", "canonicalRoutingDigest", 24),
            ("referenceMetrics", "routingDecisions", 7999), ("lookahead", "canonicalRoutingDigest", 24),
            ("callbackFreeNoLookahead", "routingObserved", true), ("callbackFreeLookahead", "routingDigest", 19),
            ("callbackFreeLookahead", "maximumAbsoluteError", 0.02), ("callbackFreePageCache", "relativeL2Error", 0.002)
        ]
        for (key, field, value) in mutations {
            var object = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(proof(batch: false))) as? [String: Any])
            if key.isEmpty { object[field] = value }
            else {
                var child = try XCTUnwrap(object[key] as? [String: Any]); child[field] = value; object[key] = child
            }
            let changed = try JSONDecoder().decode(LESArithmeticCheck.self, from: JSONSerialization.data(withJSONObject: object))
            XCTAssertFalse(changed.passed, "\(key).\(field)")
            _ = changed.prefillTokens // Invalid forward counts must not trap.
            check.arithmeticCheck = changed; XCTAssertFalse(check.nativeBenchmarkAdmitted)
        }
        check.arithmeticCheck = proof(batch: false); check.arithmeticCheckUnavailable = true
        XCTAssertFalse(check.nativeBenchmarkAdmitted)
        let encoded = String(decoding: try JSONEncoder().encode(check), as: UTF8.self)
        for key in ["promptTokenIDs", "generatedTokenIDs", "logits", "messages", "path", "values"] {
            XCTAssertFalse(encoded.contains("\"" + key + "\""))
        }
        print("ORNITH_ARITHMETIC_GATE independentReference=true originalDiscrepancyPreserved=true fullRoutingRequired=true callbackFreeRequired=true batchFallback=true syntheticOnly=true")
    }
    func testExactExpertSetsDistinguishRankPermutationWithoutAdmittingChangedExperts() throws {
        let metrics = try JSONDecoder().decode(GGUFDequantizedReferenceMetrics.self, from: Data(#"{"denseCalls":10000,"expertCalls":3000,"rows":1000000,"maximumScratchBytes":786432,"canonicalRoutingDigest":23,"routingDecisions":8000,"canonicalExpertSetDigest":41}"#.utf8))
        let serial = LESLogitComparison(positions: 4, maximumAbsoluteError: 0.00378,
            relativeL2Error: 0.000045, sameGreedy: true, routingDigest: 19, routingDecisions: 8000,
            canonicalRoutingDigest: 24, routingObserved: true, canonicalExpertSetDigest: 41)
        let free = LESLogitComparison(positions: 4, maximumAbsoluteError: 0.00378,
            relativeL2Error: 0.000045, sameGreedy: true, routingObserved: false)
        var batch = serial; batch.canonicalRoutingDigest = 27
        var value = LESArithmeticCheck(referenceArithmetic: LESArithmeticCheck.arithmetic,
            referenceRevision: Qwen4ExpCPUReference.sourceRevision, forwardCount: 25,
            referenceMetrics: metrics, ordinaryCPUComparison: free,
            noLookahead: serial, lookahead: serial, batchedPrefill: batch,
            callbackFreeNoLookahead: free, callbackFreeLookahead: free,
            callbackFreeBatchedPrefill: free, callbackFreePageCache: free, pageCachePrefillTokens: 4,
            routingComparison: LESArithmeticCheck.expertSetRouting)
        XCTAssertTrue(value.passed); XCTAssertTrue(value.batchedPrefillPassed)
        XCTAssertFalse(value.referenceRankedRoutingIdentical); XCTAssertTrue(value.referenceExpertSetsIdentical)
        let mutations: [(String, String, Any)] = [
            ("", "routingComparison", "unknown"),
            ("referenceMetrics", "canonicalExpertSetDigest", 42),
            ("noLookahead", "canonicalExpertSetDigest", 42),
            ("lookahead", "canonicalExpertSetDigest", 42),
            ("lookahead", "canonicalRoutingDigest", 25),
            ("lookahead", "routingDigest", 20),
            ("lookahead", "routingDecisions", 7999),
            ("callbackFreeLookahead", "canonicalExpertSetDigest", 41),
            ("callbackFreeNoLookahead", "maximumAbsoluteError", 0.0101)
        ]
        func changed(_ key: String, _ field: String, _ replacement: Any?) throws -> LESArithmeticCheck {
            var object = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(value)) as? [String: Any])
            if key.isEmpty { object[field] = replacement }
            else {
                var child = try XCTUnwrap(object[key] as? [String: Any]); child[field] = replacement; object[key] = child
            }
            return try JSONDecoder().decode(LESArithmeticCheck.self, from: JSONSerialization.data(withJSONObject: object))
        }
        for (key, field, replacement) in mutations {
            XCTAssertFalse(try changed(key, field, replacement).passed, "\(key).\(field)")
        }
        for key in ["referenceMetrics", "noLookahead", "lookahead"] {
            XCTAssertFalse(try changed(key, "canonicalExpertSetDigest", nil).passed)
        }
        XCTAssertFalse(try changed("", "routingComparison", nil).passed, "Legacy ranked gate retains its original meaning")
        value = try changed("batchedPrefill", "canonicalExpertSetDigest", 42)
        XCTAssertFalse(value.batchedPrefillPassed); XCTAssertEqual(value.prefillTokens, 1)
        value = try changed("", "pageCachePrefillTokens", 1)
        XCTAssertTrue(value.passed, "Changed batch experts must fall back to independently validated serial prefill")
        print("ORNITH_EXPERT_SET_GATE completeSetsRequired=true rankPermutationOnly=true predictionRankedTraceUnchanged=true strictLogits=true missingProofRejected=true batchFallback=true syntheticOnly=true")
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
