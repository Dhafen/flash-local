import Foundation
import Darwin
import GGUFCore
import MinirunKit
import MLX
import ModelAdapters
import StorageCore

public enum LESBenchmarkBackend: String, Codable, CaseIterable, Sendable {
    case existingGGUF, omniCache, omniLookahead, omniPageCache
}
public struct LESLogitComparison: Codable, Sendable, Equatable {
    public let positions: Int
    public let maximumAbsoluteError: Double
    public let relativeL2Error: Double
    public let sameGreedy: Bool
    public var routingDigest: UInt64? = nil
    public var routingDecisions: UInt64? = nil
    public var canonicalRoutingDigest: UInt64? = nil
    public var layerCheckpoints: LESCheckpointComparison? = nil
    public var projectionProbe: LESProjectionProbe? = nil
    public var activeLookaheadDepth: Int? = nil
    public var routingObserved: Bool? = nil
    public var canonicalExpertSetDigest: UInt64? = nil
    public var configuredPhysicalBatchTokens: Int? = nil
    public var maximumPhysicalBatchTokens: Int? = nil
    public var callbackFree: Bool {
        routingObserved == false && routingDigest == nil && routingDecisions == nil &&
            canonicalRoutingDigest == nil && canonicalExpertSetDigest == nil && layerCheckpoints == nil && projectionProbe == nil
    }
    public var passed: Bool {
        positions == 4 && maximumAbsoluteError.isFinite && relativeL2Error.isFinite &&
            (0...0.01).contains(maximumAbsoluteError) && (0...0.001).contains(relativeL2Error) && sameGreedy
    }
}
public struct LESNumericalCheck: Codable, Sendable {
    public let modelSHA256: String
    public let referenceRevision: String
    public let nativeRevision: String
    public let noLookahead: LESLogitComparison
    public let lookahead: LESLogitComparison
    public let batchedPrefill: LESLogitComparison
    public var diagnostics: LESNumericalDiagnostics? = nil
    public var diagnosticsUnavailable: Bool? = nil
    public var arithmeticCheck: LESArithmeticCheck? = nil
    public var arithmeticCheckUnavailable: Bool? = nil
    public let fullModelNumericsVerified = false
    /// Archived .passed retains the original CPU comparison's exact meaning.
    /// New timing requires the identified third oracle and callback-free modes.
    public var nativeBenchmarkAdmitted: Bool { arithmeticCheckUnavailable != true && arithmeticCheck?.passed == true }
    public var nativePrefillTokens: Int { arithmeticCheck?.prefillTokens ?? 1 }
    public var canonicalBatchedRoutingVerified: Bool {
        noLookahead.canonicalRoutingDigest != nil &&
            noLookahead.routingDecisions.map({ $0 > 0 }) == true &&
            noLookahead.canonicalRoutingDigest == lookahead.canonicalRoutingDigest &&
            noLookahead.canonicalRoutingDigest == batchedPrefill.canonicalRoutingDigest &&
            noLookahead.routingDecisions == lookahead.routingDecisions &&
            noLookahead.routingDecisions == batchedPrefill.routingDecisions
    }
    public var passed: Bool {
        noLookahead.passed && lookahead.passed && batchedPrefill.passed &&
            noLookahead.routingDecisions.map({ $0 > 0 }) == true &&
            noLookahead.routingDigest != nil && noLookahead.routingDigest == lookahead.routingDigest &&
            noLookahead.routingDecisions == lookahead.routingDecisions &&
            // Archived reports before this observer had no canonical checksum.
            // A partially present or inconsistent new trace cannot pass.
            ([noLookahead, lookahead, batchedPrefill].allSatisfy { $0.canonicalRoutingDigest == nil } ||
                canonicalBatchedRoutingVerified)
    }
}
public struct LESBenchmarkRun: Codable, Sendable {
    public let backend: LESBenchmarkBackend
    public let applicationCache: String
    public let osCacheControlled = false
    public let promptTokens: Int
    public let generatedTokens: Int
    public let decodeForwardPasses: Int
    public let prefillSeconds: Double
    public let decodeForwardSeconds: Double
    public let decodeIOBytes: UInt64
    public let prefillIOBytes: UInt64
    public let outputTokenSHA256: String
    public let maximumObservedFootprintBytes: UInt64
    public let thermalStateAtStart: Int
    public let maximumThermalState: Int
    public let stopReason: String
    public let memoryPlan: LESMemoryPlan?
    public let metricsBefore: LESMetrics?
    public let metricsAfterPrefill: LESMetrics?
    public let metricsAfter: LESMetrics?
    public let originalProjectionsBefore: GGUFProjectionStatistics?
    public let originalProjectionsAfterPrefill: GGUFProjectionStatistics?
    public let originalProjectionsAfter: GGUFProjectionStatistics?
    public let originalCacheBefore: GGUFResidentWeightCacheStatistics?
    public let originalCacheAfterPrefill: GGUFResidentWeightCacheStatistics?
    public let originalCacheAfter: GGUFResidentWeightCacheStatistics?
    public var prefillBatchTokens: Int? = nil
    public var decodeTokensPerSecond: Double { Double(decodeForwardPasses) / max(decodeForwardSeconds, 1e-12) }
    public var prefillTokensPerSecond: Double { Double(promptTokens) / max(prefillSeconds, 1e-12) }
    public var hasRequiredLength: Bool { generatedTokens >= 128 && decodeForwardPasses >= 128 }
}
public enum LESBenchmark {
    public static let promptID = "ornith-lru-v1"
    public static let prompt = "Write a detailed, numbered explanation of how to implement and test a bounded LRU cache in Python. Include working code, invariants, edge cases, and at least twenty distinct test ideas. Continue for at least 400 words."
    private struct Replay {
        let prompt: [Int]
        let forced: [Int]
        let logits: [[Float]]
        let checkpoints: [[GGUFCheckpoint]]
        var dequantizedMetrics: GGUFDequantizedReferenceMetrics? = nil
    }
    private static func greedy(_ values: [Float]) throws -> Int {
        guard !values.isEmpty, values.allSatisfy(\.isFinite) else { throw GGUFError.invalid("Non-finite benchmark logits") }
        return values.indices.max { values[$0] < values[$1] }!
    }
    private static func observerEffect(_ before: [[Float]], _ after: [[Float]]) throws -> LESLogitComparison {
        guard before.count == 4, after.count == 4 else { throw GGUFError.invalid("Incomplete observer comparison") }
        var absolute = 0.0, squared = 0.0, reference = 0.0, same = true
        for (a, b) in zip(before, after) {
            guard a.count == b.count else { throw GGUFError.invalid("Observer vocabulary shape") }
            same = try same && greedy(a) == greedy(b)
            for (x, y) in zip(a, b) {
                let d = Double(x) - Double(y)
                absolute = max(absolute, abs(d)); squared += d * d; reference += Double(x) * Double(x)
            }
        }
        return LESLogitComparison(positions: 4, maximumAbsoluteError: absolute,
            relativeL2Error: sqrt(squared / max(reference, 1e-30)), sameGreedy: same)
    }
    private static func reclaimIdleBuffers() -> (ProcessFootprint?, ProcessFootprint?) {
        let before = ProcessFootprint.current()
        // Called only after the preceding inference owner and its autorelease
        // pool have ended, never between a cold/warm pair or during a forward.
        MLX.Memory.clearCache()
        _ = malloc_zone_pressure_relief(nil, 0)
        return (before, ProcessFootprint.current())
    }
    public static func plan(_ artifact: VerifiedGGUFArtifact, context: Int, depth: Int,
                             pageCache: Bool = false, phase: String,
                             physicalBatchTokens: Int = 4,
                             onAdmission: (LESMemoryAdmission) -> Void) throws -> LESMemoryPlan {
        let samples = reclaimIdleBuffers()
        let nativeMemory = LESContext.runtimeMemory
        do {
            guard nativeMemory.liveContexts == 0, nativeMemory.liveMetalBuffers == 0 else {
                throw GGUFError.limit("Previous LES owner has not released its native buffers")
            }
            guard let footprint = samples.1 else { throw GGUFError.limit("LES memory sample unavailable") }
            let result = try LESMemoryPlan(index: artifact.index, contextTokens: context, footprint: footprint,
                lookaheadDepth: depth, pageCacheOnly: pageCache, physicalBatchTokens: physicalBatchTokens)
            onAdmission(.init(phase: phase, beforeReclaim: samples.0, afterReclaim: samples.1, plan: result, failure: nil,
                nativeMemory: nativeMemory))
            return result
        } catch {
            onAdmission(.init(phase: phase, beforeReclaim: samples.0, afterReclaim: samples.1, plan: nil,
                failure: error.localizedDescription, nativeMemory: nativeMemory))
            throw GGUFError.limit("\(phase): \(error.localizedDescription)")
        }
    }
    private static func reference(artifact: VerifiedGGUFArtifact, prompt: [Int], control: LESControl,
                                  traceLayers: Bool = false, forcedTokens: [Int]? = nil,
                                  arithmetic: Qwen4ExpCPUReference.Arithmetic = .pinnedCPU) throws -> Replay {
        guard forcedTokens == nil || forcedTokens?.count == 3 else { throw GGUFError.invalid("Diagnostic forced sequence length") }
        let collector = traceLayers ? GGUFCheckpointCollector() : nil
        let context = try Qwen4ExpCPUReference(artifact: artifact, contextTokens: 128, budgetBytes: 5_500_000_000,
            checkpoints: collector, arithmetic: arithmetic,
            cancellationRequested: { control.shouldAbort() })
        var logits: [Float] = []
        var checkpoints: [[GGUFCheckpoint]] = [], checkpointBytes = 0
        func capture() throws {
            guard let collector else { return }
            let step = try collector.take()
            let bytes = step.reduce(0) { $0 + $1.values.count * 4 }
            guard checkpointBytes <= (64 << 20) - bytes else { throw GGUFError.limit("Reference layer trace exceeds 64 MiB") }
            checkpointBytes += bytes; checkpoints.append(step)
        }
        for token in prompt { logits = try context.logits(for: token); try capture() }
        var vectors = [logits], forced: [Int] = []
        for index in 0..<3 {
            let token: Int
            if let forcedTokens { token = forcedTokens[index] } else { token = try greedy(logits) }
            forced.append(token)
            logits = try context.logits(for: token); vectors.append(logits); try capture()
        }
        return Replay(prompt: prompt, forced: forced, logits: vectors, checkpoints: checkpoints,
            dequantizedMetrics: arithmetic == .dequantizedWeights ? context.dequantizedMetrics : nil)
    }
    private static func compare(_ replay: Replay, artifact: VerifiedGGUFArtifact, prepared: URL,
                                depth: Int, prefillBatch: Int = 1, traceLayers: Bool = false,
                                phase: String = "numerics", traceRouting: Bool = true,
                                pageCache: Bool = false, physicalBatchTokens: Int = 4,
                                contextTokens: Int = 128, control: LESControl,
                                onAdmission: (LESMemoryAdmission) -> Void,
                                recordLogits: (([Float]) -> Void)? = nil) throws -> LESLogitComparison {
        let memory = try plan(artifact, context: contextTokens, depth: depth, pageCache: pageCache,
            phase: "\(phase).prediction\(depth).prefill\(prefillBatch)", physicalBatchTokens: physicalBatchTokens,
            onAdmission: onAdmission)
        let collector = traceLayers && prefillBatch == 1 ? GGUFCheckpointCollector() : nil
        let context = try LESContext(artifact: artifact, prepared: prepared, plan: memory, control: control,
            traceRouting: traceRouting, checkpoints: collector)
        defer { context.close() }
        var logits: [Float] = []
        var checkpoints = LESCheckpointAccumulator()
        for first in stride(from: 0, to: replay.prompt.count, by: prefillBatch) {
            logits = try context.logits(tokens: Array(replay.prompt[first..<min(first + prefillBatch, replay.prompt.count)]))
            if let collector { try checkpoints.compare(replay.checkpoints[first], collector.take(), position: first) }
        }
        var absolute = 0.0, squared = 0.0, referenceSquared = 0.0, same = true
        for position in 0..<4 {
            if position > 0 {
                logits = try context.logits(tokens: [replay.forced[position - 1]])
                if let collector {
                    let input = replay.prompt.count + position - 1
                    try checkpoints.compare(replay.checkpoints[input], collector.take(), position: input)
                }
            }
            let expected = replay.logits[position]
            recordLogits?(logits)
            guard logits.count == expected.count else { throw GGUFError.invalid("LES reference vocabulary shape") }
            same = try same && greedy(logits) == greedy(expected)
            for (a, b) in zip(logits, expected) {
                let difference = Double(a) - Double(b)
                absolute = max(absolute, abs(difference)); squared += difference * difference
                referenceSquared += Double(b) * Double(b)
            }
        }
        return LESLogitComparison(positions: 4, maximumAbsoluteError: absolute,
            relativeL2Error: sqrt(squared / max(referenceSquared, 1e-30)), sameGreedy: same,
            routingDigest: traceRouting || collector != nil ? context.metrics.routingDigest : nil,
            routingDecisions: traceRouting || collector != nil ? context.metrics.routingDecisions : nil,
            canonicalRoutingDigest: traceRouting || collector != nil ? context.metrics.canonicalRoutingDigest : nil,
            layerCheckpoints: collector == nil ? nil : checkpoints.result,
            projectionProbe: context.projectionProbe,
            activeLookaheadDepth: memory.lookaheadDepth, routingObserved: traceRouting || collector != nil,
            canonicalExpertSetDigest: traceRouting || collector != nil ? context.metrics.canonicalExpertSetDigest : nil,
            configuredPhysicalBatchTokens: context.metrics.configuredPhysicalBatchTokens,
            maximumPhysicalBatchTokens: context.metrics.maximumPhysicalBatchTokens)
    }
    private static func rebase(_ routing: LESLogitComparison, logits: [[Float]],
                               reference: Replay) throws -> LESLogitComparison {
        var result = try observerEffect(reference.logits, logits)
        result.routingDigest = routing.routingDigest; result.routingDecisions = routing.routingDecisions
        result.canonicalRoutingDigest = routing.canonicalRoutingDigest
        result.activeLookaheadDepth = routing.activeLookaheadDepth; result.routingObserved = routing.routingObserved
        result.canonicalExpertSetDigest = routing.canonicalExpertSetDigest
        return result
    }
    private static func arithmeticCheck(_ original: Replay, off: LESLogitComparison, on: LESLogitComparison,
                                        batch: LESLogitComparison, offLogits: [[Float]], onLogits: [[Float]],
                                        batchLogits: [[Float]], artifact: VerifiedGGUFArtifact, prepared: URL,
                                        control: LESControl, progress: @Sendable (String) -> Void,
                                        onAdmission: (LESMemoryAdmission) -> Void) throws -> LESArithmeticCheck {
        progress("Kontrollerar ursprungliga vikter utan extra aktiveringskvantisering…")
        let exact = try autoreleasepool {
            try reference(artifact: artifact, prompt: original.prompt, control: control,
                forcedTokens: original.forced, arithmetic: .dequantizedWeights)
        }
        guard let metrics = exact.dequantizedMetrics else { throw GGUFError.invalid("Missing third-oracle coverage") }
        let exactOff = try rebase(off, logits: offLogits, reference: exact)
        let exactOn = try rebase(on, logits: onLogits, reference: exact)
        let exactBatch = try rebase(batch, logits: batchLogits, reference: exact)
        progress("Kontrollerar körningen utan profileringsåteranrop…")
        let freeOff = try autoreleasepool {
            try compare(exact, artifact: artifact, prepared: prepared, depth: 0,
                phase: "unobserved", traceRouting: false, control: control, onAdmission: onAdmission)
        }
        let freeOn = try autoreleasepool {
            try compare(exact, artifact: artifact, prepared: prepared, depth: 4,
                phase: "unobserved", traceRouting: false, control: control, onAdmission: onAdmission)
        }
        let freeBatch = try autoreleasepool {
            try compare(exact, artifact: artifact, prepared: prepared, depth: 0, prefillBatch: 4,
                phase: "unobserved", traceRouting: false, control: control, onAdmission: onAdmission)
        }
        func proof(page: LESLogitComparison, prefill: Int) throws -> LESArithmeticCheck {
            LESArithmeticCheck(referenceArithmetic: LESArithmeticCheck.arithmetic,
                referenceRevision: Qwen4ExpCPUReference.sourceRevision,
                forwardCount: original.prompt.count + original.forced.count, referenceMetrics: metrics,
                ordinaryCPUComparison: try observerEffect(exact.logits, original.logits),
                noLookahead: exactOff, lookahead: exactOn, batchedPrefill: exactBatch,
                callbackFreeNoLookahead: freeOff, callbackFreeLookahead: freeOn,
                callbackFreeBatchedPrefill: freeBatch, callbackFreePageCache: page,
                pageCachePrefillTokens: prefill, routingComparison: LESArithmeticCheck.expertSetRouting)
        }
        let prefill = try proof(page: freeOff, prefill: 1).prefillTokens
        progress("Kontrollerar OS-cache med \(prefill) prompttoken per steg…")
        let freePage = try autoreleasepool {
            try compare(exact, artifact: artifact, prepared: prepared, depth: 0, prefillBatch: prefill,
                phase: "pagecache", traceRouting: false, pageCache: true, control: control, onAdmission: onAdmission)
        }
        var result = try proof(page: freePage, prefill: prefill)
        if !result.passed {
            progress("Lokaliserar avvikelsen mot avkvantiserade originalvikter…")
            do {
                let traced = try autoreleasepool {
                    try reference(artifact: artifact, prompt: original.prompt, control: control, traceLayers: true,
                        forcedTokens: original.forced, arithmetic: .dequantizedWeights)
                }
                var observed: [[Float]] = []
                let comparison = try autoreleasepool {
                    try compare(traced, artifact: artifact, prepared: prepared, depth: 0, traceLayers: true,
                        phase: "dequantized", control: control, onAdmission: onAdmission,
                        recordLogits: { observed.append($0) })
                }
                let nativeEffect = try observerEffect(offLogits, observed)
                let referenceEffect = try observerEffect(exact.logits, traced.logits)
                result.diagnostics = LESNumericalDiagnostics(instrumentedComparison: comparison,
                    instrumentationMaximumAbsoluteError: nativeEffect.maximumAbsoluteError,
                    instrumentationRelativeL2Error: nativeEffect.relativeL2Error,
                    instrumentationSameGreedy: nativeEffect.sameGreedy,
                    referenceInstrumentationMaximumAbsoluteError: referenceEffect.maximumAbsoluteError,
                    referenceInstrumentationRelativeL2Error: referenceEffect.relativeL2Error,
                    referenceInstrumentationSameGreedy: referenceEffect.sameGreedy,
                    referenceCheckpointPayloadBytes: traced.checkpoints.reduce(0) { $0 + $1.reduce(0) { $0 + $1.values.count * 4 } })
            } catch { result.diagnosticsUnavailable = true }
        }
        return result
    }
    public static func checkPrefill(artifact: VerifiedGGUFArtifact, prepared: URL,
                                   baseline: LESNumericalCheck, configuration: LESPrefillConfiguration,
                                   control: LESControl, progress: @Sendable (String) -> Void = { _ in },
                                   onAdmission: (LESMemoryAdmission) -> Void = { _ in }) throws -> LESPrefillCheck {
        try configuration.validate()
        guard baseline.nativeBenchmarkAdmitted, baseline.modelSHA256 == artifact.manifest.sha256,
              baseline.nativeRevision == LESContext.revision, configuration.physicalTokens > 0 else {
            throw GGUFError.invalid("Batchkontrollen kräver godkänd grundmotor och en uttrycklig fysisk batch.")
        }
        let tokenizer = try CanonicalQwen35Tokenizer(artifact: artifact)
        let prompt = try tokenizer.chat([.init(role: .user, content: "Vad är 50 + 5?")], maximumTokens: 64).tokenIDs
        guard prompt.count >= max(16, configuration.physicalTokens) else { throw GGUFError.invalid("Batch control prompt is too short") }
        progress("Kontrollerar batch \(configuration.evaluationTokens)/\(configuration.physicalTokens) mot originalvikterna…")
        let replay = try autoreleasepool {
            try reference(artifact: artifact, prompt: prompt, control: control, arithmetic: .dequantizedWeights)
        }
        guard let metrics = replay.dequantizedMetrics else { throw GGUFError.invalid("Batch reference coverage missing") }
        func run(observed: Bool) throws -> LESLogitComparison {
            try autoreleasepool {
                try compare(replay, artifact: artifact, prepared: prepared, depth: 0,
                    prefillBatch: configuration.evaluationTokens, phase: "prefill-tuning", traceRouting: observed,
                    pageCache: true, physicalBatchTokens: configuration.physicalTokens, contextTokens: 4096,
                    control: control, onAdmission: onAdmission)
            }
        }
        let observed = try run(observed: true), free = try run(observed: false)
        return .init(modelSHA256: artifact.manifest.sha256, nativeRevision: LESContext.revision,
            referenceRevision: Qwen4ExpCPUReference.sourceRevision, configuration: configuration,
            contextTokens: 4096, promptTokens: prompt.count, forwardCount: prompt.count + 3,
            referenceMetrics: metrics, observed: observed, callbackFree: free)
    }

    public static func check(artifact: VerifiedGGUFArtifact, prepared: URL, control: LESControl,
                             progress: @Sendable (String) -> Void = { _ in },
                             onAdmission: (LESMemoryAdmission) -> Void = { _ in }) throws -> LESNumericalCheck {
        let tokenizer = try CanonicalQwen35Tokenizer(artifact: artifact)
        let tokens = try tokenizer.chat([.init(role: .user, content: "Vad är 50 + 5?")], maximumTokens: 64).tokenIDs
        progress("Jämför hela modellgrafen med CPU-referensen…")
        let replay = try autoreleasepool { try reference(artifact: artifact, prompt: tokens, control: control) }
        progress("Kontrollerar LES utan förläsning…")
        var routedOffLogits: [[Float]] = [], onLogits: [[Float]] = [], batchLogits: [[Float]] = []
        let off = try autoreleasepool { try compare(replay, artifact: artifact, prepared: prepared, depth: 0,
            control: control, onAdmission: onAdmission, recordLogits: { routedOffLogits.append($0) }) }
        progress("Kontrollerar LES med förläsning…")
        let on = try autoreleasepool { try compare(replay, artifact: artifact, prepared: prepared, depth: 4,
            control: control, onAdmission: onAdmission, recordLogits: { onLogits.append($0) }) }
        progress("Kontrollerar batchad promptbearbetning…")
        let batch = try autoreleasepool { try compare(replay, artifact: artifact, prepared: prepared, depth: 0,
            prefillBatch: 4, control: control, onAdmission: onAdmission, recordLogits: { batchLogits.append($0) }) }
        var result = LESNumericalCheck(modelSHA256: artifact.manifest.sha256,
            referenceRevision: Qwen4ExpCPUReference.sourceRevision, nativeRevision: LESContext.revision,
            noLookahead: off, lookahead: on, batchedPrefill: batch)
        do {
            result.arithmeticCheck = try arithmeticCheck(replay, off: off, on: on, batch: batch,
                offLogits: routedOffLogits, onLogits: onLogits, batchLogits: batchLogits,
                artifact: artifact, prepared: prepared, control: control, progress: progress, onAdmission: onAdmission)
        } catch {
            // Keep both the original comparison and the failed independent
            // admission. Never admit from the small matched-input probe alone.
            result.arithmeticCheckUnavailable = true
        }
        if !result.passed && result.arithmeticCheck == nil {
            // Observation can alter backend fusion. It is never the run that
            // admits timing; retain the original failed gate unchanged.
            progress("Lokaliserar avvikelsen med separat lagerdiagnostik…")
            do {
                let diagnosticReplay = try autoreleasepool {
                    try reference(artifact: artifact, prompt: tokens, control: control, traceLayers: true, forcedTokens: replay.forced)
                }
                var observed: [[Float]] = []
                let comparison = try autoreleasepool {
                    try compare(diagnosticReplay, artifact: artifact, prepared: prepared, depth: 0,
                        traceLayers: true, phase: "diagnostics", control: control, onAdmission: onAdmission,
                        recordLogits: { observed.append($0) })
                }
                let nativeEffect = try observerEffect(routedOffLogits, observed)
                let referenceEffect = try observerEffect(replay.logits, diagnosticReplay.logits)
                result.diagnostics = LESNumericalDiagnostics(instrumentedComparison: comparison,
                    instrumentationMaximumAbsoluteError: nativeEffect.maximumAbsoluteError,
                    instrumentationRelativeL2Error: nativeEffect.relativeL2Error,
                    instrumentationSameGreedy: nativeEffect.sameGreedy,
                    referenceInstrumentationMaximumAbsoluteError: referenceEffect.maximumAbsoluteError,
                    referenceInstrumentationRelativeL2Error: referenceEffect.relativeL2Error,
                    referenceInstrumentationSameGreedy: referenceEffect.sameGreedy,
                    referenceCheckpointPayloadBytes: diagnosticReplay.checkpoints.reduce(0) { $0 + $1.reduce(0) { $0 + $1.values.count * 4 } })
            } catch {
                // Preserve the complete original numerical failure even when
                // extra diagnostics cannot fit or are unsupported/cancelled.
                result.diagnosticsUnavailable = true
            }
        }
        return result
    }

    /// Fresh state on every run. Two runs on the same engine distinguish an
    /// empty application cache from a warm application cache, not OS coldness.
    /// Profiling uses genuine greedy generation and refuses early EOS as a
    /// comparable 128-token result. No EOS suppression or routing change.
    public static func runPair(backend: LESBenchmarkBackend, artifact: VerifiedGGUFArtifact,
                               prepared: URL, check: LESNumericalCheck, control: LESControl,
                               progress: @Sendable (String) -> Void = { _ in },
                               onAdmission: (LESMemoryAdmission) -> Void = { _ in }) throws -> [LESBenchmarkRun] {
        guard check.modelSHA256 == artifact.manifest.sha256,
              backend == .existingGGUF || check.nativeBenchmarkAdmitted else {
            throw GGUFError.invalid("Independent LES reference and callback-free checks have not passed")
        }
        let tokenizer = try CanonicalQwen35Tokenizer(artifact: artifact)
        let prompt = try tokenizer.chat([.init(role: .user, content: Self.prompt)], maximumTokens: 512).tokenIDs
        let nativePlan: LESMemoryPlan?
        if backend == .existingGGUF {
            let samples = reclaimIdleBuffers()
            onAdmission(.init(phase: "benchmark.existingGGUF", beforeReclaim: samples.0,
                afterReclaim: samples.1, plan: nil, failure: nil, nativeMemory: LESContext.runtimeMemory))
            nativePlan = nil
        } else {
            nativePlan = try plan(artifact, context: 4096, depth: backend == .omniLookahead ? 4 : 0,
                pageCache: backend == .omniPageCache, phase: "benchmark.\(backend.rawValue)", onAdmission: onAdmission)
            if let nativePlan, nativePlan.requestedLookaheadDepth > nativePlan.lookaheadDepth {
                progress("\(backend.rawValue) · förläsning \(nativePlan.lookaheadDepth)/\(nativePlan.requestedLookaheadDepth), anpassad efter RAM · top-8 oförändrat")
            }
        }
        var original: GGUFStreamedDecoder?
        var native: LESContext?
        defer { native?.close() }
        if let nativePlan { native = try LESContext(artifact: artifact, prepared: prepared, plan: nativePlan, control: control) }
        else {
            original = try GGUFStreamedDecoder(artifact: artifact, contextTokens: 4096, budgetBytes: 5_500_000_000,
                rawCacheBytes: 64 << 20, residentCacheBytes: 3_500_000_000)
        }
        func step(_ token: Int, projecting: Bool) throws -> [Float]? {
            if control.shouldAbort() { throw CancellationError() }
            if let native { return try native.logits(tokens: [token]) }
            guard let original else { throw GGUFError.invalid("Benchmark owner unavailable") }
            if projecting { return try original.logits(for: UInt64(token), cancellationRequested: { control.shouldAbort() }).asArray(Float.self) }
            try original.consume(UInt64(token), cancellationRequested: { control.shouldAbort() }); return nil
        }
        func reads() -> UInt64 { native?.metrics.ioBytes ?? original?.readStatistics.bytesRead ?? 0 }
        var results: [LESBenchmarkRun] = []
        for warm in [false, true] {
            native?.reset(); original?.reset()
            let thermal = ProcessInfo.processInfo.thermalState.rawValue
            var worstThermal = thermal, peak = ProcessFootprint.current()?.footprintBytes ?? 0
            let before = native?.metrics, startBytes = reads(), started = MonotonicClock.now()
            let originalBefore = original?.projectionStatistics, originalCacheBefore = original?.residentCacheStatistics
            var logits: [Float] = []
            let batchSize = native == nil ? 1 : check.nativePrefillTokens
            for first in stride(from: 0, to: prompt.count, by: batchSize) {
                let end = min(first + batchSize, prompt.count)
                if let native { logits = try native.logits(tokens: Array(prompt[first..<end])) }
                else { logits = try step(prompt[first], projecting: end == prompt.count) ?? [] }
                progress("\(backend.rawValue) · \(warm ? "varm" : "tom appcache") · prompt \(end)/\(prompt.count)")
                peak = max(peak, ProcessFootprint.current()?.footprintBytes ?? 0)
                worstThermal = max(worstThermal, ProcessInfo.processInfo.thermalState.rawValue)
            }
            let prefillSeconds = MonotonicClock.seconds(since: started), prefillBytes = reads(), afterPrefill = native?.metrics
            let originalAfterPrefill = original?.projectionStatistics, originalCacheAfterPrefill = original?.residentCacheStatistics
            var generated: [Int] = [], forwardSeconds = 0.0, forwards = 0, stop = "maximumTokens"
            for index in 0..<128 {
                let token = try greedy(logits)
                if tokenizer.isEndOfGeneration(token) { stop = "endOfGeneration"; break }
                generated.append(token)
                let start = MonotonicClock.now()
                logits = try step(token, projecting: true) ?? []
                forwardSeconds += MonotonicClock.seconds(since: start); forwards += 1
                peak = max(peak, ProcessFootprint.current()?.footprintBytes ?? 0)
                worstThermal = max(worstThermal, ProcessInfo.processInfo.thermalState.rawValue)
                progress("\(backend.rawValue) · \(warm ? "varm" : "tom appcache") · \(index + 1)/128 token")
            }
            var idBytes = Data()
            for id in generated { var little = UInt32(id).littleEndian; withUnsafeBytes(of: &little) { idBytes.append(contentsOf: $0) } }
            results.append(LESBenchmarkRun(backend: backend, applicationCache: warm ? "warm" : "initiallyEmpty",
                promptTokens: prompt.count, generatedTokens: generated.count, decodeForwardPasses: forwards,
                prefillSeconds: prefillSeconds, decodeForwardSeconds: forwardSeconds,
                decodeIOBytes: reads() - prefillBytes, prefillIOBytes: prefillBytes - startBytes,
                outputTokenSHA256: FileDigestComputer.sha256(of: idBytes), maximumObservedFootprintBytes: peak,
                thermalStateAtStart: thermal, maximumThermalState: worstThermal, stopReason: stop,
                memoryPlan: nativePlan, metricsBefore: before, metricsAfterPrefill: afterPrefill, metricsAfter: native?.metrics,
                originalProjectionsBefore: originalBefore, originalProjectionsAfterPrefill: originalAfterPrefill,
                originalProjectionsAfter: original?.projectionStatistics, originalCacheBefore: originalCacheBefore,
                originalCacheAfterPrefill: originalCacheAfterPrefill, originalCacheAfter: original?.residentCacheStatistics,
                prefillBatchTokens: batchSize))
        }
        return results
    }
}
