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
    public var activeLookaheadDepth: Int? = nil
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
    public let fullModelNumericsVerified = false
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
    }
    private static func greedy(_ values: [Float]) throws -> Int {
        guard !values.isEmpty, values.allSatisfy(\.isFinite) else { throw GGUFError.invalid("Non-finite benchmark logits") }
        return values.indices.max { values[$0] < values[$1] }!
    }
    private static func reclaimIdleBuffers() -> (ProcessFootprint?, ProcessFootprint?) {
        let before = ProcessFootprint.current()
        // Called only after the preceding inference owner and its autorelease
        // pool have ended, never between a cold/warm pair or during a forward.
        MLX.Memory.clearCache()
        _ = malloc_zone_pressure_relief(nil, 0)
        return (before, ProcessFootprint.current())
    }
    static func plan(_ artifact: VerifiedGGUFArtifact, context: Int, depth: Int,
                             pageCache: Bool = false, phase: String,
                             onAdmission: (LESMemoryAdmission) -> Void) throws -> LESMemoryPlan {
        let samples = reclaimIdleBuffers()
        let nativeMemory = LESContext.runtimeMemory
        do {
            guard nativeMemory.liveContexts == 0, nativeMemory.liveMetalBuffers == 0 else {
                throw GGUFError.limit("Previous LES owner has not released its native buffers")
            }
            guard let footprint = samples.1 else { throw GGUFError.limit("LES memory sample unavailable") }
            let result = try LESMemoryPlan(index: artifact.index, contextTokens: context, footprint: footprint,
                lookaheadDepth: depth, pageCacheOnly: pageCache)
            onAdmission(.init(phase: phase, beforeReclaim: samples.0, afterReclaim: samples.1, plan: result, failure: nil,
                nativeMemory: nativeMemory))
            return result
        } catch {
            onAdmission(.init(phase: phase, beforeReclaim: samples.0, afterReclaim: samples.1, plan: nil,
                failure: error.localizedDescription, nativeMemory: nativeMemory))
            throw GGUFError.limit("\(phase): \(error.localizedDescription)")
        }
    }
    private static func reference(artifact: VerifiedGGUFArtifact, prompt: [Int], control: LESControl) throws -> Replay {
        let context = try Qwen4ExpCPUReference(artifact: artifact, contextTokens: 128, budgetBytes: 5_500_000_000,
            cancellationRequested: { control.shouldAbort() })
        var logits: [Float] = []
        for token in prompt { logits = try context.logits(for: token) }
        var vectors = [logits], forced: [Int] = []
        for _ in 0..<3 {
            let token = try greedy(logits); forced.append(token)
            logits = try context.logits(for: token); vectors.append(logits)
        }
        return Replay(prompt: prompt, forced: forced, logits: vectors)
    }
    private static func compare(_ replay: Replay, artifact: VerifiedGGUFArtifact, prepared: URL,
                                depth: Int, prefillBatch: Int = 1, control: LESControl,
                                onAdmission: (LESMemoryAdmission) -> Void) throws -> LESLogitComparison {
        let memory = try plan(artifact, context: 128, depth: depth,
            phase: "numerics.prediction\(depth).prefill\(prefillBatch)", onAdmission: onAdmission)
        let context = try LESContext(artifact: artifact, prepared: prepared, plan: memory, control: control, traceRouting: true)
        defer { context.close() }
        var logits: [Float] = []
        for first in stride(from: 0, to: replay.prompt.count, by: prefillBatch) {
            logits = try context.logits(tokens: Array(replay.prompt[first..<min(first + prefillBatch, replay.prompt.count)]))
        }
        var absolute = 0.0, squared = 0.0, referenceSquared = 0.0, same = true
        for position in 0..<4 {
            if position > 0 { logits = try context.logits(tokens: [replay.forced[position - 1]]) }
            let expected = replay.logits[position]
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
            routingDigest: context.metrics.routingDigest, routingDecisions: context.metrics.routingDecisions,
            canonicalRoutingDigest: context.metrics.canonicalRoutingDigest,
            activeLookaheadDepth: memory.lookaheadDepth)
    }
    public static func check(artifact: VerifiedGGUFArtifact, prepared: URL, control: LESControl,
                             progress: @Sendable (String) -> Void = { _ in },
                             onAdmission: (LESMemoryAdmission) -> Void = { _ in }) throws -> LESNumericalCheck {
        let tokenizer = try CanonicalQwen35Tokenizer(artifact: artifact)
        let tokens = try tokenizer.chat([.init(role: .user, content: "Vad är 50 + 5?")], maximumTokens: 64).tokenIDs
        progress("Jämför hela modellgrafen med CPU-referensen…")
        let replay = try autoreleasepool { try reference(artifact: artifact, prompt: tokens, control: control) }
        progress("Kontrollerar LES utan förläsning…")
        let off = try autoreleasepool { try compare(replay, artifact: artifact, prepared: prepared, depth: 0,
            control: control, onAdmission: onAdmission) }
        progress("Kontrollerar LES med förläsning…")
        let on = try autoreleasepool { try compare(replay, artifact: artifact, prepared: prepared, depth: 4,
            control: control, onAdmission: onAdmission) }
        progress("Kontrollerar batchad promptbearbetning…")
        let batch = try autoreleasepool { try compare(replay, artifact: artifact, prepared: prepared, depth: 0,
            prefillBatch: 4, control: control, onAdmission: onAdmission) }
        return LESNumericalCheck(modelSHA256: artifact.manifest.sha256,
            referenceRevision: Qwen4ExpCPUReference.sourceRevision, nativeRevision: LESContext.revision,
            noLookahead: off, lookahead: on, batchedPrefill: batch)
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
              backend == .existingGGUF || (check.passed && check.canonicalBatchedRoutingVerified) else {
            throw GGUFError.invalid("LES numerical and batched routing checks have not passed")
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
            let batchSize = native == nil ? 1 : 4
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
                originalCacheAfterPrefill: originalCacheAfterPrefill, originalCacheAfter: original?.residentCacheStatistics))
        }
        return results
    }
}
