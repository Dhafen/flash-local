import Foundation
import GGUFCore
import MinirunKit
import ModelAdapters
import StorageCore

/// Explicit, isolated phone experiments. Never changes the chat/API admission,
/// target weights, RoPE, sampler or saved automatic batch configuration.
public enum LESAccelerationBenchmark {
    public enum Candidate: String, Codable, CaseIterable, Sendable { case flashAttention, dflash }
    public static let draftRevision = "3427ea426d150d03c633b0d25b8a0626453b3f2f"
    public static let draftManifest = try! GGUFModelManifest(
        fileName: "Ornith-1.5-35B-A3B-DFlash-BF16.gguf", fileByteCount: 782_819_680,
        sha256: "abb048837823d068b71cea4d162076e5e2be62f504dc79bf08ac80d78c8bb732")
    public static let draftDownload = try! PinnedGGUFDownload(url: URL(string:
        "https://huggingface.co/na0x2c6/Ornith-1.5-35B-A3B-DFlash-GGUF/resolve/" + draftRevision + "/" + draftManifest.fileName)!,
        manifest: draftManifest)
    public static func additionalMemory(context: Int, candidate: Candidate) throws -> UInt64 {
        guard [8192, 16384, 32768, 65536].contains(context), candidate != .dflash || context <= 32768 else {
            throw GGUFError.limit("DFlash-test stöder 8K/16K/32K; 65K är inte minnesvaliderat.")
        }
        guard candidate == .dflash else { return 0 }
        // Actual converted BF16 file, one full + five sliding F16 KV layers,
        // bounded features/logits and draft scheduler reserve. The benchmark
        // reuses the sole existing recurrent checkpoint, never the chat's.
        let slidingCells = ((min(context, 4096 + 4) + 255) / 256) * 256
        return draftManifest.fileByteCount + 4096 * UInt64(context + 5 * slidingCells) + (64 << 20)
    }
    public struct Run: Codable, Sendable, Identifiable {
        public let number: Int
        public var id: Int { number }
        public let mode: String
        public let contextCapacityTokens, actualPromptTokens: Int
        public let generatedTokenIDs: [Int]
        public let prefillSeconds, decodeSeconds, timeToFirstTokenSeconds: Double?
        public let peakFootprintBytes: UInt64
        public let thermalStateAtStart, maximumThermalState: Int
        public let lowPowerMode: Bool
        public let memoryPlan: LESMemoryPlan
        public let observedControl, callbackFreeControl: LESLogitComparison
        public let before, afterPrefill, after: LESMetrics
        public var decodeTokensPerSecond: Double? {
            guard let decodeSeconds, decodeSeconds > 0 else { return nil }
            return Double(generatedTokenIDs.count) / decodeSeconds
        }
        public var prefillTokensPerSecond: Double? {
            guard let prefillSeconds, prefillSeconds > 0 else { return nil }
            return Double(actualPromptTokens) / prefillSeconds
        }
    }
    public struct Report: Codable, Sendable {
        public let schemaVersion = 1
        public let candidate: Candidate
        public let contextCapacityTokens: Int
        public let targetSHA256: String
        public let draftSHA256: String?
        public let nativeRevision = LESContext.revision
        public let logitReference = "same installed GGUF, ordinary native engine; four full-vocabulary positions"
        public let fullModelNumericsVerified = false
        public let fullContextQualityVerified = false
        public let osFileCacheControlled = false
        public let targetIOExcludesDraftPageFaults = true
        public let sampling = "greedy only; draft tokens never sent to chat or API"
        public let runs: [Run]
        public let failure: String?
        public var comparable: Bool {
            guard failure == nil, runs.map(\.mode) == ["baseline", candidate.rawValue, candidate.rawValue, "baseline"],
                  let first = runs.first, first.generatedTokenIDs.count == 128 else { return false }
            return runs.allSatisfy {
                $0.generatedTokenIDs == first.generatedTokenIDs && $0.observedControl.passed && $0.callbackFreeControl.passed && $0.callbackFreeControl.callbackFree &&
                $0.peakFootprintBytes <= $0.memoryPlan.processCeilingBytes && !$0.lowPowerMode &&
                $0.thermalStateAtStart == first.thermalStateAtStart && $0.maximumThermalState == first.thermalStateAtStart &&
                (candidate != .flashAttention || $0.mode == "baseline" || $0.after.flashAttentionActive == true) &&
                (candidate != .dflash || $0.mode == "baseline" ||
                    (($0.after.dflashBlocks ?? 0) > ($0.before.dflashBlocks ?? 0) &&
                     ($0.after.dflashProposals ?? 0) > ($0.before.dflashProposals ?? 0)))
            }
        }
    }
    private struct Probe {
        let rows: [[Float]]
        let forced: [Int]
        let expertSetDigest: UInt64?
        let routingDecisions: UInt64
    }
    private static func greedy(_ values: [Float]) throws -> Int {
        guard !values.isEmpty else { throw GGUFError.invalid("Missing benchmark logits") }
        var best = 0
        for i in values.indices {
            guard values[i].isFinite else { throw GGUFError.invalid("Non-finite benchmark logits") }
            if values[i] > values[best] { best = i }
        }
        return best
    }
    private static func comparison(_ expected: Probe, _ actual: Probe, routing: Bool) throws -> LESLogitComparison {
        guard expected.rows.count == 4, actual.rows.count == 4,
              !routing || (expected.routingDecisions > 0 && expected.routingDecisions == actual.routingDecisions &&
                expected.expertSetDigest != nil && expected.expertSetDigest == actual.expertSetDigest) else {
            throw GGUFError.invalid("Accelerator control changed the actual expert sets")
        }
        var absolute = 0.0, squared = 0.0, reference = 0.0, same = true
        for (a, b) in zip(expected.rows, actual.rows) {
            guard a.count == b.count else { throw GGUFError.invalid("Accelerator vocabulary mismatch") }
            same = try same && greedy(a) == greedy(b)
            for (x, y) in zip(a, b) {
                let delta = Double(x) - Double(y)
                absolute = max(absolute, abs(delta)); squared += delta * delta; reference += Double(x) * Double(x)
            }
        }
        return .init(positions: 4, maximumAbsoluteError: absolute,
            relativeL2Error: sqrt(squared / max(reference, 1e-30)), sameGreedy: same,
            routingObserved: routing, canonicalExpertSetDigest: routing ? actual.expertSetDigest : nil)
    }
    public static func execute(artifact: VerifiedGGUFArtifact, prepared: URL, draft: VerifiedGGUFArtifact?,
                               candidate: Candidate, contextTokens: Int, control: LESControl,
                               progress: @Sendable (String) -> Void,
                               onAdmission: (LESMemoryAdmission) -> Void) throws -> Report {
        _ = try additionalMemory(context: contextTokens, candidate: candidate)
        guard artifact.manifest.sha256 == GGUFModelManifest.primaryOrnith.sha256,
              candidate != .dflash || draft?.manifest == draftManifest else { throw GGUFError.invalid("Unverified accelerator model pair") }
        let tokenizer = try CanonicalQwen35Tokenizer(artifact: artifact)
        let prompt = try tokenizer.chat([.init(role: .user,
            content: "Write a numbered list of one hundred detailed tests for a software cache, concurrency, cancellation and exact state restoration. Explain each test. Continue until the list is complete.")],
            maximumTokens: 512, enableThinking: true).tokenIDs
        var runs: [Run] = [], reference: Probe?
        func report(_ failure: String?) -> Report {
            .init(candidate: candidate, contextCapacityTokens: contextTokens, targetSHA256: artifact.manifest.sha256,
                draftSHA256: candidate == .dflash ? draftManifest.sha256 : nil, runs: runs, failure: failure)
        }
        for (number, name) in ["baseline", candidate.rawValue, candidate.rawValue, "baseline"].enumerated() {
            do {
                control.setNativeDiagnosticsEnabled(true)
                defer { control.setNativeDiagnosticsEnabled(true) }
                try control.recordDiagnosticPhase("run.\(number + 1).\(name).controls")
                control.beginBenchmarkFootprintWindow()
                let result = try autoreleasepool { () throws -> (Run, Probe) in
                    let mode: Candidate? = name == "baseline" ? nil : candidate
                    let extra = try mode.map { try additionalMemory(context: contextTokens, candidate: $0) } ?? 0
                    func open(observed: Bool) throws -> LESContext {
                        let memory = try LESBenchmark.plan(artifact, context: contextTokens, depth: 0, pageCache: true,
                            phase: "accelerator.\(name).\(observed ? "numeric" : "timing")", physicalBatchTokens: 4,
                            experimentReserveBytes: extra, onAdmission: onAdmission)
                        let engine = try LESContext(artifact: artifact, prepared: prepared, plan: memory, control: control,
                            traceRouting: observed, flashAttention: mode == .flashAttention, experimentalDFlash: mode == .dflash)
                        do { if mode == .dflash, let draft { try engine.attachDraft(draft) } }
                        catch { engine.close(); throw error }
                        return engine
                    }
                    func prefill(_ engine: LESContext, stateOnly: Bool = false) throws -> [Float] {
                        var row: [Float] = []
                        for first in stride(from: 0, to: prompt.count, by: 4) {
                            let end = min(first + 4, prompt.count)
                            let tokens = Array(prompt[first..<end])
                            if stateOnly && end < prompt.count { try engine.ingest(tokens: tokens) }
                            else { row = try engine.logits(tokens: tokens) }
                        }
                        return row
                    }
                    func probe(_ engine: LESContext, forced: [Int]?, stateOnly: Bool = false) throws -> Probe {
                        var row = try prefill(engine, stateOnly: stateOnly), rows = [row], tokens: [Int] = []
                        for i in 0..<3 {
                            let id = try forced?[i] ?? greedy(row); tokens.append(id)
                            row = try engine.logits(tokens: [id]); rows.append(row)
                        }
                        return .init(rows: rows, forced: tokens, expertSetDigest: engine.metrics.canonicalExpertSetDigest,
                            routingDecisions: engine.metrics.routingDecisions)
                    }
                    progress("Acceleratortest \(number + 1)/4 · \(name) · kontrollerar logits och top-8…")
                    let observed: Probe = try autoreleasepool {
                        let engine = try open(observed: true); defer { engine.close() }
                        return try probe(engine, forced: reference?.forced)
                    }
                    let expected = reference ?? observed
                    let observedCheck = try comparison(expected, observed, routing: true)
                    guard observedCheck.passed else { throw GGUFError.invalid("Accelerator logit control failed") }
                    let engine = try open(observed: false); defer { engine.close() }
                    // Compare the timed, output-free prompt path with the
                    // ordinary full-output reference, for every candidate.
                    let free = try probe(engine, forced: expected.forced, stateOnly: true)
                    let freeCheck = try comparison(expected, free, routing: false)
                    guard freeCheck.passed else { throw GGUFError.invalid("Callback-free accelerator control failed") }
                    engine.reset()
                    let thermal = ProcessInfo.processInfo.thermalState.rawValue
                    var lowPower = ProcessInfo.processInfo.isLowPowerModeEnabled
                    var maximumThermal = thermal, peak: UInt64 = 0
                    func sample() throws {
                        guard !control.shouldAbort(), let memory = ProcessFootprint.current() else { throw CancellationError() }
                        peak = max(peak, memory.footprintBytes)
                        maximumThermal = max(maximumThermal, ProcessInfo.processInfo.thermalState.rawValue)
                        lowPower = lowPower || ProcessInfo.processInfo.isLowPowerModeEnabled
                    }
                    try sample()
                    // No journal writes inside the timed forward loops. Detailed
                    // native phases were recorded during the preceding probes.
                    control.setNativeDiagnosticsEnabled(false)
                    try control.recordDiagnosticPhase("run.\(number + 1).\(name).timed-prefill")
                    let before = engine.metrics, started = MonotonicClock.now()
                    var logits = try prefill(engine, stateOnly: true)
                    let prefillSeconds = MonotonicClock.seconds(since: started), afterPrefill = engine.metrics
                    try control.recordDiagnosticPhase("run.\(number + 1).\(name).timed-decode")
                    let decodeStart = MonotonicClock.now()
                    var ids: [Int] = [], firstToken: Double?, ended = false
                    while ids.count < 128 && !ended {
                        try sample()
                        let block: [Int]
                        if mode == .dflash { block = try engine.dflashStep(logits: &logits, maximumTokens: min(4, 128 - ids.count)) }
                        else {
                            let id = try greedy(logits)
                            if tokenizer.isEndOfGeneration(id) { break }
                            block = [id]; logits = try engine.logits(tokens: [id])
                        }
                        for id in block {
                            if tokenizer.isEndOfGeneration(id) { ended = true; break }
                            ids.append(id)
                            if firstToken == nil { firstToken = MonotonicClock.seconds(since: started) }
                        }
                        progress("Acceleratortest \(number + 1)/4 · \(name) · \(ids.count)/128 token")
                    }
                    let decodeSeconds = MonotonicClock.seconds(since: decodeStart)
                    try sample()
                    return (.init(number: number, mode: name, contextCapacityTokens: contextTokens, actualPromptTokens: prompt.count,
                        generatedTokenIDs: ids, prefillSeconds: prefillSeconds, decodeSeconds: decodeSeconds,
                        timeToFirstTokenSeconds: firstToken, peakFootprintBytes: max(peak, control.maximumObservedFootprintBytes),
                        thermalStateAtStart: thermal, maximumThermalState: maximumThermal, lowPowerMode: lowPower,
                        memoryPlan: engine.plan, observedControl: observedCheck, callbackFreeControl: freeCheck,
                        before: before, afterPrefill: afterPrefill, after: engine.metrics), observed)
                }
                runs.append(result.0)
                if reference == nil { reference = result.1 }
                guard result.0.generatedTokenIDs.count == 128 else { return report("Färre än 128 token före EOS; inget jämförbart hastighetsresultat.") }
                guard runs.allSatisfy({ $0.generatedTokenIDs == runs[0].generatedTokenIDs }) else {
                    return report("Genererade token avvek från vanliga motorn. Ingen accelerator aktiverades.")
                }
            } catch { return report(error.localizedDescription) }
        }
        return report(nil)
    }
}
