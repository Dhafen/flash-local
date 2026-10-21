#if canImport(OrnithLES)
import Foundation
import GGUFCore
import MinirunKit
import ModelAdapters
import OrnithLES
import StorageCore

/// Experimental page-cache chat, isolated from the ordinary GGUF/REAM engine.
/// The same native graph, weights, routing and cache mode ran the phone benchmark.
/// Requires an admitted reference check. The app can restore only its own
/// device-key-authenticated receipt after model/prepared integrity validation;
/// imported reports never enable this runner.
public final class OrnithLESChatGeneration {
    public struct Summary: Codable, Sendable {
        public let backend = "omniPageCache"
        public let fullModelNumericsVerified = false
        public let promptTokenIDs, generatedTokenIDs: [Int]
        public let reusedPromptTokens, evaluatedPromptTokens, decodeForwardPasses: Int
        public let prefillSeconds, decodeForwardSeconds, elapsedSeconds: Double
        public let evaluationBatchTokens, physicalBatchTokens: Int
        public let prefillConfiguration: LESPrefillConfiguration
        public let prefillCheck: LESPrefillCheck?
        public let prefillIOBytes, decodeIOBytes, maximumObservedFootprintBytes: UInt64
        public let sessionMaximumObservedFootprintBytes: UInt64
        public let thermalStateAtStart, maximumThermalState, prefillBatchTokens: Int
        public let lowPowerModeAtStart: Bool
        public let memoryPlan: LESMemoryPlan
        public let stopReason: String
        public let options: LocalGenerationOptions
        public let reasoningTokens, forcedThinkingEndTokens: Int
        public let metricsBefore, metricsAfterPrefill, metricsAfter: LESMetrics
        public var decodeTokensPerSecond: Double? {
            decodeForwardPasses > 0 && decodeForwardSeconds > 0 ? Double(decodeForwardPasses) / decodeForwardSeconds : nil
        }
    }
    private let artifact: VerifiedGGUFArtifact
    private let tokenizer: CanonicalQwen35Tokenizer
    private let control: LESControl
    private let native: LESContext
    private let batchSize: Int
    private let physicalBatchSize: Int
    private let prefillConfiguration: LESPrefillConfiguration
    private let prefillCheck: LESPrefillCheck?
    private var stateTokens: [Int] = []
    private var stateLogits: [Float] = []
    private var running = false
    public let contextTokens: Int

    static func validatedPrefill(_ check: LESNumericalCheck, modelSHA256: String, nativeRevision: String) throws -> Int {
        guard check.modelSHA256 == modelSHA256, check.nativeRevision == nativeRevision,
              check.nativeBenchmarkAdmitted, check.nativePrefillTokens == 1 || check.nativePrefillTokens == 4 else {
            throw GGUFError.invalid("Snabbchatten kräver godkänd kontroll för samma modell och motor.")
        }
        return check.nativePrefillTokens
    }
    static func reusablePrefix(previous: [Int], prompt: [Int], hasLogits: Bool) -> Int {
        !previous.isEmpty && prompt.starts(with: previous) && (previous.count < prompt.count || hasLogits)
            ? previous.count : 0
    }
    static func validatedBatch(_ configuration: LESPrefillConfiguration, proof: LESPrefillCheck?,
                               baseline: LESNumericalCheck, modelSHA256: String, nativeRevision: String) throws -> (Int, Int) {
        let legacy = try validatedPrefill(baseline, modelSHA256: modelSHA256, nativeRevision: nativeRevision)
        try configuration.validate()
        if configuration == LESPrefillConfiguration() { return (legacy, 4) }
        let physical = try configuration.physicalTokens(checkedDefault: legacy)
        var effective = configuration; effective.physicalTokens = physical
        guard let proof, proof.passed, proof.configuration == effective,
              proof.modelSHA256 == modelSHA256, proof.nativeRevision == nativeRevision else {
            throw GGUFError.invalid("Den valda batchen har inte klarat sin egen kontroll mot originalvikterna.")
        }
        return (configuration.evaluationTokens, physical)
    }
    public init(artifact: VerifiedGGUFArtifact, prepared: URL, check: LESNumericalCheck, control: LESControl,
                prefillConfiguration: LESPrefillConfiguration = .init(), prefillCheck: LESPrefillCheck? = nil,
                contextProfile: LocalContextProfile = .standard,
                onAdmission: (LESMemoryAdmission) -> Void = { _ in }) throws {
        let batch = try Self.validatedBatch(prefillConfiguration, proof: prefillCheck, baseline: check, modelSHA256: artifact.manifest.sha256,
            nativeRevision: LESContext.revision)
        batchSize = batch.0; physicalBatchSize = batch.1
        self.prefillConfiguration = prefillConfiguration; self.prefillCheck = prefillCheck
        self.artifact = artifact; self.control = control
        contextTokens = contextProfile.rawValue
        tokenizer = try CanonicalQwen35Tokenizer(artifact: artifact)
        let plan = try LESBenchmark.plan(artifact, context: contextProfile.rawValue, depth: 0, pageCache: true,
            phase: "chat.omniPageCache", physicalBatchTokens: batch.1, onAdmission: onAdmission)
        native = try LESContext(artifact: artifact, prepared: prepared, plan: plan, control: control)
        guard native.vocabularySize == tokenizer.vocabularySize else {
            native.close(); throw GGUFError.invalid("Tokenizer and native vocabulary disagree")
        }
    }
    public func close() { native.close(); stateTokens.removeAll(); stateLogits.removeAll() }
    public func resetConversation() { native.reset(); stateTokens.removeAll(); stateLogits.removeAll() }
    deinit { close() }

    /// Exact token-prefix reuse only. A changed template/history resets SSM/KV;
    /// no approximate prefix, hidden truncation or model-sized state snapshot.
    public func generate(messages: [Qwen35ChatMessage], maximumNewTokens: Int = 512, enableThinking: Bool,
                         onEvent: (Qwen4ExpTextGeneration.Event) throws -> Void) throws -> Summary {
        var options = LocalGenerationOptions()
        options.maximumNewTokens = maximumNewTokens; options.enableThinking = enableThinking
        return try generate(messages: messages, options: options, onEvent: onEvent)
    }
    public func generate(messages: [Qwen35ChatMessage], options: LocalGenerationOptions,
                         tools: [LocalToolDefinition] = [],
                         onEvent: (Qwen4ExpTextGeneration.Event) throws -> Void) throws -> Summary {
        try options.validate()
        guard !running else { throw GGUFError.limit("Native inference is already running") }
        let maximumNewTokens = options.maximumNewTokens
        running = true
        var completed = false
        defer { running = false; if !completed { resetConversation() } }
        try artifact.assertUnchanged()
        if control.shouldAbort() { throw CancellationError() }
        let prompt = try tokenizer.chat(messages, maximumTokens: contextTokens - maximumNewTokens,
            enableThinking: options.thinkingEnabled, tools: tools).tokenIDs
        let thinkingEnd = try tokenizer.tokenize("</think>\n\n", parseSpecial: true, maximumTokens: 8)
        guard !thinkingEnd.isEmpty else { throw GGUFError.invalid("Missing canonical thinking-end tokens") }
        let reused = Self.reusablePrefix(previous: stateTokens, prompt: prompt, hasLogits: !stateLogits.isEmpty)
        if reused == 0 { resetConversation() }
        let start = MonotonicClock.now(), before = native.metrics
        var peak = ProcessFootprint.current()?.footprintBytes ?? 0
        let thermal = ProcessInfo.processInfo.thermalState.rawValue
        let lowPower = ProcessInfo.processInfo.isLowPowerModeEnabled
        var maximumThermal = thermal, generated: [Int] = [], text = Qwen4ExpTokenText(), outputBytes = 0
        var thinking = QwenThinkingText(startsInThinking: options.thinkingEnabled)
        var sampler = LocalTokenSampler(seed: options.seed), reasoningTokens = 0, forcedTokens = 0
        func sample() throws {
            if control.shouldAbort() { throw CancellationError() }
            try artifact.assertUnchanged()
            guard let value = ProcessFootprint.current() else { throw GGUFError.limit("Memory sample unavailable") }
            peak = max(peak, value.footprintBytes)
            maximumThermal = max(maximumThermal, ProcessInfo.processInfo.thermalState.rawValue)
        }
        func progress(_ count: Int) -> Qwen4ExpTextGeneration.Progress {
            .init(promptTokensCompleted: count, promptTokensTotal: prompt.count, generatedTokens: generated.count,
                elapsedSeconds: MonotonicClock.seconds(since: start), ioBytesRead: native.metrics.ioBytes - before.ioBytes,
                footprintBytes: ProcessFootprint.current()?.footprintBytes ?? peak, reusedPromptTokens: reused)
        }
        if reused > 0 { try onEvent(.prefill(progress(reused))) }
        for first in stride(from: reused, to: prompt.count, by: batchSize) {
            try sample()
            let end = min(first + batchSize, prompt.count), tokens = Array(prompt[first..<end])
            stateLogits = try autoreleasepool { try native.logits(tokens: tokens) }
            stateTokens.append(contentsOf: tokens)
            try onEvent(.prefill(progress(end)))
        }
        let prefillSeconds = MonotonicClock.seconds(since: start), afterPrefill = native.metrics
        var decodeSeconds = 0.0, passes = 0, reason = "tokenLimit"
        func append(_ id: Int, forced: Bool) throws {
            let bytes = try tokenizer.bytes(for: id, includeControl: true)
            outputBytes += bytes.count
            guard outputBytes <= 4 << 20 else { throw GGUFError.limit("Generated text exceeds 4 MiB") }
            if forced { forcedTokens += 1 } else if thinking.isThinking { reasoningTokens += 1 }
            generated.append(id)
            let delta = text.append(bytes); thinking.append(delta)
            try onEvent(.token(id: id, textDelta: delta, progress: progress(prompt.count)))
            if generated.count < maximumNewTokens {
                let begin = MonotonicClock.now()
                stateLogits = try autoreleasepool { try native.logits(tokens: [id]) }
                stateTokens.append(id)
                decodeSeconds += MonotonicClock.seconds(since: begin); passes += 1
            }
        }
        while generated.count < maximumNewTokens {
            try sample()
            // An explicit user budget conditions the original model on its
            // canonical closing marker. Every inserted token is evaluated and
            // charged to the total limit; no graph/routing result is fabricated.
            if options.thinkingEnabled, thinking.isThinking, let budget = options.thinkingBudget, reasoningTokens >= budget {
                guard generated.count + thinkingEnd.count < maximumNewTokens else { break }
                for token in thinkingEnd { try sample(); try append(token, forced: true) }
                guard !thinking.isThinking else { throw GGUFError.invalid("Thinking budget did not close its canonical channel") }
                continue
            }
            let id = try sampler.select(stateLogits, options: options)
            if tokenizer.isEndOfGeneration(id) { reason = "endOfGeneration"; break }
            try append(id, forced: false)
        }
        try sample()
        let trailing = text.finish()
        if !trailing.isEmpty { try onEvent(.trailingText(trailing)) }
        let after = native.metrics
        guard after.ioBytes >= afterPrefill.ioBytes, afterPrefill.ioBytes >= before.ioBytes else {
            throw GGUFError.invalid("Native I/O accounting regressed")
        }
        completed = true
        return Summary(promptTokenIDs: prompt, generatedTokenIDs: generated, reusedPromptTokens: reused,
            evaluatedPromptTokens: prompt.count - reused, decodeForwardPasses: passes,
            prefillSeconds: prefillSeconds, decodeForwardSeconds: decodeSeconds, elapsedSeconds: MonotonicClock.seconds(since: start),
            evaluationBatchTokens: batchSize, physicalBatchTokens: physicalBatchSize,
            prefillConfiguration: prefillConfiguration, prefillCheck: prefillCheck,
            prefillIOBytes: afterPrefill.ioBytes - before.ioBytes, decodeIOBytes: after.ioBytes - afterPrefill.ioBytes,
            maximumObservedFootprintBytes: peak, sessionMaximumObservedFootprintBytes: control.maximumObservedFootprintBytes,
            thermalStateAtStart: thermal, maximumThermalState: maximumThermal, prefillBatchTokens: min(batchSize, physicalBatchSize),
            lowPowerModeAtStart: lowPower, memoryPlan: native.plan,
            stopReason: reason, options: options, reasoningTokens: reasoningTokens, forcedThinkingEndTokens: forcedTokens,
            metricsBefore: before, metricsAfterPrefill: afterPrefill, metricsAfter: after)
    }
}
#endif
