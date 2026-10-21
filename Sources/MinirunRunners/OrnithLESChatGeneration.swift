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
        public let timeToFirstTokenSeconds: Double?
        public let prefixCheckpointBytes: UInt64
        public let prefixReuseSource: String
        public let evaluationBatchTokens, physicalBatchTokens: Int
        public let maximumEvaluationBatchActuallySubmitted: Int
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
    private var prefixTokens: [Int] = []
    private var prefixLogits: [Float] = []
    private var checkpointBytes: UInt64 = 0
    private var prefixRestorationAdmitted = false
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
    static func stableCheckpointPosition(prompt: [Int], generationSuffix: [Int]) -> Int {
        guard !generationSuffix.isEmpty, prompt.count > generationSuffix.count,
              prompt.suffix(generationSuffix.count).elementsEqual(generationSuffix) else { return 0 }
        return prompt.count - generationSuffix.count
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
                prefixCheckpointAdmitted: Bool = false,
                onAdmission: (LESMemoryAdmission) -> Void = { _ in }) throws {
        let batch = try Self.validatedBatch(prefillConfiguration, proof: prefillCheck, baseline: check, modelSHA256: artifact.manifest.sha256,
            nativeRevision: LESContext.revision)
        batchSize = batch.0; physicalBatchSize = batch.1
        self.prefillConfiguration = prefillConfiguration; self.prefillCheck = prefillCheck
        self.artifact = artifact; self.control = control
        prefixRestorationAdmitted = prefixCheckpointAdmitted
        contextTokens = contextProfile.rawValue
        tokenizer = try CanonicalQwen35Tokenizer(artifact: artifact)
        let plan = try LESBenchmark.plan(artifact, context: contextProfile.rawValue, depth: 0, pageCache: true,
            phase: "chat.omniPageCache", physicalBatchTokens: batch.1, onAdmission: onAdmission)
        native = try LESContext(artifact: artifact, prepared: prepared, plan: plan, control: control)
        guard native.vocabularySize == tokenizer.vocabularySize else {
            native.close(); throw GGUFError.invalid("Tokenizer and native vocabulary disagree")
        }
    }
    public func close() { native.close(); clearTokens() }
    private func clearTokens() {
        stateTokens.removeAll(); stateLogits.removeAll(); prefixTokens.removeAll(); prefixLogits.removeAll(); checkpointBytes = 0
    }
    public func resetConversation() { native.reset(); clearTokens() }
    /// Executed on the real installed model/device before the app enables the
    /// new restore path. The app may seal this proof for its exact identity.
    public func validatePrefixCheckpoint() throws {
        guard !running else { throw GGUFError.limit("Prefix check requires an idle owner") }
        defer { resetConversation() }
        let tokens = try tokenizer.chat([.init(role: .user, content: "Compare exact cache states before and after a restored prefix.")], maximumTokens: 64).tokenIDs
        guard tokens.count >= 14 else { throw GGUFError.invalid("Prefix validation fixture is too short") }
        let prefix = Array(tokens.prefix(8)), oldSuffix = Array(tokens[8..<10]), newSuffix = Array(tokens[10..<12])
        _ = try native.logits(tokens: prefix)
        guard native.capturePrefix() > 0 else { throw GGUFError.limit("Recurrent checkpoint could not be captured within its budget") }
        _ = try native.logits(tokens: oldSuffix)
        try native.restorePrefix()
        let restored = try native.logits(tokens: newSuffix)
        native.reset()
        _ = try native.logits(tokens: prefix)
        let fresh = try native.logits(tokens: newSuffix)
        guard restored.count == fresh.count, zip(restored, fresh).allSatisfy({ pair in pair.0.bitPattern == pair.1.bitPattern }) else {
            throw GGUFError.invalid("Restored hybrid KV/SSM logits differ from fresh execution; prefix reuse was not enabled")
        }
        prefixRestorationAdmitted = true
    }
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
        let start = MonotonicClock.now()
        guard !running else { throw GGUFError.limit("Native inference is already running") }
        let maximumNewTokens = options.maximumNewTokens
        running = true
        var completed = false
        defer { running = false; if !completed { resetConversation() } }
        try artifact.assertUnchanged()
        if control.shouldAbort() { throw CancellationError() }
        let rendered = try tokenizer.chat(messages, maximumTokens: contextTokens - maximumNewTokens,
            enableThinking: options.thinkingEnabled, tools: tools)
        let prompt = rendered.tokenIDs
        // The pinned template can strip historical reasoning, and Off adds an
        // empty think block only to the generation suffix. Checkpoint before
        // that suffix, at the special-token boundary, without changing tokens.
        let suffixText = "<|im_start|>assistant\n" + (options.thinkingEnabled ? "<think>\n" : "<think>\n\n</think>\n\n")
        let suffix = try tokenizer.tokenize(suffixText, parseSpecial: true, maximumTokens: 64)
        let checkpointPosition = rendered.renderedText.hasSuffix(suffixText)
            ? Self.stableCheckpointPosition(prompt: prompt, generationSuffix: suffix) : 0
        let thinkingEnd = try tokenizer.tokenize("</think>\n\n", parseSpecial: true, maximumTokens: 8)
        guard !thinkingEnd.isEmpty else { throw GGUFError.invalid("Missing canonical thinking-end tokens") }
        var reused = Self.reusablePrefix(previous: stateTokens, prompt: prompt, hasLogits: !stateLogits.isEmpty)
        var reuseSource = reused > 0 ? "liveExactPrefix" : "none"
        if reused == 0, checkpointBytes > 0,
           Self.reusablePrefix(previous: prefixTokens, prompt: prompt, hasLogits: !prefixLogits.isEmpty) > 0 {
            try native.restorePrefix()
            stateTokens = prefixTokens; stateLogits = prefixLogits
            reused = prefixTokens.count; reuseSource = "recurrentCheckpointAndRetainedAttentionKV"
        }
        if reused == 0 { resetConversation() }
        let before = native.metrics
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
        var largestEvaluation = 0
        func captureStableBoundary() {
            checkpointBytes = prefixRestorationAdmitted ? native.capturePrefix() : 0
            prefixTokens = checkpointBytes > 0 ? stateTokens : []
            prefixLogits = checkpointBytes > 0 ? stateLogits : []
        }
        if checkpointPosition > 0, checkpointPosition == reused { captureStableBoundary() }
        var first = reused
        while first < prompt.count {
            try sample()
            var end = min(first + batchSize, prompt.count)
            if first < checkpointPosition { end = min(end, checkpointPosition) }
            let tokens = Array(prompt[first..<end])
            largestEvaluation = max(largestEvaluation, tokens.count)
            stateLogits = try autoreleasepool { try native.logits(tokens: tokens) }
            stateTokens.append(contentsOf: tokens)
            if end == checkpointPosition { captureStableBoundary() }
            try onEvent(.prefill(progress(end)))
            first = end
        }
        // Chat templates trim/normalize assistant reasoning and tool envelopes.
        // Keep the exact pre-generation boundary so that a later canonical
        // rendering can reuse all of its unchanged prefix without replaying it.
        // The snapshot contains recurrent state only (<=128 MiB), not full KV.
        // Keep the stable snapshot above: taking another at prompt.end would
        // reintroduce the unstable generation suffix and destroy Off reuse.
        let prefillSeconds = MonotonicClock.seconds(since: start), afterPrefill = native.metrics
        var decodeSeconds = 0.0, passes = 0, reason = "tokenLimit"
        var firstTokenSeconds: Double?
        func append(_ id: Int, forced: Bool) throws {
            let bytes = try tokenizer.bytes(for: id, includeControl: true)
            outputBytes += bytes.count
            guard outputBytes <= 4 << 20 else { throw GGUFError.limit("Generated text exceeds 4 MiB") }
            if forced { forcedTokens += 1 } else if thinking.isThinking { reasoningTokens += 1 }
            generated.append(id)
            if firstTokenSeconds == nil { firstTokenSeconds = MonotonicClock.seconds(since: start) }
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
            timeToFirstTokenSeconds: firstTokenSeconds, prefixCheckpointBytes: checkpointBytes, prefixReuseSource: reuseSource,
            evaluationBatchTokens: batchSize, physicalBatchTokens: physicalBatchSize,
            maximumEvaluationBatchActuallySubmitted: largestEvaluation,
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
