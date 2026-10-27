import Darwin
import Foundation
import FlashFlint
import GGUFCore
import MinirunKit
import StorageCore

public enum FlintFailure: Error, LocalizedError, Sendable {
    case invalid(String)
    public var errorDescription: String? { switch self { case .invalid(let message): return message } }
}

public struct FlintConfiguration: Equatable, Sendable {
    public enum Backend: Equatable, Sendable { case metal, cpuReference }
    public var backend: Backend = .metal
    public var contextTokens = 8192
    public var evaluationBatch = 128
    public var physicalBatch = 4
    public var cpuThreads = 4
    public var diagnosticRouting = false
    public var memoryCeilingBytes: UInt64 = 5_500_000_000
    public init() {}
    public func validate() throws {
        guard (32...65536).contains(contextTokens), (1...2048).contains(evaluationBatch),
              (1...16).contains(physicalBatch), physicalBatch <= evaluationBatch,
              (1...8).contains(cpuThreads), memoryCeilingBytes > 0, memoryCeilingBytes <= 5_500_000_000 else {
            throw FlintFailure.invalid("Invalid bounded Flint configuration")
        }
    }
}

public enum FlintArtifact {
    public static let manifest = try! GGUFModelManifest(fileName: "flint-int4-q4_1.gguf",
        fileByteCount: 25_992_966_976,
        sha256: "43c07baccdba42eb7528a9630ee15699b1942c297291dc1b39a061f508ee1a4b")
    public static let sourceURL = URL(string: "https://huggingface.co/RecursiveSI/flint-int4-gguf/resolve/9c47151036bc2c34331142dda04488bceab2bc9f/flint-int4-q4_1.gguf")!
}

public struct FlintProgress: Sendable {
    public let phase: Int
    /// Nil means a real phase with no measurable percentage.
    public let fraction: Double?
    public let observedFootprintBytes: UInt64?
}

public struct FlintMetrics: Codable, Sendable {
    /// Counts explicit expert reads, not mmap faults or physical NAND traffic.
    public let expertIOBytes, expertIOCalls, cachedTokens, evaluatedTokens, stateOnlyTokens: UInt64
    public let estimatedProcessBytes, expertBankBytes, estimatedKVBytes: UInt64
    public let forwardSeconds: Double
    public let flashAttentionActive: Bool
    init(_ value: flf_metrics) {
        expertIOBytes = value.io_bytes; expertIOCalls = value.io_calls; cachedTokens = value.cached_tokens
        evaluatedTokens = value.evaluated_tokens; stateOnlyTokens = value.state_only_tokens
        estimatedProcessBytes = value.estimated_process_bytes; expertBankBytes = value.expert_bank_bytes
        estimatedKVBytes = value.estimated_kv_bytes; forwardSeconds = value.forward_seconds
        flashAttentionActive = value.flash_attention_active
    }
}

// Safety invariant: the only cross-executor mutable fields are protected by
// this lock. onProgress is installed before the synchronous native load and
// cleared only after its loader workers return; cancellation never accesses it.
// No native pointer crosses this bridge. Replace the lock with checked atomic
// storage when the project's minimum Swift toolchain supports that migration.
private final class FlintCancellation: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false
    private var ceiling: UInt64 = 0
    private var peak: UInt64 = 0
    private var maximumThermal = 0
    var onProgress: (@Sendable (FlintProgress) -> Void)?
    func begin(ceiling: UInt64) {
        lock.lock(); cancelled = false; self.ceiling = ceiling; peak = 0
        maximumThermal = ProcessInfo.processInfo.thermalState.rawValue; lock.unlock()
    }
    func cancel() { lock.lock(); cancelled = true; lock.unlock() }
    func observation() -> (peak: UInt64, thermal: Int) {
        lock.lock(); defer { lock.unlock() }; return (peak, maximumThermal)
    }
    func shouldAbort() -> Bool {
        lock.lock(); let stopped = cancelled, limit = ceiling; lock.unlock()
        guard let memory = ProcessFootprint.current() else { return true }
        lock.lock(); peak = max(peak, memory.footprintBytes)
        maximumThermal = max(maximumThermal, ProcessInfo.processInfo.thermalState.rawValue); lock.unlock()
        return stopped || memory.footprintBytes >= limit - min(limit, 128 << 20) ||
            memory.availableBytes.map({ $0 < 128 << 20 }) == true
    }
}

private final class FlintNativeOwner {
    let pointer: OpaquePointer
    let directory: URL
    // The C callback borrows this object until flf_close completes. Retain it
    // here as well as on the actor, independent of stored-property teardown.
    let control: FlintCancellation
    let engineLease: LocalEngineLease
    init(pointer: OpaquePointer, directory: URL, control: FlintCancellation, engineLease: LocalEngineLease) {
        self.pointer = pointer; self.directory = directory; self.control = control; self.engineLease = engineLease
    }
    deinit { flf_close(pointer); _ = Darwin.rmdir(directory.path) }
}

/// Experimental inference executor, independent of SwiftUI/view lifetime.
/// This is a control/adapter API, not an enabled production chat backend.
/// The app must close its Ornith owner before load (native omni knobs are still
/// process-wide). Do not advertise model quality from successful file loading.
public actor FlintSession {
    private nonisolated let cancellation = FlintCancellation()
    private var native: FlintNativeOwner?
    private var artifact: VerifiedGGUFArtifact?
    private var configuration = FlintConfiguration()
    private var activeCeiling: UInt64 = 5_500_000_000
    private var stateTokens: [Int32] = []
    private var lastLogits: [Float]?
    private var hasCurrentLogits = false
    private var activeRequest: UUID?
    private var generationReport: FlintGenerationReport?
    private var numericalAdmission = false
    private var numericalReport: FlintNumericalCheck?

    public init() {}
    public nonisolated func cancel() { cancellation.cancel() }
    private func failure() -> FlintFailure { .invalid(String(cString: flf_last_error())) }
    private func owner() throws -> FlintNativeOwner {
        guard let native else { throw FlintFailure.invalid("Load Flint first") }
        return native
    }

    public func load(artifact: VerifiedGGUFArtifact, privateRoot: URL,
                     configuration: FlintConfiguration = .init(),
                     progress: @escaping @Sendable (FlintProgress) -> Void = { _ in }) throws {
        try Task.checkCancellation()
        try configuration.validate()
        try artifact.assertUnchanged()
        guard native == nil else {
            if self.artifact?.manifest == artifact.manifest, self.configuration == configuration {
                try self.artifact?.assertUnchanged()
                return
            }
            throw FlintFailure.invalid("Unload the existing Flint session before changing its model/configuration")
        }
        guard artifact.manifest == FlintArtifact.manifest,
              artifact.index.metadata["general.architecture"]?.text == "flint", privateRoot.isFileURL,
              let memory = ProcessFootprint.current() else {
            throw FlintFailure.invalid("Expected the verified official Flint INT4 GGUF and a private session directory")
        }
        let limit = min(configuration.memoryCeilingBytes, memory.availableBytes.map { memory.footprintBytes + $0 } ?? configuration.memoryCeilingBytes)
        cancellation.begin(ceiling: limit)
        cancellation.onProgress = progress
        defer { cancellation.onProgress = nil }
        let fm = FileManager.default
        try fm.createDirectory(at: privateRoot, withIntermediateDirectories: true)
        let root = privateRoot.resolvingSymlinksInPath().standardizedFileURL
        let directory = root.appendingPathComponent("flint-session-" + UUID().uuidString, isDirectory: true)
        try fm.createDirectory(at: directory, withIntermediateDirectories: false)
        var directoryAdopted = false
        defer { if !directoryAdopted { _ = Darwin.rmdir(directory.path) } }
        let fd = try artifact.duplicateForReading(); defer { Darwin.close(fd) }
        let engineLease = try LocalEngineLease.acquire(.flint)
        var options = flf_options()
        options.context_tokens = Int32(configuration.contextTokens)
        options.evaluation_batch = Int32(configuration.evaluationBatch)
        options.physical_batch = Int32(configuration.physicalBatch)
        options.cpu_threads = Int32(configuration.cpuThreads)
        options.process_ceiling_bytes = limit
        options.app_reserve_bytes = max(memory.footprintBytes, 400_000_000)
        options.metal = configuration.backend == .metal
        options.f32_kv_control = configuration.backend == .cpuReference
        options.diagnostic_routing = configuration.diagnosticRouting
        let control = Unmanaged.passUnretained(cancellation).toOpaque()
        let pointer = flf_open(fd, directory.path, options, { opaque in
            guard let opaque else { return true }
            return Unmanaged<FlintCancellation>.fromOpaque(opaque).takeUnretainedValue().shouldAbort()
        }, { phase, fraction, opaque in
            guard let opaque else { return }
            let control = Unmanaged<FlintCancellation>.fromOpaque(opaque).takeUnretainedValue()
            control.onProgress?(.init(phase: Int(phase), fraction: fraction >= 0 ? Double(fraction) : nil,
                observedFootprintBytes: ProcessFootprint.current()?.footprintBytes))
        }, control)
        guard let pointer else { _ = Darwin.rmdir(directory.path); throw failure() }
        let loaded = FlintNativeOwner(pointer: pointer, directory: directory, control: cancellation, engineLease: engineLease)
        try artifact.assertUnchanged()
        self.native = loaded; self.artifact = artifact; self.configuration = configuration
        activeCeiling = limit; directoryAdopted = true; numericalAdmission = false
        stateTokens.removeAll(); lastLogits = nil; hasCurrentLogits = false
    }

    public func unload() {
        cancellation.cancel(); native = nil; artifact = nil; stateTokens.removeAll(); lastLogits = nil; hasCurrentLogits = false; numericalAdmission = false
    }
    public func resetConversation() throws {
        guard activeRequest == nil else { throw LocalAPIError(status: 429, message: "Stop the current Flint request before clearing its context") }
        try resetKV()
        cancellation.begin(ceiling: activeCeiling)
    }
    private func resetKV() throws {
        let owner = try self.owner()
        guard flf_reset(owner.pointer) == 0 else { throw failure() }
        stateTokens.removeAll(); lastLogits = nil; hasCurrentLogits = false
    }
    public func metrics() throws -> FlintMetrics { .init(flf_snapshot(try owner().pointer)) }
    public func lastReport() -> FlintGenerationReport? { generationReport }
    public func lastNumericalCheck() -> FlintNumericalCheck? { numericalReport }
    public static var nativeRevision: String { String(cString: flf_revision()) }
    public func traceRouting(_ enabled: Bool) throws {
        guard activeRequest == nil else { throw LocalAPIError(status: 429, message: "Finish the active request before running diagnostics") }
        guard flf_trace_routes(try owner().pointer, enabled) == 0 else { throw failure() }
    }
    public func routingDecisions() throws -> [FlintRouteDecision] {
        let pointer = try owner().pointer, size = flf_routing_json(pointer, nil, 0)
        guard size < 0, size >= -(4 << 20) else { throw failure() }
        var bytes = [CChar](repeating: 0, count: Int(-size))
        guard flf_routing_json(pointer, &bytes, Int32(bytes.count)) == -size else { throw failure() }
        return try JSONDecoder().decode([FlintRouteDecision].self, from: Data(bytes.map { UInt8(bitPattern: $0) }))
    }

    public func renderPrompt(requestJSON: Data) throws -> String {
        guard !requestJSON.isEmpty, requestJSON.count <= 2 << 20 else { throw FlintFailure.invalid("Flint request exceeds 2 MiB") }
        return try requestJSON.withUnsafeBytes { raw in
            let input = raw.bindMemory(to: CChar.self).baseAddress!
            let size = flf_render_prompt(input, Int32(raw.count), nil, 0)
            guard size < 0 else { throw failure() }
            var output = [CChar](repeating: 0, count: Int(-size))
            guard flf_render_prompt(input, Int32(raw.count), &output, Int32(output.count)) == -size else { throw failure() }
            return String(decoding: output.map { UInt8(bitPattern: $0) }, as: UTF8.self)
        }
    }
    public func tokenize(_ prompt: String) throws -> [Int32] {
        let owner = try self.owner(), utf8 = Array(prompt.utf8CString)
        guard utf8.count <= (2 << 20) + 1 else { throw FlintFailure.invalid("Flint prompt exceeds 2 MiB") }
        return try utf8.withUnsafeBufferPointer { buffer in
            let size = flf_tokenize(owner.pointer, buffer.baseAddress!, Int32(buffer.count - 1), nil, 0)
            if size == 0, prompt.isEmpty { return [] }
            guard size < 0 else { throw failure() }
            var output = [Int32](repeating: 0, count: Int(-size))
            guard flf_tokenize(owner.pointer, buffer.baseAddress!, Int32(buffer.count - 1), &output, Int32(output.count)) == -size else { throw failure() }
            return output
        }
    }
    /// Append/evaluate exact IDs, keeping one bounded KV session. No rollback.
    public func evaluate(tokens: [Int32], outputLogits: Bool) throws -> [Float]? {
        try append(tokens: tokens, finalRow: outputLogits, copyLogits: outputLogits)
    }
    private func append(tokens: [Int32], finalRow: Bool, copyLogits: Bool) throws -> [Float]? {
        let owner = try self.owner()
        guard !tokens.isEmpty, tokens.count <= configuration.evaluationBatch else { throw FlintFailure.invalid("Invalid Flint evaluation batch") }
        try artifact?.assertUnchanged()
        if Task.isCancelled { cancellation.cancel() }
        var output = copyLogits ? [Float](repeating: 0, count: Int(flf_vocab_size(owner.pointer))) : []
        let status = tokens.withUnsafeBufferPointer { ids in
            if finalRow && !copyLogits { return flf_evaluate_for_sampling(owner.pointer, ids.baseAddress, Int32(ids.count)) }
            return output.withUnsafeMutableBufferPointer { logits in
                flf_evaluate(owner.pointer, ids.baseAddress, Int32(ids.count), copyLogits ? logits.baseAddress : nil, Int32(logits.count))
            }
        }
        guard status == 0 else {
            let error = failure()
            _ = flf_reset(owner.pointer)
            stateTokens.removeAll(); lastLogits = nil; hasCurrentLogits = false
            throw error
        }
        try artifact?.assertUnchanged()
        stateTokens.append(contentsOf: tokens); lastLogits = copyLogits ? output : nil; hasCurrentLogits = finalRow
        return lastLogits
    }
    /// Reuses only a complete, unchanged cached prefix; shortened/changed
    /// templates reset instead of attempting an invalid SWA-cache rewind.
    public func preparePrompt(tokens: [Int32], progress: @Sendable (Int, Int) -> Void = { _, _ in }) throws -> (logits: [Float], reusedTokens: Int) {
        guard activeRequest == nil else { throw LocalAPIError(status: 429, message: "Flint is already generating") }
        cancellation.begin(ceiling: activeCeiling)
        let reused = try prepare(tokens: tokens, progress: progress)
        if lastLogits == nil {
            let owner = try self.owner()
            var output = [Float](repeating: 0, count: Int(flf_vocab_size(owner.pointer)))
            guard flf_copy_logits(owner.pointer, &output, Int32(output.count)) == Int32(output.count) else { throw failure() }
            lastLogits = output
        }
        return (lastLogits!, reused)
    }
    private func prepare(tokens: [Int32], progress: @Sendable (Int, Int) -> Void) throws -> Int {
        guard !tokens.isEmpty, tokens.count < configuration.contextTokens else { throw FlintFailure.invalid("Flint prompt leaves no output capacity") }
        if try metrics().cachedTokens != UInt64(stateTokens.count) { try resetKV() }
        if FlintPrefixPolicy.requiresReset(cached: stateTokens, prompt: tokens) ||
            (tokens == stateTokens && !hasCurrentLogits) { try resetKV() }
        let reused = stateTokens.count
        var first = reused
        while first < tokens.count {
            let end = min(first + configuration.evaluationBatch, tokens.count)
            _ = try append(tokens: Array(tokens[first..<end]), finalRow: end == tokens.count, copyLogits: false)
            progress(end, tokens.count); first = end
        }
        guard hasCurrentLogits else { throw FlintFailure.invalid("Flint prefill produced no logits") }
        return reused
    }
}

/// One request's cancellation cannot cancel a later request using the same
/// loaded model. Only the locked cancellation flag crosses the actor boundary.
private final class FlintRequestLease: @unchecked Sendable {
    private let lock = NSLock()
    private var finished = false
    private let control: FlintCancellation
    init(_ control: FlintCancellation) { self.control = control }
    func cancel() { lock.lock(); defer { lock.unlock() }; if !finished { control.cancel() } }
    func finish() { lock.lock(); finished = true; lock.unlock() }
}

extension FlintSession {
    /// Reserves the sole request before HTTP headers are sent. The same actor
    /// can serve foreground chat and LAN requests without a second model copy.
    public func start(_ request: LocalChatRequest,
                      progress: @escaping @Sendable (Int, Int) -> Void = { _, _ in }) throws -> LocalChatGeneration {
        _ = try owner()
        guard numericalAdmission else { throw LocalAPIError(status: 503, message: "Complete Flint's CPU/Metal admission before generation") }
        guard activeRequest == nil else { throw LocalAPIError(status: 429, message: "Flint is already serving a request") }
        guard request.requestedModelID == LocalModelCapabilities(family: .flint).modelID else {
            throw LocalAPIError(status: 404, message: "The requested model is not loaded", parameter: "model")
        }
        let id = UUID(), started = Date.timeIntervalSinceReferenceDate
        cancellation.begin(ceiling: activeCeiling); activeRequest = id; generationReport = nil
        let lease = FlintRequestLease(cancellation)
        let pair = AsyncThrowingStream<LocalChatEvent, Error>.makeStream(bufferingPolicy: .bufferingOldest(512))
        pair.continuation.onTermination = { _ in lease.cancel() }
        // Inherits this inference actor, never MainActor. Native calls are
        // serial; cancellation is nonisolated and does not touch the handle.
        Task { [self] in
            defer { lease.finish(); if activeRequest == id { activeRequest = nil } }
            do {
                try produce(request, started: started, progress: progress, continuation: pair.continuation)
                lease.finish(); pair.continuation.finish()
            } catch {
                try? resetKV()
                lease.finish(); pair.continuation.finish(throwing: error)
            }
        }
        return .init(events: pair.stream, cancel: { lease.cancel() })
    }

    private func produce(_ request: LocalChatRequest, started: Double, progress: @Sendable (Int, Int) -> Void,
                         continuation: AsyncThrowingStream<LocalChatEvent, Error>.Continuation) throws {
        func emit(_ event: LocalChatEvent) throws {
            switch continuation.yield(event) {
            case .enqueued: break
            case .terminated: throw CancellationError()
            case .dropped: throw LocalAPIError(status: 503, message: "The client is not consuming the local response; generation stopped")
            @unknown default: throw CancellationError()
            }
        }
        let owner = try self.owner(), thermal = ProcessInfo.processInfo.thermalState.rawValue
        try artifact?.assertUnchanged()
        let prompt = try tokenize(renderPrompt(requestJSON: FlintWirePrompt.encode(request)))
        guard prompt.count + request.options.maximumNewTokens <= configuration.contextTokens else {
            throw LocalAPIError.invalid("messages", "Flint's admitted context must hold the complete template, tools, prompt and requested output")
        }
        guard !cancellation.shouldAbort() else { throw CancellationError() }
        let initial = try metrics(), prefillStart = Date.timeIntervalSinceReferenceDate
        let reused = try prepare(tokens: prompt, progress: progress)
        let prefillSeconds = Date.timeIntervalSinceReferenceDate - prefillStart, afterPrefill = try metrics()
        guard flf_configure_sampling(owner.pointer, Float(request.options.temperature), Float(request.options.topP), 20, UInt32(request.options.seed)) == 0 else { throw failure() }
        var decoder = FlintTextStream(), generated: [Int32] = [], reasoningTokens = 0, finish = "length"
        var generatedBytes = 0
        var firstToken: Double?
        for index in 0..<request.options.maximumNewTokens {
            guard !Task.isCancelled, !cancellation.shouldAbort() else { throw CancellationError() }
            let token = flf_sample(owner.pointer)
            guard token >= 0 else { throw failure() }
            if flf_is_eog(owner.pointer, token) { finish = "stop"; break }
            if firstToken == nil { firstToken = Date.timeIntervalSinceReferenceDate - started }
            let needed = flf_token_piece(owner.pointer, token, nil, 0)
            guard needed <= 0, needed >= -65536 else { throw failure() }
            var bytes = [CChar](repeating: 0, count: Int(-needed))
            if needed < 0 {
                guard flf_token_piece(owner.pointer, token, &bytes, Int32(bytes.count)) == -needed else { throw failure() }
            }
            generatedBytes += bytes.count
            guard generatedBytes <= 4 << 20 else { throw LocalAPIError(status: 500, message: "Flint output exceeds the bounded 4 MiB response budget") }
            let wasThinking = decoder.isThinking
            let delta = try decoder.append(bytes.map { UInt8(bitPattern: $0) })
            if wasThinking || decoder.isThinking || !delta.reasoning.isEmpty { reasoningTokens += 1 }
            generated.append(token)
            if !delta.content.isEmpty || !delta.reasoning.isEmpty { try emit(.delta(content: delta.content, reasoning: delta.reasoning)) }
            // Commit even the last sampled token to KV so a following exact
            // conversation prefix can be reused. Its final logits are unused.
            _ = try append(tokens: [token], finalRow: index + 1 < request.options.maximumNewTokens, copyLogits: false)
        }
        let end = try decoder.finish()
        if !end.content.isEmpty || !end.reasoning.isEmpty { try emit(.delta(content: end.content, reasoning: end.reasoning)) }
        let final = try metrics(), observation = cancellation.observation()
        let decodeSeconds = max(0, final.forwardSeconds - afterPrefill.forwardSeconds)
        let elapsed = Date.timeIntervalSinceReferenceDate - started
        let metrics = LocalInferenceMetrics(modelSHA256: FlintArtifact.manifest.sha256, contextTokens: configuration.contextTokens,
            evaluatedPromptTokens: prompt.count - reused, reusedPromptTokens: reused, decodeForwardPasses: generated.count,
            prefillSeconds: prefillSeconds, decodeSeconds: decodeSeconds, elapsedSeconds: elapsed,
            prefillIOBytes: afterPrefill.expertIOBytes - initial.expertIOBytes, decodeIOBytes: final.expertIOBytes - afterPrefill.expertIOBytes,
            peakMemoryBytes: observation.peak, thermalStateAtStart: thermal, maximumThermalState: observation.thermal,
            platform: Self.platform, appBuild: Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String,
            nativeRevision: Self.nativeRevision, memoryBudgetBytes: activeCeiling, timeToFirstTokenSeconds: firstToken)
        generationReport = .init(modelSHA256: FlintArtifact.manifest.sha256, nativeRevision: Self.nativeRevision,
            contextTokens: configuration.contextTokens, evaluationBatch: configuration.evaluationBatch,
            physicalBatch: configuration.physicalBatch, cpuThreads: configuration.cpuThreads,
            processMemoryBudgetBytes: activeCeiling, platform: Self.platform,
            operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString, sourceRevision: BuildInfo.gitRevision,
            appBuild: Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String, timestamp: Date(),
            promptTokenIDs: prompt, generatedTokenIDs: generated,
            reusedPromptTokens: reused, prefillSeconds: prefillSeconds, decodeForwardSeconds: decodeSeconds,
            elapsedSeconds: elapsed, timeToFirstTokenSeconds: firstToken, maximumObservedFootprintBytes: observation.peak,
            explicitExpertBytesRead: final.expertIOBytes - initial.expertIOBytes, maximumThermalState: observation.thermal,
            finishReason: finish)
        try emit(.complete(.init(promptTokens: prompt.count, completionTokens: generated.count, reasoningTokens: reasoningTokens,
            forcedTokens: 0, finishReason: finish, metrics: metrics)))
    }
    private static var platform: String {
        #if targetEnvironment(simulator)
        "iOSSimulator"
        #elseif os(iOS)
        "iOSDevice"
        #else
        "macOS"
        #endif
    }
}

extension FlintSession {
    /// Fixed harmless throughput prompt, measured with the real tokenizer.
    /// Selecting 10K requires a larger admitted window. No synthetic inference.
    public func benchmarkRequest(promptTokens target: Int) throws -> LocalChatRequest {
        guard numericalAdmission, [1000, 4000, 10000].contains(target), target + 256 <= configuration.contextTokens else {
            throw FlintFailure.invalid("Load a context that holds the benchmark prompt plus 256 output tokens")
        }
        var repeated = target
        for _ in 0..<6 {
            let content = "Read this harmless repeated context, then list the integers 1 through 250, one per line. Do not summarize.\n<context>" +
                String(repeating: " x", count: repeated) + "</context>\nStart the list now."
            let request = try LocalChatRequest(data: JSONSerialization.data(withJSONObject: ["model": LocalModelCapabilities(family: .flint).modelID,
                "messages": [["role": "user", "content": content]], "temperature": 0, "top_p": 1, "seed": 42, "max_tokens": 256]),
                capabilities: .init(context: LocalContextProfile(rawValue: configuration.contextTokens) ?? .standard, family: .flint))
            let actual = try tokenize(renderPrompt(requestJSON: FlintWirePrompt.encode(request))).count
            if actual == target { return request }
            repeated += target - actual
            guard repeated >= 0, repeated <= target * 2 else { break }
        }
        throw FlintFailure.invalid("The actual tokenizer did not produce the exact requested benchmark length")
    }

    /// A short, real-model CPU/Metal comparison before chat/API admission.
    /// The HMAC record is local and bound to executable/model/configuration;
    /// normal reloads reuse it but still check file identity and live memory.
    /// This verifies four positions, not model quality or long-context recall.
    public func prepare(artifact: VerifiedGGUFArtifact, privateRoot: URL, configuration: FlintConfiguration,
                        progress: @escaping @Sendable (FlintProgress) -> Void = { _ in }) throws -> FlintNumericalCheck {
        guard activeRequest == nil else { throw LocalAPIError(status: 429, message: "Finish the active Flint request before preparing its model") }
        numericalReport = nil
        try artifact.assertUnchanged(); try configuration.validate()
        guard configuration.backend == .metal, !configuration.diagnosticRouting else { throw FlintFailure.invalid("Serving requires the ordinary Metal configuration") }
        guard let executable = Bundle.main.executableURL else { throw FlintFailure.invalid("Missing executable identity") }
        guard let libraryPath = flf_library_path() else { throw failure() }
        try LocalModelStorage.prepare(root: privateRoot)
        let identity = ["model": artifact.manifest.sha256, "native": Self.nativeRevision,
            "native_binary": try FileDigestComputer.digest(ofFileAt: URL(fileURLWithPath: String(cString: libraryPath)), algorithm: .sha256).hex,
            "executable": try FileDigestComputer.digest(ofFileAt: executable, algorithm: .sha256).hex,
            "os": ProcessInfo.processInfo.operatingSystemVersionString, "context": String(configuration.contextTokens),
            "evaluation": String(configuration.evaluationBatch), "physical": String(configuration.physicalBatch),
            "policy": "flint-cpu-f32kv-metal-f16kv-four-position-expert-sets-v1"]
        let url = privateRoot.appendingPathComponent("numerical-admission-v1.json")
        let key = try LocalAPIKeyStore(service: "com.dhafen.flashlocal.flint-admission.v1").loadOrCreate()
        if let size = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize, size <= 2 << 20,
           let data = try? Data(contentsOf: url), let opened = LocalAdmissionSeal.open(data, identity: identity, key: key),
           let saved = try? JSONDecoder().decode(FlintNumericalCheck.self, from: opened), saved.passed {
            try load(artifact: artifact, privateRoot: privateRoot, configuration: configuration, progress: progress)
            numericalReport = saved; numericalAdmission = true; return saved
        }
        unload()
        do {
            var cpu = configuration; cpu.backend = .cpuReference; cpu.contextTokens = 256
            cpu.evaluationBatch = 32; cpu.physicalBatch = 1; cpu.diagnosticRouting = true
            try load(artifact: artifact, privateRoot: privateRoot, configuration: cpu, progress: progress)
            let input = try JSONSerialization.data(withJSONObject: ["messages": [["role": "user", "content": "Return the sum of 50 and 5."]]])
            let tokens = try tokenize(renderPrompt(requestJSON: input))
            guard tokens.count >= 8, tokens.count < 64 else { throw FlintFailure.invalid("Unexpected Flint control tokenizer length") }
            let positions = [0, 3, tokens.count / 2, tokens.count - 1]
            try traceRouting(true)
            var references: [[Float]] = [], first = 0
            for position in positions {
                references.append(try evaluate(tokens: Array(tokens[first...position]), outputLogits: true)!)
                first = position + 1
            }
            let referenceRoutes = try routingDecisions()
            unload(); try Task.checkCancellation()
            var metal = configuration; metal.diagnosticRouting = true
            try load(artifact: artifact, privateRoot: privateRoot, configuration: metal, progress: progress)
            try traceRouting(true)
            var comparisons: [FlintNumericalCheck.Position] = []; first = 0
            for (number, position) in positions.enumerated() {
                let logits = try evaluate(tokens: Array(tokens[first...position]), outputLogits: true)!
                comparisons.append(try FlintNumericalCheck.compare(position: position, reference: references[number], candidate: logits))
                first = position + 1
                progress(.init(phase: 4, fraction: Double(number + 1) / Double(positions.count), observedFootprintBytes: ProcessFootprint.current()?.footprintBytes))
            }
            let actualRoutes = try routingDecisions()
            let check = FlintNumericalCheck(positions: comparisons, routedTokenLayers: referenceRoutes.count,
                // Non-output positions skip the last layer's FFN. Compare
                // every actually computed top-6 set, including all 47 layers
                // at each of the four requested logit positions.
                exactExpertSets: referenceRoutes.count == tokens.count * 46 + positions.count && FlintNumericalCheck.sameRoutes(referenceRoutes, actualRoutes))
            numericalReport = check
            unload()
            guard check.passed else { throw FlintFailure.invalid("Flint's CPU/Metal comparison differed; no new chat/API backend was admitted") }
            // Serving has no routing graph callbacks or trace storage.
            try load(artifact: artifact, privateRoot: privateRoot, configuration: configuration, progress: progress)
            let sealed = try LocalAdmissionSeal.seal(JSONEncoder().encode(check), identity: identity, key: key)
            try sealed.write(to: url, options: [.atomic])
            numericalAdmission = true
            return check
        } catch { unload(); throw error }
    }
}
