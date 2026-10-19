#if canImport(OrnithLES)
import Foundation
import Observation
import OrnithLES
import MinirunKit
import MinirunRunners
import ModelAdapters
import StorageCore

private enum LESChatPresentationError: Error, LocalizedError {
    case notReady, streamFull, numericalCheck
    var errorDescription: String? {
        switch self {
        case .notReady: "Snabbchatten behöver aktiveras."
        case .streamFull: "Gränssnittet hann inte ta emot texten. Genereringen stoppades."
        case .numericalCheck: "Numerikkontrollen avvek. Snabbchatten aktiverades inte."
        }
    }
}

/// Transport teardown can race a completed owner. A finished request may not
/// cancel the reusable native context after its last event has been delivered.
private final class LESRequestLease: @unchecked Sendable {
    private let lock = NSLock()
    private var finished = false
    private let control: LESControl
    init(_ control: LESControl) { self.control = control }
    func finish() { lock.lock(); finished = true; lock.unlock() }
    func cancel() { lock.lock(); defer { lock.unlock() }; if !finished { control.cancel() } }
}

private struct LESRunExport: Encodable {
    let measurement: LESBenchmarkRun
    let decodeForwardTokensPerSecond: Double
    let prefillTokensPerSecond: Double
    let decodeIOBytesPerForward: Double
    let hasRequired128Tokens: Bool
    let metalDecodeNanoseconds: UInt64?
    let metalDecodeDispatches: UInt64?
    let expertCacheHitRate: Double?
    let lookaheadActive: Bool?
    init(_ run: LESBenchmarkRun) {
        measurement = run; decodeForwardTokensPerSecond = run.decodeTokensPerSecond
        prefillTokensPerSecond = run.prefillTokensPerSecond
        decodeIOBytesPerForward = Double(run.decodeIOBytes) / Double(max(1, run.decodeForwardPasses))
        hasRequired128Tokens = run.hasRequiredLength
        lookaheadActive = run.memoryPlan.map { $0.lookaheadDepth > 0 }
        if let before = run.metricsAfterPrefill, let after = run.metricsAfter {
            metalDecodeNanoseconds = after.metalTimestampedCommandBuffers > before.metalTimestampedCommandBuffers
                ? after.metalNanoseconds - before.metalNanoseconds : nil
            metalDecodeDispatches = after.metalDispatches - before.metalDispatches
            let hits = after.demandCacheHits - before.demandCacheHits
            let misses = after.demandCacheMisses - before.demandCacheMisses
            expertCacheHitRate = hits + misses == 0 ? nil : Double(hits) / Double(hits + misses)
        } else { metalDecodeNanoseconds = nil; metalDecodeDispatches = nil; expertCacheHitRate = nil }
    }
}
private struct LESAppReport: Encodable {
    let schemaVersion = 1
    let appVersion: String
    let appBuild: String
    let sourceRevision: String
    let timestamp: Date
    let platform: String
    let deviceIdentifier: String?
    let physicalMemoryBytes: UInt64
    let operatingSystem: String
    let lowPowerModeEnabled: Bool
    let userDeclaredAirplaneMode: Bool
    let networkInferenceUsed = false
    let nativeExpertCachePolicy = "OS page cache explicitly enabled; own decode windows vary by mode"
    let originalExpertCachePolicy = "Existing TileReader F_NOCACHE=1 advisory; original backend preserved"
    let fullModelNumericsVerified = false
    let promptID = LESBenchmark.promptID
    let contextCapacityTokens = 4096
    let numericalCheck: LESNumericalCheck?
    let measurements: [LESRunExport]
    let memoryAdmissions: [LESMemoryAdmission]
    let failure: String?
    let maximumObservedFootprintBytes: UInt64
    let modelSHA256 = GGUFModelManifest.primaryOrnith.sha256
    let nativeRevision = LESContext.revision
}
private actor LESAppWorker {
    private var artifact: VerifiedGGUFArtifact?
    private var prepared: URL?
    private var chat: OrnithLESChatGeneration?
    private var chatCheck: LESNumericalCheck?
    private var admissions: [LESMemoryAdmission] = []
    func memoryAdmissionReport() -> [LESMemoryAdmission] { admissions }
    func release() { chat?.close(); chat = nil; chatCheck = nil; artifact = nil; prepared = nil }
    func resetChat() { chat?.resetConversation() }
    func closeChat() { chat?.close(); chat = nil }
    private func prepareFiles(modelsRoot: URL, privateRoot: URL, existing: VerifiedGGUFArtifact?, control: LESControl,
                              progress: @Sendable (String) -> Void) throws {
        if artifact == nil {
            if let existing, existing.manifest.sha256 == GGUFModelManifest.primaryOrnith.sha256 {
                try existing.assertUnchanged(); artifact = existing
            } else {
                progress("Verifierar din befintliga Ornith-GGUF. Ingen modell hämtas.")
                artifact = try VerifiedGGUFArtifact.verify(insideRoot: modelsRoot, manifest: .primaryOrnith,
                    cancellationRequested: { control.shouldAbort() })
            }
        }
        guard let artifact else { throw CancellationError() }
        if prepared == nil {
            progress("Förbereder fasta vikter. Experterna ligger kvar i din GGUF.")
            prepared = try LESPreparation.prepare(artifact: artifact, privateRoot: privateRoot, control: control,
                onAdmission: { self.admissions.append($0) })
        }
    }
    func prepareChat(modelsRoot: URL, privateRoot: URL, existing: VerifiedGGUFArtifact?, control: LESControl,
                     progress: @Sendable (String) -> Void,
                     onCheck: @Sendable (LESNumericalCheck) -> Void) throws -> LESNumericalCheck {
        chat?.close(); chat = nil
        try prepareFiles(modelsRoot: modelsRoot, privateRoot: privateRoot, existing: existing, control: control, progress: progress)
        guard let artifact, let prepared else { throw CancellationError() }
        if chatCheck == nil {
            progress("Kontrollerar mot originalvikterna innan snabbchatten startas…")
            chatCheck = try LESBenchmark.check(artifact: artifact, prepared: prepared, control: control,
                progress: progress, onAdmission: { self.admissions.append($0) })
        }
        guard let check = chatCheck else { throw CancellationError() }
        onCheck(check)
        guard check.nativeBenchmarkAdmitted else { return check }
        chat = try OrnithLESChatGeneration(artifact: artifact, prepared: prepared, check: check,
            control: control, onAdmission: { self.admissions.append($0) })
        return check
    }
    func generateChat(messages: [Qwen35ChatMessage], options: LocalGenerationOptions,
                      events: AsyncThrowingStream<Qwen4ExpTextGeneration.Event, Error>.Continuation)
        -> Result<OrnithLESChatGeneration.Summary, Error> {
        let result = Result {
            guard let chat else { throw LESChatPresentationError.notReady }
            return try autoreleasepool {
                try chat.generate(messages: messages, options: options) { event in
                    if case .dropped = events.yield(event) {
                        throw LESChatPresentationError.streamFull
                    }
                }
            }
        }
        if case .failure = result { chat?.close(); chat = nil }
        return result
    }
    func execute(modelsRoot: URL, privateRoot: URL, existing: VerifiedGGUFArtifact?, control: LESControl,
                 progress: @Sendable (String) -> Void,
                 onCheck: @Sendable (LESNumericalCheck) -> Void,
                 onRun: @Sendable (LESBenchmarkRun) -> Void) throws -> (LESNumericalCheck, [LESBenchmarkRun]) {
        admissions.removeAll(keepingCapacity: true)
        chat?.close(); chat = nil; chatCheck = nil
        try prepareFiles(modelsRoot: modelsRoot, privateRoot: privateRoot, existing: existing, control: control, progress: progress)
        guard let artifact, let prepared else { throw CancellationError() }
        // Every explicit run performs its own check and memory sampling.
        // Re-exporting a cached failed check must not look like another run
        // with zero measured RAM. The verified file/preparation may be reused.
        let check = try LESBenchmark.check(artifact: artifact, prepared: prepared, control: control,
            progress: progress, onAdmission: { self.admissions.append($0) })
        chatCheck = check.nativeBenchmarkAdmitted ? check : nil
        onCheck(check)
        guard check.nativeBenchmarkAdmitted else { return (check, []) }
        var runs: [LESBenchmarkRun] = []
        for backend in LESBenchmarkBackend.allCases {
            let pair = try autoreleasepool {
                try LESBenchmark.runPair(backend: backend, artifact: artifact, prepared: prepared,
                    check: check, control: control, progress: progress, onAdmission: { self.admissions.append($0) })
            }
            for run in pair { runs.append(run); onRun(run) }
        }
        return (check, runs)
    }
}

@Observable @MainActor
final class OrnithLESAppModel {
    private(set) var isBusy = false
    private(set) var isShuttingDown = false
    private(set) var isPreview = false
    private(set) var status = "Använder Ornith-GGUF som redan finns på telefonen."
    private(set) var error: String?
    private(set) var numericalCheck: LESNumericalCheck?
    private(set) var runs: [LESBenchmarkRun] = []
    private(set) var memoryAdmissions: [LESMemoryAdmission] = []
    private(set) var measurementURL: URL?
    private(set) var chatReady = false
    private(set) var chatMessages: [REAMMessage] = []
    private(set) var chatProgress: Qwen4ExpTextGeneration.Progress?
    private(set) var chatSummary: OrnithLESChatGeneration.Summary?
    private(set) var chatMeasurementURL: URL?
    var chatDraft = ""
    var chatOptions = LocalGenerationOptions()
    var chatThinkingEnabled: Bool {
        get { chatOptions.enableThinking }
        set { chatOptions.enableThinking = newValue }
    }
    private(set) var isAPIRunning = false
    private(set) var apiStatus = "Avstängd"
    private(set) var apiAccessKey = ""
    private(set) var apiURLs: [String] = []
    private(set) var apiCompletedRequests = 0
    var apiPort = 8080
    var apiKeepAwake = true
    @ObservationIgnored private var apiServer: LocalModelAPIServer?
    @ObservationIgnored private var apiID: UUID?
    @ObservationIgnored private var operationIsAPI = false
    var userDeclaredAirplaneMode = false
    let modelsRoot: URL
    let privateRoot: URL
    let documents: URL
    @ObservationIgnored private let worker = LESAppWorker()
    @ObservationIgnored private var operation: Task<Void, Never>?
    @ObservationIgnored private var control: LESControl?
    @ObservationIgnored private var chatControl: LESControl?
    @ObservationIgnored private var activeID: UUID?
    var canStart: Bool { !isBusy && !isShuttingDown && !isPreview }
    var canSendChat: Bool { canStart && chatReady }
    var canBenchmark: Bool { canStart && apiServer == nil }
    var shouldKeepAwake: Bool { isBusy || (isAPIRunning && apiKeepAwake) }
    init(modelsRoot: URL, privateRoot: URL, documents: URL) {
        self.modelsRoot = modelsRoot; self.privateRoot = privateRoot; self.documents = documents
    }
    static func live() -> OrnithLESAppModel {
        let fm = FileManager.default
        let documents = fm.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return Self(modelsRoot: documents.appendingPathComponent("Models"),
            privateRoot: fm.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("FlashLocal/OrnithLES"), documents: documents)
    }
    static func reviewPreview() -> OrnithLESAppModel {
        let model = Self.live(); model.isPreview = true
        model.status = "Förhandsvisning av layout – ingen modell körs."
        return model
    }
    static func reviewChatPreview(thinking: Bool) -> OrnithLESAppModel {
        let model = Self.reviewPreview()
        model.chatReady = true; model.isBusy = thinking
        model.status = thinking ? "Tänker lokalt…" : "Förhandsvisning av snabbchatten – ingen modell körs."
        model.chatMessages = [.init(role: .user, text: "Hur testar jag en cache?")]
        model.chatMessages.append(.init(role: .assistant,
            text: thinking ? "" : "Testa träffar, missar och att den äldsta posten byts ut när cachen är full.",
            reasoning: "Jag börjar med cachens gränser och vilka egenskaper testet behöver kontrollera.", isThinking: thinking))
        return model
    }
    static func reviewAPIPreview() -> OrnithLESAppModel {
        let model = Self.reviewPreview()
        model.apiStatus = "Förhandsvisning · ingen server eller modell körs"
        model.apiURLs = ["http://192.168.1.25:8080/v1"]
        model.apiAccessKey = "example-only-not-a-live-key"
        model.chatReady = true; model.isAPIRunning = true
        return model
    }
    func start(existing: VerifiedGGUFArtifact? = nil) {
        guard canBenchmark else { return }
        chatReady = false; chatControl = nil
        let flag = LESControl(), id = UUID()
        activeID = id; control = flag; isBusy = true; error = nil; runs = []; memoryAdmissions = []; measurementURL = nil
        numericalCheck = nil
        let declaredAirplane = userDeclaredAirplaneMode
        operation = Task { [weak self] in
            guard let self else { return }
            defer { if activeID == id { activeID = nil; control = nil; isBusy = false; operation = nil } }
            do {
                let result = try await worker.execute(modelsRoot: modelsRoot, privateRoot: privateRoot,
                    existing: existing, control: flag, progress: { [weak self] value in
                        Task { @MainActor in guard let self, self.activeID == id else { return }; self.status = value }
                    }, onCheck: { [weak self] value in
                        Task { @MainActor in guard let self, self.activeID == id else { return }; self.numericalCheck = value }
                    }, onRun: { [weak self] value in
                        Task { @MainActor in guard let self, self.activeID == id else { return }; self.runs.append(value) }
                    })
                guard activeID == id else { return }
                memoryAdmissions = await worker.memoryAdmissionReport()
                numericalCheck = result.0; runs = result.1
                status = !result.0.nativeBenchmarkAdmitted ? "Numeriken avvek. LES-testet stoppades; den vanliga chatten är oförändrad." :
                    runs.allSatisfy(\.hasRequiredLength) ? "Jämförelsen är klar." : "Något svar blev kortare än 128 token. Den körningen är inte jämförbar."
                try save(check: result.0, runs: result.1, airplane: declaredAirplane)
            } catch {
                guard activeID == id else { return }
                memoryAdmissions = await worker.memoryAdmissionReport()
                self.error = error.localizedDescription; status = "Testet stoppades."
                try? save(check: numericalCheck, runs: runs, airplane: declaredAirplane, failure: error.localizedDescription)
            }
        }
    }
    func stop() { guard isBusy else { return }; status = "Stoppar och frigör motorn…"; control?.cancel() }
    func shutdown() async {
        guard !isShuttingDown else { await operation?.value; return }
        isShuttingDown = true
        stopAPI()
        stop(); await operation?.value; await worker.release()
        chatReady = false; chatControl = nil; isShuttingDown = false
    }
    func activateChat(existing: VerifiedGGUFArtifact? = nil) {
        guard canStart, !chatReady else { return }
        let flag = LESControl(), id = UUID()
        activeID = id; control = flag; chatControl = flag; isBusy = true; error = nil
        status = "Förbereder Ornith snabbchatt…"
        operation = Task { [weak self] in
            guard let self else { return }
            defer { if activeID == id { activeID = nil; control = nil; isBusy = false; operation = nil } }
            do {
                let check = try await worker.prepareChat(modelsRoot: modelsRoot, privateRoot: privateRoot, existing: existing,
                    control: flag, progress: { [weak self] value in
                        Task { @MainActor in guard let self, self.activeID == id else { return }; self.status = value }
                    }, onCheck: { [weak self] value in
                        Task { @MainActor in guard let self, self.activeID == id else { return }; self.numericalCheck = value }
                    })
                // The awaited owner result is authoritative. Progress tasks
                // may arrive after this operation has left its active ID.
                numericalCheck = check
                guard check.nativeBenchmarkAdmitted else {
                    throw LESChatPresentationError.numericalCheck
                }
                if flag.isCancellationRequested { throw CancellationError() }
                chatReady = true; status = "Snabbchatten är redo · samma Ornith-GGUF · OS-cache."
            } catch {
                await worker.release(); chatReady = false; chatControl = nil
                if !flag.isCancellationRequested { self.error = error.localizedDescription }
                status = "Snabbchatten kunde inte startas."
            }
        }
    }
    func sendChat() {
        guard canSendChat, let flag = chatControl else { return }
        let input = chatDraft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !input.isEmpty, input.utf8.count <= 262144 else { return }
        do { try chatOptions.validate() } catch { self.error = error.localizedDescription; return }
        chatDraft = ""; chatMessages.append(.init(role: .user, text: input))
        let prompt = REAMMessage.generationPrompt(chatMessages)
        chatMessages.append(.init(role: .assistant, text: "", interrupted: true))
        let response = chatMessages.last!.id, id = UUID(), options = chatOptions
        let thinking = options.thinkingEnabled
        activeID = id; control = flag; isBusy = true; error = nil
        chatProgress = nil; chatSummary = nil; chatMeasurementURL = nil; status = "Förbereder lokal generering…"
        let channel = AsyncThrowingStream<Qwen4ExpTextGeneration.Event, Error>.makeStream(bufferingPolicy: .bufferingOldest(64))
        channel.continuation.onTermination = { if case .cancelled = $0 { flag.cancel() } }
        let worker = worker
        let task = Task.detached(priority: .userInitiated) {
            let result = await worker.generateChat(messages: prompt, options: options, events: channel.continuation)
            switch result {
            case .success: channel.continuation.finish()
            case .failure(let error): channel.continuation.finish(throwing: error)
            }
            return result
        }
        operation = Task { [weak self] in
            guard let self else { flag.cancel(); _ = await task.value; return }
            defer { if activeID == id { activeID = nil; control = nil; isBusy = false; operation = nil } }
            var output = QwenThinkingText(startsInThinking: thinking)
            do {
                for try await event in channel.stream {
                    guard let index = chatMessages.firstIndex(where: { $0.id == response }) else { continue }
                    switch event {
                    case .prefill(let value):
                        chatProgress = value
                        if !flag.isCancellationRequested { status = "Läser prompt: \(value.promptTokensCompleted)/\(value.promptTokensTotal) · \(value.reusedPromptTokens) återanvända" }
                    case .token(_, let text, let value):
                        chatProgress = value; output.append(text)
                        chatMessages[index].text = output.answer
                        chatMessages[index].reasoning = output.reasoning.isEmpty ? nil : output.reasoning
                        chatMessages[index].isThinking = output.isThinking
                        if !flag.isCancellationRequested { status = output.isThinking ? "Tänker lokalt…" : "Skriver svaret lokalt…" }
                    case .trailingText(let text): output.append(text)
                    }
                }
            } catch { /* Await the inference owner and teardown below. */ }
            let result = await task.value
            output.finish()
            if let index = chatMessages.firstIndex(where: { $0.id == response }) {
                chatMessages[index].text = output.answer
                chatMessages[index].reasoning = output.reasoning.isEmpty ? nil : output.reasoning
                chatMessages[index].isThinking = false
                if case .success = result { chatMessages[index].interrupted = output.isThinking }
            }
            switch result {
            case .success(let value):
                chatSummary = value
                status = output.isThinking ? "Tokengränsen nåddes under tänkandet · inget färdigt svar." : "Klar · \(value.generatedTokenIDs.count) genererade token"
            case .failure(let failure):
                chatReady = false; chatControl = nil
                if !flag.isCancellationRequested { error = failure.localizedDescription }
                status = "Genereringen stoppades. Aktivera snabbchatten för att fortsätta."
            }
        }
    }
    func startAPI() {
        guard canStart, chatReady, apiServer == nil else { return }
        guard (1024...65535).contains(apiPort) else { error = "Välj en port mellan 1024 och 65535."; return }
        do {
            let key = try LocalModelAPIServer.makeAccessKey(), id = UUID()
            apiAccessKey = key; apiID = id; apiStatus = "Startar lokalt API…"; error = nil
            apiServer = try LocalModelAPIServer(key: key, port: UInt16(apiPort), defaults: chatOptions, handler: { [weak self] request in
                guard let self else { throw LocalAPIError(status: 503, message: "Phone service is closed") }
                return try await self.openAPIRequest(request)
            }, onState: { [weak self] state in
                Task { @MainActor in
                    guard let self, self.apiID == id else { return }
                    switch state {
                    case .ready(let port, let addresses):
                        self.isAPIRunning = true
                        self.apiURLs = addresses.map { "http://\($0):\(port)/v1" }
                        self.apiStatus = addresses.isEmpty ? "Servern lyssnar. Anslut till Wi-Fi eller Ethernet och starta om API:t för att visa adressen." : "API aktivt · samma lokala Ornith-motor"
                    case .failed(let message):
                        self.error = message; self.stopAPI()
                    case .stopped: self.stopAPI()
                    }
                }
            })
        } catch { self.error = error.localizedDescription; stopAPI() }
    }
    func stopAPI() {
        apiID = nil; apiServer?.stop(); apiServer = nil
        isAPIRunning = false; apiAccessKey = ""; apiURLs = []; apiStatus = "Avstängd"
        if operationIsAPI { control?.cancel() }
    }
    /// The transport calls this before sending successful headers. All UI and
    /// API requests reserve the same owner; no second model context is loaded.
    func openAPIRequest(_ request: LocalChatRequest) throws -> LocalChatGeneration {
        guard apiServer != nil, !isShuttingDown, !isPreview else {
            throw LocalAPIError(status: 503, message: "Start the local API on the phone")
        }
        guard !isBusy else { throw LocalAPIError(status: 429, message: "The phone is processing another request") }
        guard chatReady, let flag = chatControl else {
            throw LocalAPIError(status: 503, message: "Activate Ornith quick chat on the phone first")
        }
        let id = UUID(), lease = LESRequestLease(flag)
        let output = AsyncThrowingStream<LocalChatEvent, Error>.makeStream(bufferingPolicy: .bufferingOldest(64))
        output.continuation.onTermination = { if case .cancelled = $0 { lease.cancel() } }
        let input = request.messages.map { Qwen35ChatMessage(role: Qwen35ChatMessage.Role(rawValue: $0.role)!, content: $0.content, reasoningContent: $0.reasoning) }
        let channel = AsyncThrowingStream<Qwen4ExpTextGeneration.Event, Error>.makeStream(bufferingPolicy: .bufferingOldest(64))
        activeID = id; control = flag; isBusy = true; operationIsAPI = true; error = nil
        status = "API: förbereder lokal generering…"
        let worker = worker
        let owner = Task.detached(priority: .userInitiated) {
            let result = await worker.generateChat(messages: input, options: request.options, events: channel.continuation)
            lease.finish()
            switch result { case .success: channel.continuation.finish(); case .failure(let error): channel.continuation.finish(throwing: error) }
            return result
        }
        operation = Task { [weak self] in
            guard let self else { lease.cancel(); _ = await owner.value; output.continuation.finish(); return }
            defer {
                lease.finish()
                if activeID == id { activeID = nil; control = nil; isBusy = false; operationIsAPI = false; operation = nil }
            }
            var text = QwenThinkingText(startsInThinking: request.options.thinkingEnabled)
            var answerBytes = 0, reasoningBytes = 0
            func emit() throws {
                let answer = String(decoding: text.answer.utf8.dropFirst(answerBytes), as: UTF8.self)
                let reasoning = String(decoding: text.reasoning.utf8.dropFirst(reasoningBytes), as: UTF8.self)
                answerBytes = text.answer.utf8.count; reasoningBytes = text.reasoning.utf8.count
                if !answer.isEmpty || !reasoning.isEmpty {
                    switch output.continuation.yield(.delta(content: answer, reasoning: reasoning)) {
                    case .enqueued: break
                    default: lease.cancel(); throw LESChatPresentationError.streamFull
                    }
                }
            }
            do {
                for try await event in channel.stream {
                    switch event {
                    case .prefill(let value): status = "API: läser prompt \(value.promptTokensCompleted)/\(value.promptTokensTotal)"
                    case .token(_, let delta, _):
                        text.append(delta); try emit()
                        status = text.isThinking ? "API: tänker lokalt…" : "API: skriver svaret…"
                    case .trailingText(let delta): text.append(delta); try emit()
                    }
                }
                text.finish(); try emit()
            } catch { lease.cancel() }
            var result = await owner.value
            if flag.isCancellationRequested { await worker.closeChat(); result = .failure(CancellationError()) }
            lease.finish()
            switch result {
            case .success(let summary):
                output.continuation.yield(.complete(.init(promptTokens: summary.promptTokenIDs.count,
                    completionTokens: summary.generatedTokenIDs.count, reasoningTokens: summary.reasoningTokens,
                    forcedTokens: summary.forcedThinkingEndTokens, finishReason: summary.stopReason == "endOfGeneration" ? "stop" : "length")))
                output.continuation.finish(); apiCompletedRequests += 1
                status = "API: klart · \(summary.generatedTokenIDs.count) token"
            case .failure(let failure):
                output.continuation.finish(throwing: failure); chatReady = false; chatControl = nil
                if !flag.isCancellationRequested { error = failure.localizedDescription }
                status = "API-anropet stoppades."
                // A disconnected client cancels its owner, not the listener.
                // Recreate only that context after awaiting native teardown;
                // reuse the same process-local, source-bound numerical check.
                if flag.isCancellationRequested, apiServer != nil, !isShuttingDown {
                    let recovery = LESControl(); control = recovery
                    do {
                        let check = try await worker.prepareChat(modelsRoot: modelsRoot, privateRoot: privateRoot, existing: nil,
                            control: recovery, progress: { _ in }, onCheck: { _ in })
                        guard check.nativeBenchmarkAdmitted, !recovery.isCancellationRequested else { throw CancellationError() }
                        numericalCheck = check; chatControl = recovery; chatReady = true; status = "API redo för nästa anrop."
                    } catch { await worker.closeChat(); status = "Aktivera snabbchatten igen för att återuppta API:t." }
                }
            }
        }
        return .init(events: output.stream, cancel: { lease.cancel() })
    }
    func clearChat() {
        guard canStart else { return }
        isBusy = true; chatMessages = []; chatSummary = nil; chatProgress = nil; chatMeasurementURL = nil
        operation = Task { await worker.resetChat(); isBusy = false; operation = nil }
    }
    func saveChatMeasurement() {
        guard canStart, let chatSummary, let numericalCheck else { return }
        struct Report: Encodable {
            let schemaVersion = 1
            let appBuild, sourceRevision, modelSHA256, nativeRevision: String
            let platform, operatingSystem: String
            let deviceIdentifier: String?
            let physicalMemoryBytes: UInt64
            let userDeclaredAirplaneMode: Bool
            let networkInferenceUsed = false
            let osPageCacheControlled = false
            let fullModelNumericsVerified = false
            let timestamp: Date
            let numericalCheck: LESNumericalCheck
            let generation: OrnithLESChatGeneration.Summary
        }
        do {
            #if targetEnvironment(simulator)
            let platform = "iOSSimulator"
            #elseif os(iOS)
            let platform = "iOSDevice"
            #else
            let platform = "macOS"
            #endif
            let report = Report(appBuild: Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "unknown",
                sourceRevision: BuildInfo.gitRevision, modelSHA256: GGUFModelManifest.primaryOrnith.sha256,
                nativeRevision: LESContext.revision, platform: platform,
                operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString, deviceIdentifier: Sysctl.string("hw.machine"),
                physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory, userDeclaredAirplaneMode: userDeclaredAirplaneMode,
                timestamp: Date(), numericalCheck: numericalCheck, generation: chatSummary)
            let folder = documents.appendingPathComponent("Diagnostics")
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            let url = folder.appendingPathComponent("ornith-chat-" + UUID().uuidString + ".json")
            let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]; encoder.dateEncodingStrategy = .iso8601
            try encoder.encode(report).write(to: url, options: .atomic); chatMeasurementURL = url
        } catch { self.error = error.localizedDescription }
    }
    private func save(check: LESNumericalCheck?, runs: [LESBenchmarkRun], airplane: Bool, failure: String? = nil) throws {
        #if targetEnvironment(simulator)
        let platform = "iOSSimulator"
        #elseif os(iOS)
        let platform = "iOSDevice"
        #else
        let platform = "macOS"
        #endif
        let record = LESAppReport(appVersion: Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "unknown",
            appBuild: Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "unknown",
            sourceRevision: BuildInfo.gitRevision, timestamp: Date(), platform: platform, deviceIdentifier: Sysctl.string("hw.machine"),
            physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory, operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString,
            lowPowerModeEnabled: ProcessInfo.processInfo.isLowPowerModeEnabled, userDeclaredAirplaneMode: airplane,
            numericalCheck: check, measurements: runs.map(LESRunExport.init), memoryAdmissions: memoryAdmissions, failure: failure,
            maximumObservedFootprintBytes: control?.maximumObservedFootprintBytes ?? 0)
        let folder = documents.appendingPathComponent("Diagnostics")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let url = folder.appendingPathComponent("ornith-les-" + UUID().uuidString + ".json")
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]; encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(record).write(to: url, options: .atomic); measurementURL = url
    }
}
#endif
