#if canImport(FlintEngine)
import FlintEngine
import Foundation
import MinirunKit
import Observation
import StorageCore

// Safety: only these booleans cross between MainActor, file verification and
// the Network queue. All accesses take the lock; no native pointer is shared.
private final class FlintAppGate: @unchecked Sendable {
    private let lock = NSLock()
    private var stopped = false, available = false
    func begin() { lock.lock(); stopped = false; lock.unlock() }
    func cancel() { lock.lock(); stopped = true; lock.unlock() }
    func cancelled() -> Bool { lock.lock(); defer { lock.unlock() }; return stopped }
    func setAvailable(_ value: Bool) { lock.lock(); available = value; lock.unlock() }
    func canServe() -> Bool { lock.lock(); defer { lock.unlock() }; return available }
}

private actor FlintFileStore {
    private var verified: VerifiedGGUFArtifact?
    func artifact(root: URL, gate: FlintAppGate) throws -> VerifiedGGUFArtifact {
        if let verified {
            do {
                try verified.assertUnchanged()
                return verified
            } catch {
                // The previously pinned file may have been replaced after a
                // failed download or Files import. Reverify the current bytes
                // rather than permanently pinning a stale artifact identity.
                self.verified = nil
            }
        }
        try LocalModelStorage.prepare(root: root)
        let result = try VerifiedGGUFArtifact.verify(insideRoot: root, manifest: FlintArtifact.manifest,
            cancellationRequested: { gate.cancelled() })
        verified = result; return result
    }
    func download(root: URL, gate: FlintAppGate,
                  progress: @escaping @Sendable (GGUFModelTransfer.Progress) -> Void) async throws {
        try LocalModelStorage.prepare(root: root)
        let source = try PinnedGGUFDownload(url: FlintArtifact.sourceURL, manifest: FlintArtifact.manifest)
        verified = try await GGUFModelTransfer().download(insideRoot: root, source: source,
            cancellationRequested: { gate.cancelled() }, onProgress: progress)
    }
}

struct FlintChatTurn: Codable, Identifiable {
    let id: UUID
    let prompt: String
    var answer = "", reasoning = ""
    var completed = false
}

/// App-owned lifetime, shared by chat/API. Creating/reopening a view does not
/// create a native context, re-download weights or repeat a benchmark.
@Observable @MainActor
final class FlintAppModel {
    var prompt = ""
    var systemPrompt = ""
    var contextProfile: LocalContextProfile = .standard
    // Experimental: additional shared expert bank slots, disabled by default.
    var expertCacheSlots = 0
    // Independent per-layer decode residency, opt-in; never changes top-k.
    var decodeCacheWindow = 0
    var options = LocalModelCapabilities(family: .flint).defaultOptions
    private(set) var history: [FlintChatTurn] = []
    private(set) var ready = false
    private(set) var isBusy = false
    private(set) var isShuttingDown = false
    private(set) var sceneIsActive = true
    private(set) var hasModel = false
    private(set) var status = "Ej laddad"
    private(set) var error: String?
    private(set) var fraction: Double?
    private(set) var loadingStartedAt: Date?
    private(set) var report: FlintGenerationReport?
    private(set) var reportURL: URL?
    private(set) var numericalReportURL: URL?
    private(set) var apiStatus = "Avstängd"
    private(set) var apiPromptProcessed = 0
    private(set) var apiPromptTotal = 0
    private(set) var apiGeneratedTokens = 0
    private(set) var apiRequestPhase = "Ingen aktiv förfrågan"
    private(set) var apiURLs: [String] = []
    private(set) var apiAccessKey = ""
    private(set) var apiListening = false
    var apiPort = 8080
    var apiKeyDraft = ""
    let modelsRoot: URL, privateRoot: URL, documents: URL
    @ObservationIgnored private let session = FlintSession()
    @ObservationIgnored private let files = FlintFileStore()
    @ObservationIgnored private let gate = FlintAppGate()
    @ObservationIgnored private var operation: Task<Void, Never>?
    @ObservationIgnored private var apiOperation: Task<Void, Never>?
    @ObservationIgnored private var shutdownOperation: Task<Void, Never>?
    @ObservationIgnored private var activeGeneration: LocalChatGeneration?
    @ObservationIgnored private var apiServer: LocalModelAPIServer?
    @ObservationIgnored private var apiID: UUID?
    @ObservationIgnored private var operationID: UUID?
    private var historyURL: URL { privateRoot.appendingPathComponent("chat-v1.json") }
    var canStart: Bool { sceneIsActive && !isBusy && !isShuttingDown }
    var canSend: Bool { canStart && ready }

    init(modelsRoot: URL, privateRoot: URL, documents: URL) {
        self.modelsRoot = modelsRoot; self.privateRoot = privateRoot; self.documents = documents
        refreshFile()
        if let size = try? historyURL.resourceValues(forKeys: [.fileSizeKey]).fileSize, size <= 16 << 20,
           let data = try? Data(contentsOf: historyURL), let saved = try? JSONDecoder().decode([FlintChatTurn].self, from: data), saved.count <= 128 {
            history = saved
        }
        let preferences = UserDefaults.standard
        if let value = preferences.string(forKey: "FlashLocal.Flint.System.v1"), value.utf8.count <= 32768 { systemPrompt = value }
        // Older builds offered 65K, which the current 5.5 GB phone budget cannot
        // admit. Never restore an unsupported picker value on launch.
        if let profile = LocalContextProfile(rawValue: preferences.integer(forKey: "FlashLocal.Flint.Context.v1")),
           [.standard, .test16K, .test32K].contains(profile) { contextProfile = profile }
        let savedSlots = preferences.integer(forKey: "FlashLocal.Flint.ExpertSlots.v1")
        if [0, 32, 64, 128, 256].contains(savedSlots) { expertCacheSlots = savedSlots }
        // Previous experiments could persist nonzero decode residency. The native
        // runtime now explicitly rejects it; keeping that preference would
        // make every subsequent model load fail with no visible way to reset.
        decodeCacheWindow = 0
        if let data = preferences.data(forKey: "FlashLocal.Flint.Options.v1"), let saved = try? JSONDecoder().decode(LocalGenerationOptions.self, from: data),
           (try? saved.validate()) != nil {
            options = saved; options.enableThinking = true; options.thinkingBudget = nil
        }
    }
    static func live() -> FlintAppModel {
        let fm = FileManager.default, docs = fm.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return .init(modelsRoot: docs.appendingPathComponent("Models/Flint", isDirectory: true),
            privateRoot: fm.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("FlashLocal/Flint", isDirectory: true), documents: docs)
    }
    func refreshFile() { hasModel = FileManager.default.fileExists(atPath: modelsRoot.appendingPathComponent(FlintArtifact.manifest.fileName).path) }
    func savePreferences() {
        guard systemPrompt.utf8.count <= 32768, (try? options.validate()) != nil else { return }
        UserDefaults.standard.set(systemPrompt, forKey: "FlashLocal.Flint.System.v1")
        UserDefaults.standard.set(contextProfile.rawValue, forKey: "FlashLocal.Flint.Context.v1")
        UserDefaults.standard.set(expertCacheSlots, forKey: "FlashLocal.Flint.ExpertSlots.v1")
        UserDefaults.standard.set(decodeCacheWindow, forKey: "FlashLocal.Flint.DecodeWindow.v1")
        UserDefaults.standard.set(try? JSONEncoder().encode(options), forKey: "FlashLocal.Flint.Options.v1")
    }
    private func saveHistory() {
        do {
            let data = try JSONEncoder().encode(history)
            guard data.count <= 16 << 20 else { throw LocalAPIError.invalid("messages", "Flint-historiken är för stor för säker lokal lagring. Tidigare sparad historik finns kvar.") }
            try LocalModelStorage.prepare(root: privateRoot); try data.write(to: historyURL, options: [.atomic])
        }
        catch { self.error = "Chatten kunde inte sparas: \(error.localizedDescription)" }
    }
    private func begin(_ text: String) -> UUID {
        let id = UUID(); operationID = id; gate.begin(); isBusy = true; error = nil; fraction = nil; status = text
        return id
    }
    private func end(_ id: UUID) {
        guard operationID == id else { return }
        operationID = nil; operation = nil; isBusy = false; fraction = nil; loadingStartedAt = nil
        gate.setAvailable(sceneIsActive && ready)
    }
    func download() {
        guard canStart, !hasModel else { return }
        let id = begin("Hämtar officiell Flint · 25,99 GB")
        operation = Task {
            defer { end(id); refreshFile() }
            do {
                try await files.download(root: modelsRoot, gate: gate) { [weak self] value in
                    Task { @MainActor in
                        guard let self, self.operationID == id else { return }
                        self.fraction = value.totalBytes > 0 ? Double(value.completedBytes) / Double(value.totalBytes) : nil
                        self.status = value.phase == .verifying ? "Verifierar hela modellfilen…" : "Hämtar Flint till intern lagring…"
                    }
                }
                status = "Modellfilen är hämtad och SHA256-verifierad."
            } catch { self.error = error.localizedDescription; status = "Hämtningen stoppades · redan skrivna delar kan återupptas." }
        }
    }
    func load() {
        guard canStart, !ready, hasModel else { return }
        savePreferences()
        let id = begin("Verifierar Flint-filen…"); loadingStartedAt = Date()
        let requestedContext = contextProfile
        let requestedExpertSlots = expertCacheSlots
        let requestedDecodeWindow = decodeCacheWindow
        operation = Task {
            defer { end(id) }
            do {
                let artifact = try await files.artifact(root: modelsRoot, gate: gate)
                var config = FlintConfiguration(); config.contextTokens = requestedContext.rawValue
                config.expertCacheSlots = requestedExpertSlots
                config.decodeCacheWindow = requestedDecodeWindow
                let check = try await session.prepare(artifact: artifact, privateRoot: privateRoot, configuration: config) { [weak self] value in
                    Task { @MainActor in
                        guard let self, self.operationID == id else { return }
                        self.fraction = value.fraction
                        self.status = [0: "Kontrollerar metadata och minnesbudget", 1: "Läser fasta vikter", 2: "Allokerar KV och arbetsbuffertar",
                            3: "Kontrollerar modellkontext", 4: "Jämför logitpositioner och expertval"][value.phase] ?? "Förbereder Flint"
                    }
                }
                guard let mode = check.servingMode, !gate.cancelled(), sceneIsActive else { await session.unload(); throw CancellationError() }
                ready = true
                switch mode {
                case .standard: status = "Redo · CPU/Metal-kontroll godkänd"
                case .singleToken: status = "Redo · verifierad tokenvis promptinläsning"
                case .singleTokenF32KV: status = "Redo · verifierat F32-KV-läge, tokenvis prompt"
                }
            } catch {
                ready = false; self.error = error.localizedDescription; status = "Flint kunde inte laddas."
                if let check = await session.lastNumericalCheck() {
                    do {
                        let directory = documents.appendingPathComponent("Diagnostics/Flint", isDirectory: true)
                        try LocalModelStorage.prepare(root: directory)
                        let url = directory.appendingPathComponent("flint-numerics-\(UUID().uuidString).json")
                        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                        try encoder.encode(check).write(to: url, options: [.atomic]); numericalReportURL = url
                    } catch { /* Keep the original load error visible. */ }
                }
            }
        }
    }
    func stop() { gate.cancel(); session.cancel(); activeGeneration?.cancel(); operation?.cancel() }
    func unloadModel() async {
        if let shutdownOperation { await shutdownOperation.value; return }
        isShuttingDown = true; ready = false; gate.setAvailable(false); stop()
        let work = operation, api = apiOperation, session = self.session
        let closing = Task { await work?.value; await api?.value; await session.unload() }
        shutdownOperation = closing; await closing.value; shutdownOperation = nil
        isShuttingDown = false; status = "Ej laddad · chatten finns kvar."
        if apiListening { apiStatus = "Servern lyssnar · modellen är urladdad" }
    }
    func shutdown() async { stopAPI(); await unloadModel() }
    func clearChat() {
        guard canStart else { return }
        history = []; saveHistory()
        guard ready else { return }
        // Keep the shared chat/LAN session unavailable until the native KV
        // reset has actually finished. Otherwise a new request can start with
        // the previous conversation's state while reset is still queued.
        let id = begin("Rensar samtalets modellminne…")
        operation = Task {
            defer { end(id) }
            do {
                try await session.resetConversation()
                status = "Chatten rensad · modellminnet nollställt."
            } catch {
                // Never continue serving from a potentially stale KV cache.
                ready = false
                await session.unload()
                self.error = "Modellminnet kunde inte rensas. Ladda om Flint: " + error.localizedDescription
                status = "Flint måste laddas om."
            }
        }
    }
    private func request() throws -> LocalChatRequest {
        guard history.count < 128, systemPrompt.utf8.count <= 32768, try JSONEncoder().encode(history).count < 12 << 20 else {
            throw LocalAPIError.invalid("messages", "Rensa chatten eller förkorta systemprompten")
        }
        var messages: [[String: Any]] = []
        if !systemPrompt.isEmpty { messages.append(["role": "system", "content": systemPrompt]) }
        // Exclude both halves of interrupted turns. No orphaned user messages,
        // empty assistant placeholders or hidden stale KV enter the next turn.
        for turn in history where turn.completed {
            messages.append(["role": "user", "content": turn.prompt])
            messages.append(["role": "assistant", "content": turn.answer, "reasoning_content": turn.reasoning])
        }
        messages.append(["role": "user", "content": prompt])
        return try LocalChatRequest(data: JSONSerialization.data(withJSONObject: ["model": LocalModelCapabilities(family: .flint).modelID,
            "messages": messages, "max_tokens": options.maximumNewTokens, "temperature": options.temperature, "top_p": options.topP, "seed": options.seed]),
            capabilities: .init(context: contextProfile, family: .flint))
    }
    func send() {
        guard canSend, !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        do {
            let request = try request(), turn = FlintChatTurn(id: UUID(), prompt: prompt)
            // Never expose a previous turn's metrics as this request's report.
            report = nil
            history.append(turn); prompt = ""; saveHistory(); savePreferences()
            let id = begin("Läser prompt…")
            operation = Task {
                defer { activeGeneration = nil; end(id); saveHistory() }
                do {
                    let source = try await session.start(request) { [weak self] done, total in
                        Task { @MainActor in guard let self, self.operationID == id else { return }; self.status = "Läser prompt: \(done)/\(total) token" }
                    }
                    activeGeneration = source
                    var completion = FlintEventCompletion()
                    for try await event in source.events {
                        try completion.observe(event)
                        guard let index = history.firstIndex(where: { $0.id == turn.id }) else { source.cancel(); throw CancellationError() }
                        switch event {
                        case .delta(let text, let thought):
                            history[index].answer += text; history[index].reasoning += thought
                            status = text.isEmpty && !thought.isEmpty ? "Tänker…" : "Genererar…"
                        case .complete:
                            break
                        }
                    }
                    try completion.requireSuccessfulEnd()
                    guard let completedReport = await session.lastReport() else {
                        throw FlintFailure.invalid("Flint saknar genereringsrapport för avslutat svar")
                    }
                    try Task.checkCancellation()
                    guard let index = history.firstIndex(where: { $0.id == turn.id }) else { throw CancellationError() }
                    report = completedReport
                    history[index].completed = true; status = "Klar"
                } catch { self.error = error.localizedDescription; status = "Svaret avbröts." }
            }
        } catch { self.error = error.localizedDescription }
    }
    func saveReport() {
        guard let report else { return }
        do {
            let directory = documents.appendingPathComponent("Diagnostics/Flint", isDirectory: true)
            try LocalModelStorage.prepare(root: directory)
            let url = directory.appendingPathComponent("flint-\(UUID().uuidString).json")
            let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]; encoder.dateEncodingStrategy = .iso8601
            try encoder.encode(report).write(to: url, options: [.atomic]); reportURL = url
        } catch { self.error = error.localizedDescription }
    }
    func benchmark(promptTokens: Int) {
        guard canSend, !apiListening else { return }
        // A previous benchmark report must never be mistaken for this run's
        // result if the new stream ends early without a completion event.
        report = nil
        let id = begin("Benchmark · förbereder \(promptTokens) verkliga prompttoken")
        operation = Task {
            defer { activeGeneration = nil; end(id) }
            do {
                let request = try await session.benchmarkRequest(promptTokens: promptTokens)
                let source = try await session.start(request) { [weak self] done, total in
                    Task { @MainActor in guard let self, self.operationID == id else { return }; self.status = "Benchmark · prompt \(done)/\(total)" }
                }
                activeGeneration = source
                var completion = FlintEventCompletion()
                for try await event in source.events {
                    try completion.observe(event)
                }
                // Do not publish measurements until the entire stream has
                // ended with exactly one terminal event. Late events or an
                // incomplete stream invalidate this run, not just its status.
                try completion.requireSuccessfulEnd()
                guard let completedReport = await session.lastReport() else {
                    throw FlintFailure.invalid("Benchmark saknar genereringsrapport · ingen giltig mätning")
                }
                try Task.checkCancellation()
                // A short or non-finite run must not populate the shared
                // report: the UI/export would otherwise present its token/s
                // as a comparable benchmark despite the warning in status.
                guard completedReport.isComparableDecodeBenchmark else {
                    status = "Ej jämförbar mätning · kräver exakt prompt och minst 128 token"
                    return
                }
                report = completedReport
                status = "Benchmark klar · minst 128 genererade token"
            } catch { self.error = error.localizedDescription; status = "Benchmark stoppades." }
        }
    }
    func setActive(_ active: Bool) {
        if active { refreshFile() }
        sceneIsActive = active; gate.setAvailable(active && ready)
        if !active { stop(); if apiListening { apiStatus = "Pausat · håll appen öppen för LAN-API" } }
        else { apiServer?.refreshState() }
    }
    func startAPI() {
        guard canSend, apiServer == nil, (1024...65535).contains(apiPort) else { return }
        do {
            savePreferences()
            let keys = LocalAPIKeyStore(service: "com.dhafen.flashlocal.flint-api.v1")
            if !apiKeyDraft.isEmpty { try keys.save(apiKeyDraft) }
            let key = try keys.loadOrCreate(), id = UUID(), gate = self.gate
            apiAccessKey = key; apiKeyDraft = ""; apiID = id; apiStatus = "Startar…"
            apiServer = try LocalModelAPIServer(key: key, port: UInt16(apiPort), defaults: options,
                capabilities: .init(context: contextProfile, family: .flint), canServe: { gate.canServe() },
                handler: { [weak self] request in
                    guard let self else { throw LocalAPIError(status: 503, message: "Phone owner is closed") }
                    return try await self.openAPI(request)
                }, onState: { [weak self] state in
                    Task { @MainActor in
                        guard let self, self.apiID == id else { return }
                        switch state {
                        case .ready(let port, let addresses):
                            self.apiListening = true; self.apiURLs = addresses.map { "http://\($0):\(port)/v1" }
                            self.apiStatus = self.ready && self.sceneIsActive ?
                                (addresses.isEmpty ? "Servern lyssnar · anslut till Wi-Fi eller Ethernet" : "API aktivt · samma Flint-session") : "Pausat · ladda modellen och håll appen öppen"
                        case .failed(let message): self.error = message; self.stopAPI()
                        case .stopped: self.stopAPI()
                        }
                    }
                })
        } catch { self.error = error.localizedDescription; stopAPI() }
    }
    private func openAPI(_ request: LocalChatRequest) async throws -> LocalChatGeneration {
        guard canSend else { throw LocalAPIError(status: isBusy ? 429 : 503, message: "Flint is busy, unloaded or the phone app is not active") }
        let id = begin("API · läser prompt…")
        apiPromptProcessed = 0; apiPromptTotal = 0; apiGeneratedTokens = 0
        report = nil
        apiRequestPhase = "Tar emot och tokeniserar prompt…"
        let source: LocalChatGeneration
        do {
            source = try await session.start(request, progress: { [weak self] processed, total in
                Task { @MainActor [weak self] in
                    guard let self, self.operationID == id else { return }
                    self.apiPromptProcessed = processed
                    self.apiPromptTotal = total
                    self.apiRequestPhase = "Läser prompt · \(processed)/\(total) token"
                }
            })
        } catch {
            apiRequestPhase = "Förfrågan misslyckades"
            end(id); throw error
        }
        activeGeneration = source
        let pair = AsyncThrowingStream<LocalChatEvent, Error>.makeStream(bufferingPolicy: .bufferingOldest(512))
        pair.continuation.onTermination = { _ in source.cancel() }
        apiOperation = Task {
            defer { activeGeneration = nil; apiOperation = nil; end(id) }
            do {
                var completion = FlintEventCompletion()
                var terminal: LocalChatEvent?
                for try await event in source.events {
                    try completion.observe(event)
                    switch event {
                    case .delta:
                        apiRequestPhase = "Genererar svar…"
                        switch pair.continuation.yield(event) {
                        case .enqueued: break
                        case .dropped, .terminated: source.cancel(); throw CancellationError()
                        @unknown default: source.cancel(); throw CancellationError()
                        }
                    case .complete:
                        // Do not expose success to the LAN client until the
                        // upstream stream has really ended. A later error or
                        // duplicate completion must fail closed.
                        terminal = event
                    }
                }
                try completion.requireSuccessfulEnd()
                guard let completedReport = await session.lastReport() else {
                    throw FlintFailure.invalid("Flint saknar genereringsrapport för API-svaret")
                }
                try Task.checkCancellation()
                guard let terminal else { throw FlintEventCompletion.Failure.missingCompletion }
                switch pair.continuation.yield(terminal) {
                case .enqueued: break
                case .dropped, .terminated: source.cancel(); throw CancellationError()
                @unknown default: source.cancel(); throw CancellationError()
                }
                report = completedReport
                apiGeneratedTokens = completedReport.generatedTokenIDs.count
                apiRequestPhase = "Klar · \(apiGeneratedTokens) genererade token"
                status = "API · klar"; pair.continuation.finish()
            } catch { self.error = error.localizedDescription; pair.continuation.finish(throwing: error) }
        }
        return .init(events: pair.stream, cancel: source.cancel)
    }
    func stopAPI() {
        apiID = nil; apiServer?.stop(); apiServer = nil; apiListening = false; apiStatus = "Avstängd"; apiURLs = []; apiAccessKey = ""
        apiPromptProcessed = 0; apiPromptTotal = 0; apiGeneratedTokens = 0
        apiRequestPhase = "Ingen aktiv förfrågan"
        if apiOperation != nil { activeGeneration?.cancel(); apiOperation?.cancel() }
    }
}
#endif
