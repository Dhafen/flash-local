import Foundation
import MinirunKit
import MinirunRunners
import ModelAdapters
import Observation
import StorageCore

struct REAMMessage: Codable, Identifiable, Sendable {
    let id: UUID
    let role: Qwen35ChatMessage.Role
    var text: String
    var interrupted: Bool
    var reasoning: String? = nil
    var isThinking: Bool? = nil
    init(role: Qwen35ChatMessage.Role, text: String, interrupted: Bool = false,
         reasoning: String? = nil, isThinking: Bool? = nil) {
        id = UUID(); self.role = role; self.text = text; self.interrupted = interrupted
        self.reasoning = reasoning; self.isThinking = isThinking
    }

    /// Keep visible failed attempts, but omit an unanswered turn as a pair
    /// from the model context. Removing only its empty assistant would leave
    /// consecutive user roles, which the canonical chat template rejects.
    static func generationPrompt(_ history: [REAMMessage]) -> [Qwen35ChatMessage] {
        var prompt: [Qwen35ChatMessage] = []
        for message in history {
            if message.role == .assistant && message.interrupted && message.text.isEmpty && (message.reasoning ?? "").isEmpty {
                if prompt.last?.role == .user { prompt.removeLast() }
                continue
            }
            // Recover old persisted histories that already contain an
            // unanswered user: the next send supersedes that pending turn.
            if message.role == .user && prompt.last?.role == .user { prompt.removeLast() }
            prompt.append(.init(role: message.role, content: message.text,
                reasoningContent: message.role == .assistant ? message.reasoning : nil))
        }
        return prompt
    }
}

private final class REAMCancellation: @unchecked Sendable {
    private let lock = NSLock(); private var cancelled = false
    func cancel() { lock.lock(); cancelled = true; lock.unlock() }
    func requested() -> Bool { lock.lock(); defer { lock.unlock() }; return cancelled }
}
private enum REAMPresentationError: Error, LocalizedError {
    case streamFull
    var errorDescription: String? { "Gränssnittet hann inte ta emot texten. Genereringen stoppades." }
}

/// Immutable run policy, shared by preflight and execution. Test fixtures use
/// the same engine with a shorter run; production always starts conservatively.
struct REAMRunSettings: Sendable {
    let contextTokens: Int
    let budgetBytes: UInt64
    let maximumNewTokens: Int
    let rawCacheBytes: Int
    let residentCacheBytes: Int
    init(contextTokens: Int, budgetBytes: UInt64, maximumNewTokens: Int, rawCacheBytes: Int,
         residentCacheBytes: Int = 0) {
        self.contextTokens = contextTokens; self.budgetBytes = budgetBytes
        self.maximumNewTokens = maximumNewTokens; self.rawCacheBytes = rawCacheBytes
        self.residentCacheBytes = residentCacheBytes
    }
    static let initial = REAMRunSettings(contextTokens: 1024, budgetBytes: 5_500_000_000,
        maximumNewTokens: 128, rawCacheBytes: 64 << 20, residentCacheBytes: 4_250_000_000)
    static let ornith = REAMRunSettings(contextTokens: 4096, budgetBytes: 5_500_000_000,
        maximumNewTokens: 512, rawCacheBytes: 64 << 20, residentCacheBytes: 3_500_000_000)
}

enum LocalChatModel: String, CaseIterable, Identifiable {
    case ornith, ream
    var id: String { rawValue }
    var title: String {
        switch self { case .ornith: "Ornith 1.5 · 35B-A3B"; case .ream: "Qwen3.8 Flash Next · REAM" }
    }
    var source: PinnedGGUFDownload { self == .ornith ? .primaryOrnith : .primaryREAM }
    var settings: REAMRunSettings { self == .ornith ? .ornith : .initial }
    var chatName: String { self == .ornith ? "ornith-chat.json" : "chat.json" }
}

/// One off-main owner for the tokenizer, weights and recurrent/KV prefix. An
/// actor prevents two sessions or an eviction from running MLX concurrently.
private actor REAMInferenceSession {
    private var engine: Qwen4ExpTextGeneration?
    private var owner: VerifiedGGUFArtifact?

    func release() { engine = nil; owner = nil }
    func clearConversation() {
        do { try engine?.resetConversation() } catch { release() }
    }

    func generate(artifact: VerifiedGGUFArtifact, settings: REAMRunSettings, thinkingEnabled: Bool,
                  messages: [Qwen35ChatMessage], cancellation: REAMCancellation,
                  events: AsyncThrowingStream<Qwen4ExpTextGeneration.Event, Error>.Continuation)
        -> Result<Qwen4ExpTextGeneration.Summary, Error> {
        let result = Result {
            try autoreleasepool {
                if owner !== artifact {
                    release() // release the old allocation BEFORE creating another engine
                    engine = try Qwen4ExpTextGeneration(artifact: artifact, contextTokens: settings.contextTokens,
                        budgetBytes: settings.budgetBytes, rawCacheBytes: settings.rawCacheBytes,
                        residentCacheBytes: settings.residentCacheBytes)
                    owner = artifact
                }
                var lastProgress = 0.0
                return try engine!.generate(messages: messages, maximumNewTokens: settings.maximumNewTokens,
                    reusePromptPrefix: true, enableThinking: thinkingEnabled,
                    cancellationRequested: { cancellation.requested() || Task.isCancelled },
                    captureCancelledSummary: true) { event in
                        if case .prefill(let p) = event {
                            let now = ProcessInfo.processInfo.systemUptime
                            guard now - lastProgress >= 0.2 || p.promptTokensCompleted == p.promptTokensTotal else { return }
                            lastProgress = now
                        }
                        if case .dropped = events.yield(event) {
                            if case .prefill = event { return }
                            throw REAMPresentationError.streamFull
                        }
                    }
            }
        }
        if case .failure = result { release() }
        return result
    }
}

/// Saved only by an explicit action. No report is uploaded. Token IDs preserve
/// the exact test input/output for later local reference comparisons.
struct REAMRunReport: Codable, Sendable {
    let schemaVersion: Int
    let timestamp: String
    let sourceRevision: String
    let appVersion: String
    let appBuild: String
    let platform: String
    let deviceIdentifier: String?
    let operatingSystem: String
    let physicalMemoryBytes: UInt64
    let modelSourceURL: String
    let modelFileBytes: UInt64
    let modelProvenance: GGUFModelProvenance?
    let contextTokens: Int
    let budgetBytes: UInt64
    let maximumNewTokens: Int
    let sampling: String
    let airplaneModeVerified: Bool
    let fullModelNumericsVerified: Bool
    let decodeForwardTokensPerSecond: Double?
    let generation: Qwen4ExpTextGeneration.Summary

    @MainActor init(summary: Qwen4ExpTextGeneration.Summary, settings: REAMRunSettings, source: PinnedGGUFDownload) {
        schemaVersion = 1; sampling = "greedy"
        airplaneModeVerified = false; fullModelNumericsVerified = false
        timestamp = ISO8601DateFormatter().string(from: Date())
        sourceRevision = BuildInfo.gitRevision
        appVersion = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "unknown"
        appBuild = Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "unknown"
        #if targetEnvironment(simulator)
        platform = "iOSSimulator"
        #elseif os(iOS)
        platform = "iOSDevice"
        #else
        platform = "macOS"
        #endif
        #if os(iOS)
        deviceIdentifier = Sysctl.string("hw.machine")
        #else
        deviceIdentifier = Sysctl.string("hw.model")
        #endif
        operatingSystem = ProcessInfo.processInfo.operatingSystemVersionString
        physicalMemoryBytes = ProcessInfo.processInfo.physicalMemory
        modelSourceURL = source.url.absoluteString; modelFileBytes = source.manifest.fileByteCount
        modelProvenance = source.manifest.provenance
        contextTokens = settings.contextTokens; budgetBytes = settings.budgetBytes
        maximumNewTokens = settings.maximumNewTokens
        decodeForwardTokensPerSecond = summary.decodeTokensPerSecond; generation = summary
    }
}

/// Main-actor presentation state; hashing, transfer and MLX execution live on
/// workers. A Stop waits for actual worker teardown before allowing another
/// operation. A verified descriptor is the only authority that enables Send.
@Observable @MainActor
final class REAMAppModel {
    #if canImport(OrnithLES)
    // Owned by the process-level model, not by a sheet or tab identity.
    @ObservationIgnored lazy var ornithSession = OrnithLESAppModel.live()
    #endif
    #if canImport(FlintEngine)
    @ObservationIgnored lazy var flintSession = FlintAppModel.live()
    var showsFlint = false
    #endif
    // Switching models is a shell concern, independent of optional engine packages.
    var isSwitchingInferenceModel = false
    @ObservationIgnored private var apiServer: LocalModelAPIServer?
    @ObservationIgnored private var apiRequestID: UUID?
    @ObservationIgnored private var apiCancellation: REAMCancellation?
    var apiPort = 8080 {
        didSet {
            if (1024...65535).contains(apiPort), modelSelectionDirectory != nil {
                UserDefaults.standard.set(apiPort, forKey: "FlashLocal.API.Port.v1")
            }
        }
    }
    var apiKeyDraft = ""
    private(set) var apiAccessKey = ""
    private(set) var apiURLs: [String] = []
    private(set) var apiStatus = "Avstängd"
    private(set) var apiCompletedRequests = 0
    var isAPIListening: Bool { apiServer != nil }
    var canStartAPI: Bool { canChat && selectedModel == .ornith && !isPreview && apiServer == nil }
    enum Activity { case idle, downloading, verifying, generating, checkingNumerics, resetting, stopping }
    private(set) var activity = Activity.idle
    private(set) var messages: [REAMMessage] = []
    private(set) var status = "Modellen är inte hämtad."
    private(set) var error: String?
    private(set) var hasFile = false
    private(set) var partialBytes: UInt64 = 0
    private(set) var progress: GGUFModelTransfer.Progress?
    private(set) var generationProgress: Qwen4ExpTextGeneration.Progress?
    private(set) var summary: Qwen4ExpTextGeneration.Summary?
    private(set) var lastMeasurementURL: URL?
    private(set) var lastNumericsURL: URL?
    private(set) var lastNumericsReport: Qwen4ExpLocalParity.Report?
    private(set) var isPreview = false
    private(set) var alternateRunnerActive = false
    private(set) var thinkingEnabled = false
    /// Drafting is a text-editing operation, never an inference admission.
    /// It must remain available while a 21 GB model is being verified.
    var draft = "" {
        didSet {
            if modelSelectionDirectory != nil, draft.utf8.count <= 32_768 {
                UserDefaults.standard.set(draft, forKey: "FlashLocal.ChatDraft.\(selectedModel.rawValue)")
            }
        }
    }
    let root: URL
    private(set) var source: PinnedGGUFDownload
    private(set) var settings: REAMRunSettings
    var contextTokens: Int { settings.contextTokens }
    var budgetBytes: UInt64 { settings.budgetBytes }
    var maximumNewTokens: Int { settings.maximumNewTokens }
    @ObservationIgnored private var artifact: VerifiedGGUFArtifact?
    @ObservationIgnored private let inferenceSession = REAMInferenceSession()
    @ObservationIgnored private var operation: Task<Void, Never>?
    @ObservationIgnored private var cancelWorker: (() -> Void)?
    @ObservationIgnored private var cancellation: REAMCancellation?
    @ObservationIgnored private var activeID: UUID?
    @ObservationIgnored private var prepared = false
    @ObservationIgnored private var chatFile: URL?
    @ObservationIgnored private let modelSelectionDirectory: URL?
    var isBusy: Bool { activity != .idle }
    var canRunNumerics: Bool { artifact != nil && !isBusy && !isPreview && !alternateRunnerActive }
    var canChat: Bool { canRunNumerics && lastNumericsReport?.passedAtTestedPositions != false }
    var modelSize: String { ByteCountFormatter.string(fromByteCount: Int64(source.manifest.fileByteCount), countStyle: .file) }
    var selectedModel: LocalChatModel {
        source.manifest.sha256 == GGUFModelManifest.primaryREAM.sha256 ? .ream : .ornith
    }
    var modelTitle: String {
        if source.manifest.sha256 == GGUFModelManifest.primaryOrnith.sha256 { return LocalChatModel.ornith.title }
        if source.manifest.sha256 == GGUFModelManifest.primaryREAM.sha256 { return LocalChatModel.ream.title }
        return "Lokal modell"
    }
    var canSelectModel: Bool { modelSelectionDirectory != nil && !isPreview && !isBusy && !alternateRunnerActive }
    var canOpenAlternateRunner: Bool { !isBusy && !isPreview && !alternateRunnerActive }
    var supportsThinking: Bool {
        source.manifest.sha256 == GGUFModelManifest.primaryOrnith.sha256 ||
            artifact?.index.metadata["general.architecture"]?.text == "qwen35moe"
    }
    func setThinkingEnabled(_ enabled: Bool) {
        guard supportsThinking, !isBusy, !isPreview, !alternateRunnerActive else { return }
        thinkingEnabled = enabled
        if modelSelectionDirectory != nil {
            UserDefaults.standard.set(enabled, forKey: "FlashLocal.ornithThinking")
        }
    }
    func isGenerating(_ message: REAMMessage) -> Bool {
        message.id == messages.last?.id && message.role == .assistant &&
            (activity == .generating || activity == .stopping)
    }

    func pauseForAlternateRunner() async {
        stopAPI()
        alternateRunnerActive = true
        stop()
        await operation?.value
        await inferenceSession.release()
    }
    func resumeAfterAlternateRunner() {
        alternateRunnerActive = false
        refreshStorage()
    }

    /// Activate the legacy GGUF runner only when its page is actually shown.
    /// Do not start verification of the previously selected model first: that
    /// would race with selectModel and could prevent a requested model switch.
    func activateForChatModel(_ target: LocalChatModel) {
        alternateRunnerActive = false
        if selectedModel != target {
            selectModel(target)
        } else if !prepared {
            prepare()
        } else {
            refreshStorage()
        }
    }
    #if canImport(OrnithLES)
    func verifiedOrnithForAlternateRunner() -> VerifiedGGUFArtifact? {
        guard alternateRunnerActive, !isBusy, let artifact,
              artifact.manifest.sha256 == GGUFModelManifest.primaryOrnith.sha256 else { return nil }
        do { try artifact.assertUnchanged(); return artifact } catch { return nil }
    }
    #endif

    init(root: URL, source: PinnedGGUFDownload = .primaryOrnith, chatFile: URL? = nil,
         settings: REAMRunSettings = .ornith, modelSelectionDirectory: URL? = nil) {
        self.root = root; self.source = source; self.chatFile = chatFile; self.settings = settings
        self.modelSelectionDirectory = modelSelectionDirectory
        if source.manifest.sha256 == GGUFModelManifest.primaryOrnith.sha256 {
            thinkingEnabled = modelSelectionDirectory == nil ? true :
                UserDefaults.standard.object(forKey: "FlashLocal.ornithThinking") as? Bool ?? true
        }
    }
    static func live() -> REAMAppModel {
        let manager = FileManager.default
        let documents = manager.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let support = manager.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        let directory = support.appendingPathComponent("FlashLocal", isDirectory: true)
        let selected = UserDefaults.standard.string(forKey: "FlashLocal.selectedChatModel")
            .flatMap(LocalChatModel.init(rawValue:)) ?? .ornith
        let model = REAMAppModel(root: documents.appendingPathComponent("Models", isDirectory: true), source: selected.source,
            chatFile: directory.appendingPathComponent(selected.chatName), settings: selected.settings,
            modelSelectionDirectory: directory)
        model.draft = UserDefaults.standard.string(forKey: "FlashLocal.ChatDraft.\(selected.rawValue)") ?? ""
        let savedPort = UserDefaults.standard.integer(forKey: "FlashLocal.API.Port.v1")
        if (1024...65535).contains(savedPort) { model.apiPort = savedPort }
        return model
    }
    func selectModel(_ target: LocalChatModel) {
        guard canSelectModel, target != selectedModel, let directory = modelSelectionDirectory else { return }
        stopAPI()
        let (id, flag) = begin(.resetting, status: "Byter modell…")
        let session = inferenceSession
        operation = Task {
            await session.release()
            guard activeID == id else { return }
            if flag.requested() { status = "Modellbytet avbröts."; finish(id); return }
            artifact = nil; source = target.source; settings = target.settings
            thinkingEnabled = target == .ornith &&
                (UserDefaults.standard.object(forKey: "FlashLocal.ornithThinking") as? Bool ?? true)
            chatFile = directory.appendingPathComponent(target.chatName)
            messages.removeAll()
            draft = UserDefaults.standard.string(forKey: "FlashLocal.ChatDraft.\(target.rawValue)") ?? ""
            summary = nil; generationProgress = nil
            lastMeasurementURL = nil; lastNumericsURL = nil; lastNumericsReport = nil
            UserDefaults.standard.set(target.rawValue, forKey: "FlashLocal.selectedChatModel")
            prepared = false; status = "Modellen är inte hämtad."
            finish(id); prepare()
        }
    }
    func prepare() {
        guard !prepared, !isPreview else { return }; prepared = true
        do {
            try LocalModelStorage.prepare(root: root)
            if let chatFile, FileManager.default.fileExists(atPath: chatFile.path) {
                guard try FileDigestComputer.byteCount(ofFileAt: chatFile) <= 4 << 20 else {
                    throw CocoaError(.fileReadTooLarge)
                }
                let restored = try JSONDecoder().decode([REAMMessage].self, from: Data(contentsOf: chatFile))
                guard restored.count <= 4096 else { throw CocoaError(.fileReadTooLarge) }
                messages = restored.map { message in
                    var value = message
                    if value.isThinking == true { value.isThinking = false; value.interrupted = true }
                    return value
                }
            }
            scan(); if hasFile { verify() }
        } catch { self.error = error.localizedDescription }
    }
    private func scan() {
        hasFile = FileManager.default.fileExists(atPath: root.appendingPathComponent(source.manifest.fileName).path)
        partialBytes = (try? FileDigestComputer.byteCount(ofFileAt: root.appendingPathComponent(source.manifest.fileName + ".partial"))) ?? 0
    }
    func refreshStorage() {
        guard prepared, !isBusy, !isPreview else { return }
        scan()
        if !hasFile { artifact = nil }
        if let artifact { do { try artifact.assertUnchanged() } catch { self.artifact = nil } }
        if artifact == nil { releaseInferenceResources() }
        if hasFile && artifact == nil { verify() }
    }
    private func begin(_ activity: Activity, status: String) -> (UUID, REAMCancellation) {
        let id = UUID(), flag = REAMCancellation()
        self.activity = activity; self.status = status; error = nil; progress = nil
        activeID = id; cancellation = flag; return (id, flag)
    }
    private func finish(_ id: UUID) {
        guard activeID == id else { return }
        activeID = nil; activity = .idle; operation = nil; cancelWorker = nil; cancellation = nil; scan()
        if !hasFile { artifact = nil }
        if let artifact { do { try artifact.assertUnchanged() } catch { self.artifact = nil } }
    }
    func stop() {
        guard isBusy else { return }
        activity = .stopping; status = "Stoppar arbetet…"
        cancellation?.cancel(); cancelWorker?()
    }
    func releaseInferenceResources() {
        stopAPI()
        stop()
        let session = inferenceSession
        Task { await session.release() }
    }
    nonisolated private static func preflight(_ artifact: VerifiedGGUFArtifact, settings: REAMRunSettings) throws -> VerifiedGGUFArtifact {
        try autoreleasepool {
            let vocabularySize = try GGUFStreamedDecoder.vocabularySize(index: artifact.index)
            let tokenizer = try CanonicalQwen35Tokenizer(artifact: artifact)
            guard vocabularySize == tokenizer.vocabularySize else { throw CocoaError(.fileReadCorruptFile) }
            _ = try GGUFStreamedDecoder.memoryPlan(index: artifact.index, contextTokens: settings.contextTokens,
                budgetBytes: settings.budgetBytes)
            try artifact.assertUnchanged(); return artifact
        }
    }
    func verify() {
        guard !isBusy, !isPreview else { return }
        artifact = nil
        let (id, flag) = begin(.verifying, status: "Verifierar hela modellfilens SHA-256…")
        let root = root, manifest = source.manifest, settings = settings, session = inferenceSession
        let worker = Task.detached(priority: .utility) {
            await session.release()
            return try Self.preflight(VerifiedGGUFArtifact.verify(insideRoot: root, manifest: manifest,
                cancellationRequested: { flag.requested() }), settings: settings)
        }
        cancelWorker = { worker.cancel() }
        operation = Task {
            do { artifact = try await worker.value; status = "Modellfil och tokenizer är verifierade." }
            catch { if !flag.requested() { self.error = error.localizedDescription }; status = "Verifieringen slutfördes inte." }
            finish(id)
        }
    }
    func download() {
        guard !isBusy, !isPreview else { return }
        let (id, flag) = begin(.downloading, status: "Hämtar modellen till intern lagring…")
        let root = root, source = source, settings = settings, session = inferenceSession
        let channel = AsyncStream<GGUFModelTransfer.Progress>.makeStream(bufferingPolicy: .bufferingNewest(1))
        let worker = Task.detached(priority: .utility) { () -> Result<VerifiedGGUFArtifact, Error> in
            defer { channel.continuation.finish() }
            await session.release()
            do {
                let value = try await GGUFModelTransfer().download(insideRoot: root, source: source,
                    cancellationRequested: { flag.requested() }, onProgress: { _ = channel.continuation.yield($0) })
                return .success(try Self.preflight(value, settings: settings))
            } catch { return .failure(error) }
        }
        cancelWorker = { worker.cancel() }
        operation = Task {
            for await value in channel.stream {
                progress = value
                if activity != .stopping {
                    switch value.phase {
                    case .checkingResume: activity = .downloading; status = "Kontrollerar tidigare hämtade delar…"
                    case .downloading: activity = .downloading; status = "Hämtar till intern lagring…"
                    case .verifying, .ready: activity = .verifying; status = "Kontrollerar modellfil och tokenizer…"
                    }
                }
            }
            switch await worker.value {
            case .success(let value): artifact = value; status = "Modellfil och tokenizer är verifierade."
            case .failure(let failure):
                if !flag.requested() { error = failure.localizedDescription }
                status = "Hämtningen är pausad. Du kan återuppta den."
            }
            finish(id)
        }
    }
    func discardPartial() {
        guard !isBusy, !isPreview else { return }
        let (id, _) = begin(.resetting, status: "Rensar påbörjad hämtning…")
        let root = root, manifest = source.manifest
        let worker = Task.detached(priority: .utility) { try GGUFModelTransfer.discardPartial(insideRoot: root, manifest: manifest) }
        operation = Task {
            do { try await worker.value; status = "Den påbörjade hämtningen har rensats." }
            catch { self.error = error.localizedDescription }
            finish(id)
        }
    }
    func send() {
        guard canChat, let artifact else { return }
        let input = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !input.isEmpty, input.utf8.count <= 262144 else { return }
        draft = ""; messages.append(.init(role: .user, text: input))
        let prompt = REAMMessage.generationPrompt(messages)
        messages.append(.init(role: .assistant, text: "", interrupted: true))
        let responseID = messages.last!.id
        let (id, flag) = begin(.generating, status: "Förbereder lokal generering…")
        generationProgress = nil; summary = nil; lastMeasurementURL = nil; save()
        let session = inferenceSession, settings = settings, thinking = thinkingEnabled && supportsThinking
        let channel = AsyncThrowingStream<Qwen4ExpTextGeneration.Event, Error>.makeStream(bufferingPolicy: .bufferingOldest(32))
        channel.continuation.onTermination = { if case .cancelled = $0 { flag.cancel() } }
        let worker = Task.detached(priority: .userInitiated) { () -> Result<Qwen4ExpTextGeneration.Summary, Error> in
            let result = await session.generate(artifact: artifact, settings: settings, thinkingEnabled: thinking, messages: prompt,
                cancellation: flag, events: channel.continuation)
            switch result { case .success: channel.continuation.finish(); case .failure(let error): channel.continuation.finish(throwing: error) }
            return result
        }
        cancelWorker = { worker.cancel() }
        operation = Task {
            var output = QwenThinkingText(startsInThinking: thinking)
            do {
                for try await event in channel.stream {
                    guard let index = messages.firstIndex(where: { $0.id == responseID }) else { continue }
                    switch event {
                    case .prefill(let p):
                        generationProgress = p
                        if activity != .stopping {
                            status = "Läser prompt: \(p.promptTokensCompleted)/\(p.promptTokensTotal) token"
                                + (p.reusedPromptTokens > 0 ? " · \(p.reusedPromptTokens) återanvända" : "")
                        }
                    case .token(_, let text, let p):
                        generationProgress = p; output.append(text)
                        messages[index].text = output.answer
                        messages[index].reasoning = output.reasoning.isEmpty ? nil : output.reasoning
                        messages[index].isThinking = output.isThinking
                        if activity != .stopping { status = output.isThinking ? "Tänker lokalt…" : "Skriver svaret lokalt…" }
                    case .trailingText(let text): output.append(text)
                    }
                }
            } catch { /* Worker result below is authoritative, after teardown. */ }
            let result = await worker.value
            output.finish()
            if let index = messages.firstIndex(where: { $0.id == responseID }) {
                messages[index].text = output.answer
                messages[index].reasoning = output.reasoning.isEmpty ? nil : output.reasoning
                messages[index].isThinking = false
            }
            switch result {
            case .success(let value):
                summary = value
                if let index = messages.firstIndex(where: { $0.id == responseID }) {
                    messages[index].interrupted = value.stopReason == .cancelled || output.isThinking
                }
                if value.stopReason == .cancelled { status = "Stoppad · mätningen kan sparas i Filer." }
                else if output.isThinking {
                    status = value.stopReason == .tokenLimit ? "Tokengränsen nåddes under tänkandet · inget färdigt svar."
                        : "Modellen avslutade under tänkandet · inget färdigt svar."
                } else { status = "Klar · \(value.generatedTokenIDs.count) genererade token" }
            case .failure(let failure):
                if !flag.requested() { error = failure.localizedDescription }
                status = "Genereringen avbröts."
            }
            save(); finish(id)
        }
    }
    func clearMessages() {
        guard !isBusy else { return }
        let (id, _) = begin(.resetting, status: "Rensar samtalets modellminne…")
        messages.removeAll(); summary = nil; generationProgress = nil; lastMeasurementURL = nil; save()
        let session = inferenceSession
        operation = Task { await session.clearConversation(); finish(id) }
    }
    func saveMeasurement() {
        guard !isBusy, !isPreview, let summary else { return }
        let report = REAMRunReport(summary: summary, settings: settings, source: source)
        let directory = root.deletingLastPathComponent().appendingPathComponent("Diagnostics", isDirectory: true)
        let (id, _) = begin(.resetting, status: "Sparar mätningen lokalt…")
        let worker = Task.detached(priority: .utility) {
            try LocalModelStorage.prepare(root: directory)
            let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(report)
            guard data.count <= 4 << 20 else { throw CocoaError(.fileWriteOutOfSpace) }
            let file = directory.appendingPathComponent("run-\(UUID().uuidString).json")
            try data.write(to: file, options: .atomic)
            return file
        }
        operation = Task {
            do {
                lastMeasurementURL = try await worker.value
                status = "Mätningen sparades i Flash Local › Diagnostics i Filer."
            } catch { self.error = "Mätningen kunde inte sparas: " + error.localizedDescription }
            finish(id)
        }
    }
    /// Explicit, short local comparison using a public fixed prompt. The CPU
    /// model is released before MLX starts; this never contacts a reference API.
    func checkNumerics() {
        guard canRunNumerics, let artifact else { return }
        let (id, flag) = begin(.checkingNumerics, status: "Startar kort lokal numerikjämförelse…")
        // A cancelled retry must not erase a previously observed mismatch.
        let budget = budgetBytes, cache = settings.rawCacheBytes, session = inferenceSession
        let directory = root.deletingLastPathComponent().appendingPathComponent("Diagnostics", isDirectory: true)
        typealias Update = (Qwen4ExpLocalParity.Phase, Int, Int)
        let channel = AsyncStream<Update>.makeStream(bufferingPolicy: .bufferingNewest(1))
        let worker = Task.detached(priority: .userInitiated) { () -> Result<(Qwen4ExpLocalParity.Report, URL), Error> in
            defer { channel.continuation.finish() }
            await session.release() // the CPU diagnostic must never coexist with retained MLX weights
            return Result {
                let report = try Qwen4ExpLocalParity.run(artifact: artifact,
                    messages: [.init(role: .user, content: "Hi")], budgetBytes: budget, rawCacheBytes: cache,
                    cancellationRequested: { flag.requested() },
                    onProgress: { _ = channel.continuation.yield(($0, $1, $2)) })
                if flag.requested() { throw CancellationError() }
                try LocalModelStorage.prepare(root: directory)
                let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                let data = try encoder.encode(report)
                guard data.count <= 4 << 20 else { throw CocoaError(.fileWriteOutOfSpace) }
                let file = directory.appendingPathComponent("parity-\(UUID().uuidString).json")
                try data.write(to: file, options: .atomic)
                return (report, file)
            }
        }
        cancelWorker = { worker.cancel() }
        operation = Task {
            for await (phase, complete, total) in channel.stream where activity != .stopping {
                status = "Numeriktest · \(phase == .cpuReference ? "CPU-referens" : "streamad MLX") · \(complete)/\(total)"
            }
            switch await worker.value {
            case .success(let value):
                lastNumericsReport = value.0; lastNumericsURL = value.1
                status = value.0.passedAtTestedPositions
                    ? "Kort numeriktest inom toleranserna vid \(value.0.positions.count) positioner. Rapport i Diagnostics."
                    : "Numerikavvikelse upptäckt. Se rapporten i Diagnostics."
                if !value.0.passedAtTestedPositions {
                    error = "Generering är spärrad efter numerikavvikelsen. En ny jämförelse måste klara testet."
                }
            case .failure(let failure):
                if !flag.requested() { error = failure.localizedDescription }
                status = "Numeriktestet slutfördes inte."
            }
            finish(id)
        }
    }
    private func save() {
        guard let chatFile else { return }
        do {
            try LocalModelStorage.prepare(root: chatFile.deletingLastPathComponent())
            let data = try JSONEncoder().encode(messages)
            guard data.count <= 4 << 20 else { throw CocoaError(.fileWriteOutOfSpace) }
            try data.write(to: chatFile, options: .atomic)
        } catch { self.error = "Chatten kunde inte sparas: " + error.localizedDescription }
    }
    func waitUntilIdle() async { await operation?.value }
    #if DEBUG
    static func reviewPreview() -> REAMAppModel {
        let value = REAMAppModel(root: FileManager.default.temporaryDirectory.appendingPathComponent("flash-local-review"))
        value.isPreview = true; value.status = "Layoutförhandsvisning – ingen modell körs."
        value.messages = [.init(role: .user, text: "Hur körs modellen?"),
                          .init(role: .assistant, text: "Exempeltext för granskning av chattens layout.")]
        return value
    }
    static func reviewThinkingPreview(active: Bool) -> REAMAppModel {
        let value = reviewPreview()
        value.messages = [.init(role: .user, text: "Vad är 50 + 5?"),
            .init(role: .assistant, text: active ? "" : "50 + 5 = 55.",
                reasoning: "Jag summerar 50 och 5 för att få resultatet.", isThinking: active)]
        return value
    }
    #endif
}

/// Authenticated LAN inference on the standard, non-LES Ornith engine.
/// No optional experimental package is required to show or use this API.
extension REAMAppModel {
    func startAPI() {
        guard canStartAPI else {
            error = "Verifiera och ladda Ornith innan du startar API:t."
            return
        }
        guard (1024...65535).contains(apiPort) else {
            error = "Välj en port mellan 1024 och 65535."
            return
        }
        do {
            let store = LocalAPIKeyStore()
            if !apiKeyDraft.isEmpty { try store.save(apiKeyDraft) }
            let key = try store.loadOrCreate()
            let id = UUID()
            apiAccessKey = key
            apiKeyDraft = ""
            apiRequestID = id
            apiStatus = "Startar lokalt API…"
            error = nil
            apiServer = try LocalModelAPIServer(
                key: key, port: UInt16(apiPort), defaults: .init(),
                capabilities: .init(context: .legacy4K),
                handler: { [weak self] request in
                    guard let self else {
                        throw LocalAPIError(status: 503, message: "The phone service has closed")
                    }
                    return try await self.openAPIRequest(request)
                },
                onState: { [weak self] state in
                    Task { @MainActor in
                        guard let self, self.apiRequestID == id else { return }
                        switch state {
                        case .ready(let port, let addresses):
                            self.apiURLs = addresses.map { "http://\($0):\(port)/v1" }
                            self.apiStatus = addresses.isEmpty
                                ? "Servern lyssnar. Anslut telefonen till Wi-Fi eller Ethernet."
                                : "API aktivt · Ornith på telefonen"
                        case .failed(let message):
                            self.error = message
                            self.stopAPI()
                        case .stopped:
                            self.stopAPI()
                        }
                    }
                })
        } catch {
            error = error.localizedDescription
            stopAPI()
        }
    }

    func stopAPI() {
        apiRequestID = nil
        apiServer?.stop()
        apiServer = nil
        apiCancellation?.cancel()
        apiCancellation = nil
        apiURLs = []
        apiAccessKey = ""
        apiStatus = "Avstängd"
    }

    func rotateAPIKey() {
        guard apiServer == nil, !isBusy else { return }
        do {
            try LocalAPIKeyStore().save(LocalModelAPIServer.makeAccessKey())
            apiKeyDraft = ""
            apiStatus = "Nyckeln är sparad i iPhones nyckelring."
            error = nil
        } catch { error = error.localizedDescription }
    }

    private func openAPIRequest(_ request: LocalChatRequest) throws -> LocalChatGeneration {
        guard apiServer != nil, canChat, selectedModel == .ornith, let artifact else {
            throw LocalAPIError(status: 503, message: "Load Ornith and keep the phone app active")
        }
        guard request.tools.isEmpty, request.toolChoice == .none,
              request.messages.allSatisfy({ ["system", "user", "assistant"].contains($0.role) }) else {
            throw LocalAPIError.invalid("messages", "The standard Ornith engine supports text messages, not tool calls")
        }
        guard request.options.temperature == 0, request.options.topP == 1 else {
            throw LocalAPIError.invalid("temperature", "This engine supports greedy decoding only")
        }
        guard request.options.maximumNewTokens < settings.contextTokens else {
            throw LocalAPIError.invalid("max_tokens", "The output must fit within the 4096-token context")
        }
        let prompt = request.messages.map {
            Qwen35ChatMessage(role: Qwen35ChatMessage.Role(rawValue: $0.role)!, content: $0.content,
                              reasoningContent: $0.reasoning)
        }
        let chosen = REAMRunSettings(
            contextTokens: settings.contextTokens, budgetBytes: settings.budgetBytes,
            maximumNewTokens: request.options.maximumNewTokens, rawCacheBytes: settings.rawCacheBytes,
            residentCacheBytes: settings.residentCacheBytes)
        let flag = REAMCancellation(), id = UUID(), session = inferenceSession
        let output = AsyncThrowingStream<LocalChatEvent, Error>.makeStream(
            bufferingPolicy: .bufferingOldest(64))
        output.continuation.onTermination = { if case .cancelled = $0 { flag.cancel() } }
        let channel = AsyncThrowingStream<Qwen4ExpTextGeneration.Event, Error>.makeStream(
            bufferingPolicy: .bufferingOldest(64))
        apiCancellation = flag
        activeID = id
        activity = .generating
        status = "API: läser prompt…"
        error = nil
        let owner = Task.detached(priority: .userInitiated) {
            let result = await session.generate(
                artifact: artifact, settings: chosen, thinkingEnabled: request.options.thinkingEnabled,
                messages: prompt, cancellation: flag, events: channel.continuation)
            switch result {
            case .success: channel.continuation.finish()
            case .failure(let failure): channel.continuation.finish(throwing: failure)
            }
            return result
        }
        operation = Task { [weak self] in
            guard let self else {
                flag.cancel(); _ = await owner.value; output.continuation.finish(); return
            }
            defer { if activeID == id { apiCancellation = nil; finish(id) } }
            var rendered = QwenThinkingText(startsInThinking: request.options.thinkingEnabled)
            var answerBytes = 0, reasoningBytes = 0
            func emit() throws {
                let answer = String(decoding: rendered.answer.utf8.dropFirst(answerBytes), as: UTF8.self)
                let reasoning = String(decoding: rendered.reasoning.utf8.dropFirst(reasoningBytes), as: UTF8.self)
                answerBytes = rendered.answer.utf8.count
                reasoningBytes = rendered.reasoning.utf8.count
                if !answer.isEmpty || !reasoning.isEmpty {
                    guard case .enqueued = output.continuation.yield(.delta(content: answer, reasoning: reasoning)) else {
                        flag.cancel()
                        throw LocalAPIError(status: 503, message: "Streaming client cannot keep up")
                    }
                }
            }
            do {
                for try await event in channel.stream {
                    switch event {
                    case .prefill(let p):
                        status = "API: läser prompt \(p.promptTokensCompleted)/\(p.promptTokensTotal)"
                    case .token(_, let delta, _):
                        rendered.append(delta)
                        try emit()
                        status = rendered.isThinking ? "API: tänker…" : "API: skriver svar…"
                    case .trailingText(let delta):
                        rendered.append(delta)
                        try emit()
                    }
                }
                rendered.finish()
                try emit()
            } catch { flag.cancel() }
            switch await owner.value {
            case .success(let summary):
                if flag.requested() {
                    output.continuation.finish(throwing: CancellationError())
                    status = "API: avbrutet"
                } else {
                    output.continuation.yield(.complete(.init(
                        promptTokens: summary.promptTokenIDs.count,
                        completionTokens: summary.generatedTokenIDs.count,
                        reasoningTokens: 0, forcedTokens: 0,
                        finishReason: summary.stopReason == .endOfGeneration ? "stop" : "length")))
                    output.continuation.finish()
                    apiCompletedRequests += 1
                    status = "API: klart · \(summary.generatedTokenIDs.count) token"
                }
            case .failure(let failure):
                output.continuation.finish(throwing: failure)
                if !flag.requested() { error = failure.localizedDescription }
                status = "API: genereringen stoppades"
            }
        }
        return LocalChatGeneration(events: output.stream, cancel: { flag.cancel() })
    }
}
