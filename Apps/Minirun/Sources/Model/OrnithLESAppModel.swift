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
final class LESRequestLease: @unchecked Sendable {
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
    private struct SavedAdmission: Codable {
        let baseline: LESNumericalCheck
        let prefill: LESPrefillCheck?
        var automaticConfiguration: LESPrefillConfiguration? = nil
        var automaticContextTokens: Int? = nil
    }
    private var artifact: VerifiedGGUFArtifact?
    private var prepared: URL?
    private var chat: OrnithLESChatGeneration?
    private var chatCheck: LESNumericalCheck?
    private var prefillCheck: LESPrefillCheck?
    private var reusedAdmission = false
    private var automaticConfiguration: LESPrefillConfiguration?
    private var automaticContextTokens: Int?
    private var effectiveConfiguration = LESPrefillConfiguration()
    func usedConfiguration() -> LESPrefillConfiguration { effectiveConfiguration }
    private var admissions: [LESMemoryAdmission] = []
    func memoryAdmissionReport() -> [LESMemoryAdmission] { admissions }
    func admissionWasReused() -> Bool { reusedAdmission }
    func release() { chat?.close(); chat = nil; chatCheck = nil; prefillCheck = nil; artifact = nil; prepared = nil }
    func resetChat() { chat?.resetConversation() }
    func closeChat() { chat?.close(); chat = nil }
    private func admissionIdentity(_ artifact: VerifiedGGUFArtifact, _ prepared: URL) throws -> [String: String] {
        try artifact.assertUnchanged()
        guard let executable = Bundle.main.executableURL else { throw LESChatPresentationError.notReady }
        return ["model": artifact.manifest.sha256, "native": LESContext.revision,
            "reference": Qwen4ExpCPUReference.sourceRevision, "source": BuildInfo.gitRevision,
            // Also bind the actual executable: a developer can rebuild a dirty
            // checkout without changing either its commit label or build number.
            "executable": try FileDigestComputer.digest(ofFileAt: executable, algorithm: .sha256).hex,
            "build": Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "unknown",
            "device": Sysctl.string("hw.machine") ?? "unknown",
            "os": ProcessInfo.processInfo.operatingSystemVersionString,
            "prepared": try FileDigestComputer.digest(ofFileAt: prepared.appendingPathComponent("seal.json"), algorithm: .sha256).hex]
    }
    private func restoreAdmission(_ artifact: VerifiedGGUFArtifact, _ prepared: URL) -> SavedAdmission? {
        let url = prepared.appendingPathComponent("chat-admission-v1.json")
        do {
            let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey])
            guard values.isRegularFile == true, values.isSymbolicLink != true,
                  let size = values.fileSize, size <= 2 << 20,
                  let key = try LocalAPIKeyStore(service: "com.dhafen.flashlocal.ornith-admission.v1").load(),
                  let data = LocalAdmissionSeal.open(try Data(contentsOf: url), identity: try admissionIdentity(artifact, prepared), key: key),
                  let saved = try? JSONDecoder().decode(SavedAdmission.self, from: data),
                  saved.baseline.nativeBenchmarkAdmitted,
                  saved.baseline.modelSHA256 == artifact.manifest.sha256,
                  saved.baseline.nativeRevision == LESContext.revision,
                  saved.baseline.referenceRevision == Qwen4ExpCPUReference.sourceRevision else { return nil }
            return saved
        } catch { return nil } // A missing/invalid/locked seal always runs the real check.
    }
    private func saveAdmission(_ artifact: VerifiedGGUFArtifact, _ prepared: URL) {
        guard let chatCheck, chatCheck.nativeBenchmarkAdmitted else { return }
        do {
            let key = try LocalAPIKeyStore(service: "com.dhafen.flashlocal.ornith-admission.v1").loadOrCreate()
            let data = try LocalAdmissionSeal.seal(JSONEncoder().encode(SavedAdmission(baseline: chatCheck, prefill: prefillCheck,
                automaticConfiguration: automaticConfiguration, automaticContextTokens: automaticContextTokens)),
                identity: admissionIdentity(artifact, prepared), key: key)
            let url = prepared.appendingPathComponent("chat-admission-v1.json")
            try data.write(to: url, options: .atomic)
            try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
        } catch { /* Persistence is optional; it never substitutes for admission. */ }
    }
    private func prepareFiles(modelsRoot: URL, privateRoot: URL, existing: VerifiedGGUFArtifact?, control: LESControl,
                              progress: @Sendable (String) -> Void) throws {
        if artifact == nil {
            // The active owner also bootstraps the Files model directory; it
            // no longer depends on an invisible legacy root preparing storage.
            try LocalModelStorage.prepare(root: modelsRoot)
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
            prepared = try LESPreparation.prepare(artifact: artifact, privateRoot: privateRoot, control: control, progress: progress,
                onAdmission: { self.admissions.append($0) })
        }
    }
    func prepareChat(modelsRoot: URL, privateRoot: URL, existing: VerifiedGGUFArtifact?, control: LESControl,
                     configuration requested: LESPrefillConfiguration = .init(),
                     contextProfile: LocalContextProfile = .standard,
                     preferAutomatic: Bool = false,
                     progress: @Sendable (String) -> Void,
                     onCheck: @Sendable (LESNumericalCheck) -> Void) throws -> LESNumericalCheck {
        chat?.close(); chat = nil
        reusedAdmission = chatCheck != nil
        try prepareFiles(modelsRoot: modelsRoot, privateRoot: privateRoot, existing: existing, control: control, progress: progress)
        guard let artifact, let prepared else { throw CancellationError() }
        try artifact.assertUnchanged()
        if chatCheck == nil, let saved = restoreAdmission(artifact, prepared) {
            chatCheck = saved.baseline; prefillCheck = saved.prefill
            automaticConfiguration = saved.automaticConfiguration; automaticContextTokens = saved.automaticContextTokens
            reusedAdmission = true
            progress("Återanvänder godkänd kontroll för samma app, modell och motor…")
        }
        if chatCheck == nil {
            progress("Kontrollerar mot originalvikterna innan snabbchatten startas…")
            chatCheck = try LESBenchmark.check(artifact: artifact, prepared: prepared, control: control,
                progress: progress, onAdmission: { self.admissions.append($0) })
        }
        guard let check = chatCheck else { throw CancellationError() }
        onCheck(check)
        guard check.nativeBenchmarkAdmitted else { return check }
        var configuration = requested
        if preferAutomatic, automaticContextTokens == contextProfile.rawValue, let automaticConfiguration {
            configuration = automaticConfiguration
            progress("Använder sparad, jämförd batchprofil för samma modell, motor, telefon och kontext…")
        }
        try configuration.validate()
        if configuration != LESPrefillConfiguration() {
            var effective = configuration
            effective.physicalTokens = try configuration.physicalTokens(checkedDefault: check.nativePrefillTokens)
            if prefillCheck?.configuration != effective || prefillCheck?.passed != true {
                prefillCheck = try LESBenchmark.checkPrefill(artifact: artifact, prepared: prepared,
                    baseline: check, configuration: effective, control: control, progress: progress,
                    onAdmission: { self.admissions.append($0) })
            }
        }
        func open(_ value: LESPrefillConfiguration) throws -> OrnithLESChatGeneration {
            try OrnithLESChatGeneration(artifact: artifact, prepared: prepared, check: check,
                control: control, prefillConfiguration: value, prefillCheck: prefillCheck, contextProfile: contextProfile,
                onAdmission: { self.admissions.append($0) })
        }
        do { chat = try open(configuration) }
        catch GGUFError.limit where preferAutomatic && configuration != LESPrefillConfiguration() {
            progress("Den större batchen ryms inte i aktuellt iOS-minne. Använder kontrollerad grundbatch.")
            configuration = .init(); chat = try open(configuration)
        }
        effectiveConfiguration = configuration
        if !control.shouldAbort() { saveAdmission(artifact, prepared) }
        return check
    }
    func prefillAdmission() -> LESPrefillCheck? { prefillCheck }
    func benchmarkCache(modelsRoot: URL, privateRoot: URL, control: LESControl,
                        configuration: LESPrefillConfiguration, context: LocalContextProfile,
                        progress: @Sendable (String) -> Void) throws -> OrnithCacheBenchmark.Report {
        let check = try prepareChat(modelsRoot: modelsRoot, privateRoot: privateRoot, existing: nil,
            control: control, configuration: configuration, contextProfile: context,
            progress: progress, onCheck: { _ in })
        guard check.nativeBenchmarkAdmitted, let chat else { throw LESChatPresentationError.numericalCheck }
        defer { self.chat?.close(); self.chat = nil }
        return try OrnithCacheBenchmark.execute(engine: chat, progress: progress)
    }
    func benchmarkPrefill(modelsRoot: URL, privateRoot: URL, existing: VerifiedGGUFArtifact?,
                          configuration: LESPrefillConfiguration, control: LESControl,
                          contextProfile: LocalContextProfile = .standard,
                          progress: @Sendable (String) -> Void,
                          onCheck: @Sendable (LESNumericalCheck) -> Void) throws -> OrnithPrefillBenchmark.Report {
        let baseline = try prepareChat(modelsRoot: modelsRoot, privateRoot: privateRoot, existing: existing,
            control: control, configuration: configuration, contextProfile: contextProfile, progress: progress, onCheck: onCheck)
        chat?.close(); chat = nil
        guard let artifact, let prepared else { throw CancellationError() }
        let report = try OrnithPrefillBenchmark.execute(artifact: artifact, prepared: prepared, baseline: baseline,
            candidate: configuration, check: prefillCheck, control: control, contextProfile: contextProfile, progress: progress,
            onAdmission: { self.admissions.append($0) })
        if let winner = report.preferredConfiguration {
            automaticConfiguration = winner; automaticContextTokens = contextProfile.rawValue
            saveAdmission(artifact, prepared)
        }
        return report
    }
    func generateChat(messages: [Qwen35ChatMessage], options: LocalGenerationOptions,
                      tools: [LocalToolDefinition] = [],
                      events: AsyncThrowingStream<Qwen4ExpTextGeneration.Event, Error>.Continuation)
        -> Result<OrnithLESChatGeneration.Summary, Error> {
        let result = Result {
            guard let chat else { throw LESChatPresentationError.notReady }
            return try autoreleasepool {
                try chat.generate(messages: messages, options: options, tools: tools) { event in
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
        chat?.close(); chat = nil; chatCheck = nil; prefillCheck = nil
        try prepareFiles(modelsRoot: modelsRoot, privateRoot: privateRoot, existing: existing, control: control, progress: progress)
        guard let artifact, let prepared else { throw CancellationError() }
        // An explicit comparison is always fresh; a later failure must not
        // leave a previously admitted startup receipt behind.
        let receipt = prepared.appendingPathComponent("chat-admission-v1.json")
        if FileManager.default.fileExists(atPath: receipt.path) { try FileManager.default.removeItem(at: receipt) }
        // Every explicit run performs its own check and memory sampling.
        // Re-exporting a cached failed check must not look like another run
        // with zero measured RAM. The verified file/preparation may be reused.
        let check = try LESBenchmark.check(artifact: artifact, prepared: prepared, control: control,
            progress: progress, onAdmission: { self.admissions.append($0) })
        chatCheck = check.nativeBenchmarkAdmitted ? check : nil
        if !control.shouldAbort() { saveAdmission(artifact, prepared) }
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
    private(set) var sceneIsActive = true
    private(set) var loadingStartedAt: Date?
    @ObservationIgnored private var wantsModelLoaded = false
    var isAPIListening: Bool { apiServer != nil || (isPreview && isAPIRunning) }
    var loadState: String {
        if isShuttingDown { return "Laddar ur" }
        if loadingStartedAt != nil && isBusy { return "Laddar" }
        if chatReady { return isBusy ? "Genererar" : "Redo" }
        return error == nil ? "Ej laddad" : "Fel"
    }
    var loadingFootprintText: String {
        ProcessFootprint.current().map { String(format: "Appminne %.2f GB", Double($0.footprintBytes) / 1e9) }
            ?? "Minnesmätning saknas"
    }
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
    private(set) var startupAdmissionReused = false
    private(set) var activationSeconds: Double?
    private(set) var chatMessages: [REAMMessage] = []
    private(set) var chatProgress: Qwen4ExpTextGeneration.Progress?
    private(set) var chatSummary: OrnithLESChatGeneration.Summary?
    private(set) var chatMeasurementURL: URL?
    var chatDraft = ""
    var batchConfiguration = LESPrefillConfiguration() {
        didSet {
            if persistOptions, (try? batchConfiguration.validate()) != nil,
               let data = try? JSONEncoder().encode(batchConfiguration) {
                preferences.set(data, forKey: Self.batchPreferenceKey)
            }
        }
    }
    private(set) var activeBatchConfiguration: LESPrefillConfiguration?
    var automaticBatch = true {
        didSet { if persistOptions { preferences.set(automaticBatch, forKey: "FlashLocal.Ornith.AutomaticBatch.v1") } }
    }
    private(set) var activeAutomaticBatch = true
    private(set) var activePrefillCheck: LESPrefillCheck?
    private(set) var prefillBenchmark: OrnithPrefillBenchmark.Report?
    private(set) var prefillBenchmarkFailure: String?
    private(set) var prefillBenchmarkURL: URL?
    private(set) var cacheBenchmark: OrnithCacheBenchmark.Report?
    private(set) var cacheBenchmarkURL: URL?
    var batchSettingsChanged: Bool { chatReady && (activeBatchConfiguration != batchConfiguration || activeContextProfile != contextProfile || activeAutomaticBatch != automaticBatch) }
    var systemPrompt = "" {
        didSet {
            if persistOptions, systemPrompt.utf8.count <= 32768 {
                preferences.set(systemPrompt, forKey: Self.systemPreferenceKey)
            }
        }
    }
    func applyProfile(_ profile: LocalChatProfile) {
        guard canStart, !isAPIRunning else { return }
        systemPrompt = profile.systemPrompt
    }
    var contextProfile: LocalContextProfile = .standard {
        didSet { if persistOptions { preferences.set(contextProfile.rawValue, forKey: "FlashLocal.Ornith.Context.v1") } }
    }
    private(set) var activeContextProfile: LocalContextProfile = .standard
    var contextCapacityTokens: Int { activeContextProfile.rawValue }
    var chatOptions = LocalGenerationOptions() {
        didSet {
            if persistOptions {
                do {
                    try chatOptions.validate()
                    let data = try JSONEncoder().encode(chatOptions)
                    preferences.set(data, forKey: Self.optionsPreferenceKey)
                } catch { /* Incomplete numeric edits are not persisted. */ }
            }
        }
    }
    private static let optionsPreferenceKey = "FlashLocal.Ornith.GenerationOptions.v1"
    private static let batchPreferenceKey = "FlashLocal.Ornith.PrefillOptions.v1"
    private static let systemPreferenceKey = "FlashLocal.Ornith.SystemPrompt.v1"
    @ObservationIgnored private var persistOptions = false
    @ObservationIgnored private var preferences: UserDefaults = .standard
    var chatThinkingEnabled: Bool {
        get { chatOptions.enableThinking }
        set { chatOptions = (newValue ? OrnithThinkingPreset.medium : OrnithThinkingPreset.off).applying(to: chatOptions) }
    }
    private(set) var isAPIRunning = false
    private(set) var apiStatus = "Avstängd"
    private(set) var apiAccessKey = ""
    private(set) var apiURLs: [String] = []
    private(set) var apiCompletedRequests = 0
    var apiKeyDraft = ""
    private(set) var apiLastSummary: OrnithLESChatGeneration.Summary?
    var apiPort = 8080
    var apiKeepAwake = true
    @ObservationIgnored private var apiServer: LocalModelAPIServer?
    @ObservationIgnored private var apiID: UUID?
    @ObservationIgnored private var apiSystemPrompt = ""
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
    var canSendChat: Bool { canStart && sceneIsActive && chatReady && !batchSettingsChanged }
    var canBenchmark: Bool { canStart && apiServer == nil }
    var shouldKeepAwake: Bool { isBusy || (isAPIRunning && apiKeepAwake) }
    init(modelsRoot: URL, privateRoot: URL, documents: URL, preferences: UserDefaults? = nil) {
        self.modelsRoot = modelsRoot; self.privateRoot = privateRoot; self.documents = documents
        if let preferences { self.preferences = preferences; persistOptions = true; restoreHistory() }
    }
    static func live(preferences: UserDefaults = .standard) -> OrnithLESAppModel {
        let fm = FileManager.default
        let documents = fm.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let model = Self(modelsRoot: documents.appendingPathComponent("Models"),
            privateRoot: fm.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("FlashLocal/OrnithLES"), documents: documents)
        model.preferences = preferences
        if let data = preferences.data(forKey: Self.optionsPreferenceKey),
           let options = try? JSONDecoder().decode(LocalGenerationOptions.self, from: data),
           let restored = try? OrnithThinkingPreset.restoring(options) {
            model.chatOptions = restored
        }
        if let data = preferences.data(forKey: Self.batchPreferenceKey),
           let batch = try? JSONDecoder().decode(LESPrefillConfiguration.self, from: data),
           (try? batch.validate()) != nil { model.batchConfiguration = batch }
        if let prompt = preferences.string(forKey: Self.systemPreferenceKey), prompt.utf8.count <= 32768 {
            model.systemPrompt = prompt
        }
        if let context = LocalContextProfile(rawValue: preferences.integer(forKey: "FlashLocal.Ornith.Context.v1")) {
            model.contextProfile = context
        }
        if preferences.object(forKey: "FlashLocal.Ornith.AutomaticBatch.v1") != nil {
            model.automaticBatch = preferences.bool(forKey: "FlashLocal.Ornith.AutomaticBatch.v1")
        }
        model.restoreHistory()
        model.persistOptions = true
        return model
    }
    static func reviewPreview() -> OrnithLESAppModel {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("FlashLocal-Preview-NoAccess")
        let model = Self(modelsRoot: root, privateRoot: root, documents: root); model.isPreview = true
        model.chatOptions = .init(); model.batchConfiguration = .init(); model.systemPrompt = ""
        model.status = "Förhandsvisning av layout – ingen modell körs."
        return model
    }
    static func reviewChatPreview(thinking: Bool) -> OrnithLESAppModel {
        let model = Self.reviewPreview()
        model.activeBatchConfiguration = model.batchConfiguration
        model.chatReady = true; model.isBusy = thinking
        model.status = thinking ? "Tänker lokalt…" : "Förhandsvisning av snabbchatten – ingen modell körs."
        model.chatMessages = [.init(role: .user, text: "Hur testar jag en cache?")]
        model.chatMessages.append(.init(role: .assistant,
            text: thinking ? "" : "**Cachetest:**\n\n- Testa träffar och missar.\n- Kontrollera att `oldest` byts ut när cachen är full.\n\n```swift\nassert(cache.count <= capacity)\n```",
            reasoning: "Jag börjar med cachens gränser och vilka egenskaper testet behöver kontrollera.", isThinking: thinking))
        return model
    }
    static func reviewAPIPreview() -> OrnithLESAppModel {
        let model = Self.reviewPreview()
        model.activeBatchConfiguration = model.batchConfiguration
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
        stopAPI()
        await unloadModel()
    }
    func unloadModel() async {
        guard !isShuttingDown else { await operation?.value; return }
        wantsModelLoaded = false
        isShuttingDown = true
        stop(); await operation?.value; await worker.release()
        chatReady = false; chatControl = nil; isShuttingDown = false
        loadingStartedAt = nil; status = "Ej laddad · chatthistoriken finns kvar."
        if apiServer != nil { isAPIRunning = false; apiStatus = "Servern lyssnar · modellen är urladdad" }
        saveHistory()
    }
    func prepareForInactivity() {
        sceneIsActive = false
        if apiServer != nil { isAPIRunning = false; apiStatus = "API pausat · appen är inte aktiv" }
        // Metal work must not continue after the app is backgrounded. An idle
        // context is retained; only an interrupted/poisoned graph is rebuilt.
        if isBusy { control?.cancel() }
    }
    func enterBackground() async {
        prepareForInactivity()
        let pending = operation
        await pending?.value
        saveHistory()
        if sceneIsActive { enterForeground() }
        else if chatReady { status = "Modellen behålls · inferens pausad i bakgrunden." }
    }
    func enterForeground() {
        sceneIsActive = true
        apiServer?.refreshState()
        guard !isPreview else { return }
        let pending = operation
        Task { [weak self] in
            await pending?.value
            guard let self, self.sceneIsActive, self.wantsModelLoaded, !self.chatReady, self.canStart else { return }
            self.activateChat()
        }
    }
    func handleMemoryPressure() async {
        await unloadModel()
        status = "Modellen laddades ur vid iOS-minnestryck. Samtalet finns kvar."
    }
    func activateChat(existing: VerifiedGGUFArtifact? = nil) {
        guard canStart, sceneIsActive, !isAPIRunning, (!chatReady || batchSettingsChanged) else { return }
        wantsModelLoaded = true
        do { try batchConfiguration.validate() } catch { self.error = error.localizedDescription; return }
        let requestedBatch = batchConfiguration
        let requestedContext = contextProfile
        let useAutomatic = automaticBatch
        let activationStart = MonotonicClock.now()
        activationSeconds = nil; startupAdmissionReused = false
        loadingStartedAt = Date()
        chatReady = false; activePrefillCheck = nil
        let flag = LESControl(), id = UUID()
        activeID = id; control = flag; chatControl = flag; isBusy = true; error = nil
        status = "Förbereder Ornith snabbchatt…"
        operation = Task { [weak self] in
            guard let self else { return }
            defer { if activeID == id { activeID = nil; control = nil; isBusy = false; operation = nil; loadingStartedAt = nil } }
            do {
                let check = try await worker.prepareChat(modelsRoot: modelsRoot, privateRoot: privateRoot, existing: existing,
                    control: flag, configuration: requestedBatch, contextProfile: requestedContext, preferAutomatic: useAutomatic, progress: { [weak self] value in
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
                let effective = await worker.usedConfiguration()
                activeBatchConfiguration = effective; batchConfiguration = effective; activeAutomaticBatch = useAutomatic
                activeContextProfile = requestedContext
                if effective == LESPrefillConfiguration() { activePrefillCheck = nil }
                else { activePrefillCheck = await worker.prefillAdmission() }
                startupAdmissionReused = await worker.admissionWasReused()
                activationSeconds = MonotonicClock.seconds(since: activationStart)
                chatReady = true
                apiServer?.refreshState()
                status = startupAdmissionReused ? "Snabbchatten är redo · tidigare godkänd startkontroll återanvänd."
                    : "Snabbchatten är redo · ny startkontroll godkänd."
            } catch {
                await worker.closeChat(); chatReady = false; chatControl = nil
                if !flag.isCancellationRequested { self.error = error.localizedDescription }
                status = "Snabbchatten kunde inte startas."
            }
        }
    }
    func sendChat() {
        guard canSendChat, let flag = chatControl else { return }
        let input = chatDraft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !input.isEmpty, input.utf8.count <= 262144 else { return }
        guard systemPrompt.utf8.count <= 32768 else { error = "Systemprompt får vara högst 32 KiB."; return }
        do { try chatOptions.validate() } catch { self.error = error.localizedDescription; return }
        chatDraft = ""; chatMessages.append(.init(role: .user, text: input))
        let prompt = Qwen35ChatMessage.applyingSystemPrompt(systemPrompt, to: REAMMessage.generationPrompt(chatMessages))
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
            var result = await task.value
            if flag.isCancellationRequested { await worker.closeChat(); result = .failure(CancellationError()) }
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
                if flag.isCancellationRequested { await recoverAPIContext() }
            }
            saveHistory()
        }
    }
    func startPrefillBenchmark(existing: VerifiedGGUFArtifact? = nil) {
        guard canBenchmark else { return }
        do { try batchConfiguration.validate() } catch { self.error = error.localizedDescription; return }
        guard batchConfiguration != LESPrefillConfiguration() else { error = "Välj en annan batch än Auto 4 för jämförelsen."; return }
        let flag = LESControl(), id = UUID(), selected = batchConfiguration, selectedContext = contextProfile
        activeID = id; control = flag; isBusy = true; chatReady = false; chatControl = nil
        error = nil; prefillBenchmark = nil; prefillBenchmarkFailure = nil; prefillBenchmarkURL = nil
        status = "Förbereder jämförelse av promptbatcher…"
        operation = Task { [weak self] in
            guard let self else { return }
            defer { if activeID == id { activeID = nil; control = nil; isBusy = false; operation = nil } }
            do {
                prefillBenchmark = try await worker.benchmarkPrefill(modelsRoot: modelsRoot, privateRoot: privateRoot,
                    existing: existing, configuration: selected, control: flag, contextProfile: selectedContext, progress: { [weak self] value in
                        Task { @MainActor in guard let self, self.activeID == id else { return }; self.status = value }
                    }, onCheck: { [weak self] value in
                        Task { @MainActor in guard let self, self.activeID == id else { return }; self.numericalCheck = value }
                    })
                status = prefillBenchmark?.comparable == true ? "Batchjämförelsen är klar · samma genererade token." :
                    "Batchtestet är klart men längd, token eller termiskt tillstånd skiljer. Ingen säker vinnare utses."
            } catch {
                prefillBenchmarkFailure = error.localizedDescription; self.error = error.localizedDescription
                status = "Batchtestet stoppades. Auto är kvar som separat val."
            }
            activePrefillCheck = await worker.prefillAdmission()
            memoryAdmissions = await worker.memoryAdmissionReport()
        }
    }
    func savePrefillBenchmark() {
        guard canStart, prefillBenchmark != nil || prefillBenchmarkFailure != nil else { return }
        struct Report: Encodable {
            let schemaVersion = 1
            let appBuild, sourceRevision, modelSHA256, nativeRevision: String
            let deviceIdentifier: String?
            let operatingSystem: String
            let timestamp: Date
            let userDeclaredAirplaneMode: Bool
            let numericalCheck: LESNumericalCheck?
            let candidateCheck: LESPrefillCheck?
            let benchmark: OrnithPrefillBenchmark.Report?
            let admissions: [LESMemoryAdmission]
            let failure: String?
        }
        do {
            let report = Report(appBuild: Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "unknown",
                sourceRevision: BuildInfo.gitRevision, modelSHA256: GGUFModelManifest.primaryOrnith.sha256,
                nativeRevision: LESContext.revision, deviceIdentifier: Sysctl.string("hw.machine"),
                operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString, timestamp: Date(),
                userDeclaredAirplaneMode: userDeclaredAirplaneMode, numericalCheck: numericalCheck,
                candidateCheck: activePrefillCheck, benchmark: prefillBenchmark, admissions: memoryAdmissions, failure: prefillBenchmarkFailure)
            let folder = documents.appendingPathComponent("Diagnostics")
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            let url = folder.appendingPathComponent("ornith-prefill-" + UUID().uuidString + ".json")
            let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]; encoder.dateEncodingStrategy = .iso8601
            try encoder.encode(report).write(to: url, options: .atomic); prefillBenchmarkURL = url
        } catch { self.error = error.localizedDescription }
    }
    func startCacheBenchmark() {
        guard canBenchmark, sceneIsActive else { return }
        let flag = LESControl(), id = UUID(), batch = batchConfiguration, context = contextProfile
        activeID = id; control = flag; isBusy = true; chatReady = false; chatControl = nil
        error = nil; cacheBenchmark = nil; cacheBenchmarkURL = nil
        status = "Jämför verklig prefixåteranvändning med en ny kontext…"
        operation = Task { [weak self] in
            guard let self else { return }
            defer { if activeID == id { activeID = nil; control = nil; isBusy = false; operation = nil } }
            do {
                let report = try await worker.benchmarkCache(modelsRoot: modelsRoot, privateRoot: privateRoot,
                    control: flag, configuration: batch, context: context, progress: { [weak self] value in
                        Task { @MainActor in guard let self, self.activeID == id else { return }; self.status = value }
                    })
                cacheBenchmark = report
                struct Export: Encodable {
                    let sourceRevision, appBuild: String
                    let timestamp: Date
                    let report: OrnithCacheBenchmark.Report
                }
                let folder = documents.appendingPathComponent("Diagnostics")
                try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
                let url = folder.appendingPathComponent("ornith-cache-" + UUID().uuidString + ".json")
                let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]; encoder.dateEncodingStrategy = .iso8601
                try encoder.encode(Export(sourceRevision: BuildInfo.gitRevision,
                    appBuild: Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "unknown",
                    timestamp: Date(), report: report)).write(to: url, options: .atomic)
                cacheBenchmarkURL = url
                status = report.cacheVerified ? "Cachetest klart · identiska svar med och utan återanvändning." :
                    "Cachetest klart · återanvändning eller svar avvek. Se rapporten."
            } catch { self.error = error.localizedDescription; status = "Cachetestet stoppades."; await worker.closeChat() }
        }
    }
    func startAPI() {
        guard canStart, chatReady, !batchSettingsChanged, apiServer == nil else { return }
        guard (1024...65535).contains(apiPort) else { error = "Välj en port mellan 1024 och 65535."; return }
        do {
            let store = LocalAPIKeyStore()
            if !apiKeyDraft.isEmpty { try store.save(apiKeyDraft) }
            let key = try store.loadOrCreate(), id = UUID()
            guard systemPrompt.utf8.count <= 32768 else { throw LocalAPIError.invalid("system", "Systemprompt får vara högst 32 KiB.") }
            apiAccessKey = key; apiKeyDraft = ""; apiID = id; apiSystemPrompt = systemPrompt
            apiStatus = "Startar lokalt API…"; error = nil
            apiServer = try LocalModelAPIServer(key: key, port: UInt16(apiPort), defaults: chatOptions,
                capabilities: .init(context: activeContextProfile), handler: { [weak self] request in
                guard let self else { throw LocalAPIError(status: 503, message: "Phone service is closed") }
                return try await self.openAPIRequest(request)
            }, onState: { [weak self] state in
                Task { @MainActor in
                    guard let self, self.apiID == id else { return }
                    switch state {
                    case .ready(let port, let addresses):
                        self.isAPIRunning = self.sceneIsActive && self.chatReady
                        self.apiURLs = addresses.map { "http://\($0):\(port)/v1" }
                        self.apiStatus = !self.sceneIsActive ? "API pausat · appen är inte aktiv" :
                            !self.chatReady ? "Servern lyssnar · ladda modellen" :
                            addresses.isEmpty ? "Servern lyssnar. Anslut till Wi-Fi eller Ethernet och återgå till appen för att visa adressen." : "API aktivt · samma lokala Ornith-motor"
                    case .failed(let message):
                        self.error = message; self.stopAPI()
                    case .stopped: self.stopAPI()
                    }
                }
            })
        } catch { self.error = error.localizedDescription; stopAPI() }
    }
    func stopAPI() {
        apiID = nil; apiServer?.stop(); apiServer = nil; apiSystemPrompt = ""
        isAPIRunning = false; apiAccessKey = ""; apiURLs = []; apiStatus = "Avstängd"
        if operationIsAPI { control?.cancel() }
    }
    func rotateAPIKey() {
        guard canStart, apiServer == nil else { return }
        do {
            try LocalAPIKeyStore().save(LocalModelAPIServer.makeAccessKey())
            apiKeyDraft = ""; apiStatus = "Ny nyckel sparad på enheten. Kopiera den när API:t startats."; error = nil
        } catch { self.error = error.localizedDescription }
    }
    private func recoverAPIContext() async {
        guard wantsModelLoaded, sceneIsActive, !isShuttingDown else { return }
        let recovery = LESControl(); control = recovery
        do {
            let check = try await worker.prepareChat(modelsRoot: modelsRoot, privateRoot: privateRoot, existing: nil,
                control: recovery, configuration: activeBatchConfiguration ?? .init(), contextProfile: activeContextProfile,
                progress: { _ in }, onCheck: { _ in })
            guard check.nativeBenchmarkAdmitted, !recovery.isCancellationRequested else { throw CancellationError() }
            numericalCheck = check; chatControl = recovery; chatReady = true; status = "Modellen är redo för nästa anrop."
            apiServer?.refreshState()
        } catch { await worker.closeChat(); status = "Aktivera snabbchatten igen för att återuppta API:t." }
    }
    /// The transport calls this before sending successful headers. All UI and
    /// API requests reserve the same owner; no second model context is loaded.
    func openAPIRequest(_ request: LocalChatRequest) throws -> LocalChatGeneration {
        guard apiServer != nil, sceneIsActive, !isShuttingDown, !isPreview else {
            throw LocalAPIError(status: 503, message: "Start the local API on the phone")
        }
        guard !isBusy else { throw LocalAPIError(status: 429, message: "The phone is processing another request") }
        guard chatReady, let flag = chatControl else {
            throw LocalAPIError(status: 503, message: "Activate Ornith quick chat on the phone first")
        }
        let id = UUID(), lease = LESRequestLease(flag)
        let output = AsyncThrowingStream<LocalChatEvent, Error>.makeStream(bufferingPolicy: .bufferingOldest(64))
        output.continuation.onTermination = { if case .cancelled = $0 { lease.cancel() } }
        var input = Qwen35ChatMessage.applyingSystemPrompt(apiSystemPrompt, to: request.messages.map {
            Qwen35ChatMessage(role: Qwen35ChatMessage.Role(rawValue: $0.role)!, content: $0.content, reasoningContent: $0.reasoning,
                toolCalls: $0.toolCalls.isEmpty ? nil : $0.toolCalls, toolCallID: $0.toolCallID)
        })
        if !request.toolInstructions.isEmpty {
            let original = input.first?.role == .system ? input.removeFirst().content : ""
            input.insert(.init(role: .system, content: original + "\n\n" + request.toolInstructions), at: 0)
        }
        let definitions = request.toolChoice.enabled ? request.tools.filter {
            if case .named(let name) = request.toolChoice { return $0.name == name }
            return true
        } : []
        let channel = AsyncThrowingStream<Qwen4ExpTextGeneration.Event, Error>.makeStream(bufferingPolicy: .bufferingOldest(64))
        activeID = id; control = flag; isBusy = true; operationIsAPI = true; error = nil
        status = "API: förbereder lokal generering…"
        let worker = worker
        let promptMessages = input
        let owner = Task.detached(priority: .userInitiated) {
            let result = await worker.generateChat(messages: promptMessages, options: request.options, tools: definitions, events: channel.continuation)
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
                apiLastSummary = summary
                do {
                    let parsed = try LocalToolOutput.parse(text.answer, tools: request.tools, choice: request.toolChoice,
                        parallel: request.parallelToolCalls)
                    if !parsed.calls.isEmpty, summary.stopReason != "endOfGeneration" {
                        throw LocalAPIError(status: 502, message: "Function generation reached its token limit", parameter: "tool_calls")
                    }
                } catch {
                    output.continuation.finish(throwing: error)
                    self.error = error.localizedDescription; status = "API: verktygsanropet godkändes inte. Motorn är redo för ett nytt anrop."
                    break
                }
                #if targetEnvironment(simulator)
                let platform = "iOSSimulator"
                #elseif os(iOS)
                let platform = "iOSDevice"
                #else
                let platform = "macOS"
                #endif
                output.continuation.yield(.complete(.init(promptTokens: summary.promptTokenIDs.count,
                    completionTokens: summary.generatedTokenIDs.count, reasoningTokens: summary.reasoningTokens,
                    forcedTokens: summary.forcedThinkingEndTokens, finishReason: summary.stopReason == "endOfGeneration" ? "stop" : "length",
                    metrics: .init(modelSHA256: numericalCheck?.modelSHA256 ?? "", contextTokens: contextCapacityTokens,
                        evaluatedPromptTokens: summary.evaluatedPromptTokens, reusedPromptTokens: summary.reusedPromptTokens,
                        decodeForwardPasses: summary.decodeForwardPasses, prefillSeconds: summary.prefillSeconds,
                        decodeSeconds: summary.decodeForwardSeconds, elapsedSeconds: summary.elapsedSeconds,
                        prefillIOBytes: summary.prefillIOBytes, decodeIOBytes: summary.decodeIOBytes,
                        peakMemoryBytes: summary.maximumObservedFootprintBytes, thermalStateAtStart: summary.thermalStateAtStart,
                        maximumThermalState: summary.maximumThermalState, platform: platform,
                        deviceIdentifier: Sysctl.string("hw.machine"),
                        appBuild: Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String,
                        sourceRevision: BuildInfo.gitRevision, nativeRevision: LESContext.revision,
                        memoryBudgetBytes: summary.memoryPlan.processCeilingBytes,
                        timeToFirstTokenSeconds: summary.timeToFirstTokenSeconds))))
                output.continuation.finish(); apiCompletedRequests += 1
                status = "API: klart · \(summary.generatedTokenIDs.count) token"
            case .failure(let failure):
                output.continuation.finish(throwing: failure); chatReady = false; chatControl = nil
                if !flag.isCancellationRequested { error = failure.localizedDescription }
                status = "API-anropet stoppades."
                // A disconnected client cancels its owner, not the listener.
                // Recreate only that context after awaiting native teardown;
                // reuse the same process-local, source-bound numerical check.
                if flag.isCancellationRequested, !isShuttingDown {
                    await recoverAPIContext()
                }
            }
        }
        return .init(events: output.stream, cancel: { lease.cancel() })
    }
    func clearChat() {
        guard canStart else { return }
        isBusy = true; chatMessages = []; chatSummary = nil; chatProgress = nil; chatMeasurementURL = nil
        operation = Task { await worker.resetChat(); isBusy = false; operation = nil }
        saveHistory()
    }
    private var historyURL: URL { privateRoot.appendingPathComponent("quick-chat-v1.json") }
    func restoreHistory() {
        guard let attributes = try? historyURL.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey]),
              attributes.isRegularFile == true, attributes.isSymbolicLink != true,
              let size = attributes.fileSize, size <= 4 << 20,
              let data = try? Data(contentsOf: historyURL),
              let history = try? JSONDecoder().decode([REAMMessage].self, from: data), history.count <= 4096 else { return }
        chatMessages = history.map { original in
            var value = original
            if value.isThinking == true { value.interrupted = true; value.isThinking = false }
            return value
        }
        if !history.isEmpty { status = "Samtalet återställt. Efter processomstart behöver kontexten beräknas igen." }
    }
    private func saveHistory() {
        guard persistOptions, !isPreview else { return }
        do {
            let data = try JSONEncoder().encode(chatMessages)
            guard data.count <= 4 << 20 else { throw LESChatPresentationError.streamFull }
            try FileManager.default.createDirectory(at: privateRoot, withIntermediateDirectories: true)
            var directory = privateRoot
            var values = URLResourceValues(); values.isExcludedFromBackup = true
            try directory.setResourceValues(values)
            try data.write(to: historyURL, options: .atomic)
            try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: historyURL.path)
            #if os(iOS)
            try FileManager.default.setAttributes([.protectionKey: FileProtectionType.completeUntilFirstUserAuthentication], ofItemAtPath: historyURL.path)
            #endif
        } catch { self.error = "Samtalet kunde inte sparas lokalt: \(error.localizedDescription)" }
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
            let startupAdmissionReused: Bool
            let activationSeconds: Double?
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
                timestamp: Date(), startupAdmissionReused: startupAdmissionReused, activationSeconds: activationSeconds,
                numericalCheck: numericalCheck, generation: chatSummary)
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
