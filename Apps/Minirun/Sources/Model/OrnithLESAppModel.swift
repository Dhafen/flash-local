#if canImport(OrnithLES)
import Foundation
import Observation
import OrnithLES
import MinirunKit
import MinirunRunners
import ModelAdapters
import StorageCore

private enum LESChatPresentationError: Error, LocalizedError {
    case notReady, streamFull
    var errorDescription: String? {
        switch self {
        case .notReady: "Snabbchatten behöver aktiveras."
        case .streamFull: "Gränssnittet hann inte ta emot texten. Genereringen stoppades."
        }
    }
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
                     onCheck: @Sendable (LESNumericalCheck) -> Void) throws {
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
        chat = try OrnithLESChatGeneration(artifact: artifact, prepared: prepared, check: check,
            control: control, onAdmission: { self.admissions.append($0) })
    }
    func generateChat(messages: [Qwen35ChatMessage], thinking: Bool,
                      events: AsyncThrowingStream<Qwen4ExpTextGeneration.Event, Error>.Continuation)
        -> Result<OrnithLESChatGeneration.Summary, Error> {
        let result = Result {
            guard let chat else { throw LESChatPresentationError.notReady }
            return try autoreleasepool {
                try chat.generate(messages: messages, enableThinking: thinking) { event in
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
    var chatThinkingEnabled = true
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
    func start(existing: VerifiedGGUFArtifact? = nil) {
        guard canStart else { return }
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
                try await worker.prepareChat(modelsRoot: modelsRoot, privateRoot: privateRoot, existing: existing,
                    control: flag, progress: { [weak self] value in
                        Task { @MainActor in guard let self, self.activeID == id else { return }; self.status = value }
                    }, onCheck: { [weak self] value in
                        Task { @MainActor in guard let self, self.activeID == id else { return }; self.numericalCheck = value }
                    })
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
        chatDraft = ""; chatMessages.append(.init(role: .user, text: input))
        let prompt = REAMMessage.generationPrompt(chatMessages)
        chatMessages.append(.init(role: .assistant, text: "", interrupted: true))
        let response = chatMessages.last!.id, id = UUID(), thinking = chatThinkingEnabled
        activeID = id; control = flag; isBusy = true; error = nil
        chatProgress = nil; chatSummary = nil; chatMeasurementURL = nil; status = "Förbereder lokal generering…"
        let channel = AsyncThrowingStream<Qwen4ExpTextGeneration.Event, Error>.makeStream(bufferingPolicy: .bufferingOldest(64))
        channel.continuation.onTermination = { if case .cancelled = $0 { flag.cancel() } }
        let worker = worker
        let task = Task.detached(priority: .userInitiated) {
            let result = await worker.generateChat(messages: prompt, thinking: thinking, events: channel.continuation)
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
