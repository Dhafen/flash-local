import Foundation
import MinirunKit
import ModelAdapters
import Observation
import OrnithPerformance
import StorageCore

/// Separate ownership from the GGUF runner. Closing the sheet waits for the
/// active operation before this owner releases Metal/cache allocations.
private actor PerformanceAppRunner {
    private var engine: OrnithPerformanceEngine?
    private var session: OrnithPerformanceSession?

    func prepare(_ install: VerifiedQPackInstall, plan: QPackInstallPlan,
                 cancellation: PerformanceCancellation) async throws -> PerformanceMemoryPlan {
        session = nil; engine = nil
        let codec = try await OrnithChatCodec(localDirectory: install.directory, manifest: plan.source.manifest)
        guard !cancellation.isCancelled, !Task.isCancelled else { throw CancellationError() }
        let engine = try OrnithPerformanceEngine(install: install, plan: plan, contextTokens: 8192)
        self.engine = engine
        session = OrnithPerformanceSession(engine: engine, codec: codec)
        return await engine.memoryPlan
    }
    func generate(_ messages: [OrnithChatMessage], cancellation: PerformanceCancellation,
                  onDelta: @escaping @Sendable (String) async -> Void) async throws -> OrnithGenerationResult {
        guard let session else { throw OrnithPerformanceError.io("model is not open") }
        return try await session.generate(messages: messages, maximumNewTokens: 512,
                                          thinking: false, cancellation: cancellation, onDelta: onDelta)
    }
    func reset() async { await session?.reset() }
    func release() { session = nil; engine = nil }
}

struct PerformanceAppReport: Encodable {
    struct Shard: Encodable {
        let name: String
        let byteCount: Int
        let sha256: String
    }
    let schemaVersion = 1
    let timestamp = ISO8601DateFormatter().string(from: Date())
    let appVersion = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "unknown"
    let appBuild = Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "unknown"
    let sourceRevision = BuildInfo.gitRevision
    let inferenceBackend = "SwiftletCore Metal / local QPack"
    let modelRepository: String
    let modelRevision: String
    let sourceIdentity: String
    let sourceShards: [Shard]
    let memoryPlan: PerformanceMemoryPlan
    let generation: OrnithGenerationReport
    let operatingSystem = ProcessInfo.processInfo.operatingSystemVersionString
    let physicalMemoryBytes = ProcessInfo.processInfo.physicalMemory
    let deviceIdentifier: String? = Sysctl.string("hw.machine")
    let airplaneModeVerified = false
    let fullModelNumericsVerified = false
    #if targetEnvironment(simulator)
    let platform = "iOSSimulator"
    #elseif os(iOS)
    let platform = "iOSDevice"
    #else
    let platform = "macOS"
    #endif
}

@Observable @MainActor
final class OrnithPerformanceAppModel {
    enum Activity { case idle, installing, opening, generating, stopping }
    private(set) var activity = Activity.idle
    private(set) var status = "Ornith QPack är inte installerad."
    private(set) var error: String?
    private(set) var messages: [REAMMessage] = []
    private(set) var progress: QPackInstallProgress?
    private(set) var memoryPlan: PerformanceMemoryPlan?
    private(set) var report: OrnithGenerationReport?
    private(set) var isReady = false
    private(set) var hasInstalledFiles = false
    private(set) var isPreview = false
    private(set) var isShuttingDown = false
    private(set) var lastMeasurementURL: URL?
    var draft = ""
    let privateRoot: URL
    let documents: URL
    @ObservationIgnored private let runner = PerformanceAppRunner()
    @ObservationIgnored private var plan: QPackInstallPlan?
    @ObservationIgnored private var reportMemoryPlan: PerformanceMemoryPlan?
    @ObservationIgnored private var operation: Task<Void, Never>?
    @ObservationIgnored private var shutdownOperation: Task<Void, Never>?
    @ObservationIgnored private var cancellation: PerformanceCancellation?
    @ObservationIgnored private var activeID: UUID?
    @ObservationIgnored private var prepared = false
    var isBusy: Bool { activity != .idle || isShuttingDown }
    var canSend: Bool { isReady && !isBusy && !isPreview }
    var expectedFileBytes: Int { plan?.source.manifest.shards.reduce(0) { $0 + $1.byteCount } ?? 19_509_024_201 }
    private var chatFile: URL { privateRoot.appendingPathComponent("qpack-chat.json") }

    init(privateRoot: URL, documents: URL) { self.privateRoot = privateRoot; self.documents = documents }
    static func live() -> Self {
        let fm = FileManager.default
        return Self(privateRoot: fm.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("FlashLocal/OrnithPerformance", isDirectory: true),
            documents: fm.urls(for: .documentDirectory, in: .userDomainMask)[0])
    }
    func prepare() {
        guard !prepared, !isPreview else { return }
        prepared = true
        do {
            try FileManager.default.createDirectory(at: privateRoot, withIntermediateDirectories: true)
            var root = privateRoot
            var attributes = URLResourceValues(); attributes.isExcludedFromBackup = true
            try root.setResourceValues(attributes)
            plan = try QPackInstallPlan(source: MLXSourceIndex(manifest: MLXSourceManifest.ornith()))
            if let plan {
                hasInstalledFiles = FileManager.default.fileExists(atPath: privateRoot
                    .appendingPathComponent("ornith-\(plan.source.manifest.revision.prefix(12)).qpack").path)
            }
            if FileManager.default.fileExists(atPath: chatFile.path) {
                let file = try FileHandle(forReadingFrom: chatFile)
                defer { try? file.close() }
                let bytes = try file.read(upToCount: (4 << 20) + 1) ?? Data()
                guard bytes.count <= 4 << 20 else { throw CocoaError(.fileReadTooLarge) }
                let history = try JSONDecoder().decode([REAMMessage].self, from: bytes)
                guard history.count <= 4096 else { throw CocoaError(.fileReadTooLarge) }
                messages = history
            }
            if hasInstalledFiles { status = "Installerad modell finns. Öppna för att verifiera och chatta." }
        } catch { self.error = error.localizedDescription }
    }
    func installAndOpen() {
        guard !isBusy, !isPreview, let plan else { return }
        let id = UUID(), control = PerformanceCancellation()
        activeID = id; cancellation = control; activity = .installing
        error = nil; progress = nil; isReady = false; memoryPlan = nil
        status = hasInstalledFiles ? "Verifierar installerade modellfiler…" : "Hämtar och installerar Ornith…"
        operation = Task {
            do {
                await runner.release()
                let installer = try QPackInstaller(plan: plan, privateRoot: privateRoot,
                    source: HTTPQPackByteSource(manifest: plan.source.manifest))
                let installed = try await installer.install { [weak self] update in
                    Task { @MainActor in
                        guard let self, self.activeID == id, self.activity != .stopping else { return }
                        self.progress = update
                        self.status = update.phase == .downloading ? "Hämtar och installerar Ornith…" : "Verifierar modellens käll- och filhashar…"
                    }
                }
                hasInstalledFiles = true
                try Task.checkCancellation()
                guard !control.isCancelled else { throw CancellationError() }
                activity = .opening; status = "Öppnar lokal modell och planerar RAM…"
                memoryPlan = try await runner.prepare(installed, plan: plan, cancellation: control)
                try Task.checkCancellation()
                guard !control.isCancelled else { throw CancellationError() }
                isReady = true; status = "Redo · lokal Metal-inferens · 8 192 token kontextgräns."
            } catch {
                await runner.release()
                if control.isCancelled || Task.isCancelled { status = "Pausad. Hämtningen kan återupptas." }
                else { self.error = error.localizedDescription; status = "Modellen kunde inte öppnas." }
            }
            finish(id)
        }
    }
    func send() {
        let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard canSend, !text.isEmpty else { return }
        let id = UUID(), control = PerformanceCancellation()
        let runMemoryPlan = memoryPlan
        activeID = id; cancellation = control; activity = .generating
        draft = ""; error = nil; report = nil; lastMeasurementURL = nil
        messages.append(REAMMessage(role: .user, text: text))
        let prompt = REAMMessage.generationPrompt(messages).map {
            OrnithChatMessage(role: $0.role == .user ? .user : .assistant, content: $0.content)
        }
        messages.append(REAMMessage(role: .assistant, text: "", interrupted: true))
        let replyID = messages.last!.id
        status = "Läser prompten lokalt…"
        saveHistory()
        operation = Task {
            do {
                let result = try await runner.generate(prompt, cancellation: control) { [weak self] delta in
                    await self?.append(delta, to: replyID, operationID: id)
                }
                if let index = messages.firstIndex(where: { $0.id == replyID }) {
                    messages[index].text = result.text
                    messages[index].interrupted = result.report.finishReason == .cancelled
                }
                report = result.report
                reportMemoryPlan = runMemoryPlan
                status = result.report.finishReason == .cancelled ? "Genereringen avbröts." : "Klar · \(result.report.generatedTokens) genererade token."
            } catch {
                self.error = error.localizedDescription
                status = "Genereringen avbröts."
                await runner.release(); isReady = false
            }
            saveHistory(); finish(id)
        }
    }
    private func append(_ delta: String, to replyID: UUID, operationID: UUID) {
        guard activeID == operationID, let index = messages.firstIndex(where: { $0.id == replyID }) else { return }
        messages[index].text += delta
        if activity != .stopping { status = "Genererar lokalt…" }
    }
    func stop() {
        guard isBusy else { return }
        activity = .stopping; status = "Stoppar…"
        cancellation?.cancel()
        operation?.cancel()
    }
    func shutdown() async {
        if let shutdownOperation { await shutdownOperation.value; return }
        isShuttingDown = true
        stop()
        let pending = operation, runner = runner
        let teardown = Task { await pending?.value; await runner.release() }
        shutdownOperation = teardown
        await teardown.value
        isReady = false; memoryPlan = nil
        if hasInstalledFiles { status = "Installerad modell finns. Öppna för att chatta." }
        shutdownOperation = nil; isShuttingDown = false
    }
    func clearMessages() {
        guard !isBusy, !isPreview else { return }
        messages = []; report = nil; reportMemoryPlan = nil; lastMeasurementURL = nil
        saveHistory()
        activity = .opening
        operation = Task { await runner.reset(); activity = .idle; operation = nil }
    }
    func saveMeasurement() {
        guard !isBusy, let report, let plan, let memoryPlan = reportMemoryPlan else { return }
        do {
            let diagnostics = documents.appendingPathComponent("Diagnostics", isDirectory: true)
            try FileManager.default.createDirectory(at: diagnostics, withIntermediateDirectories: true)
            let source = plan.source.manifest
            let value = PerformanceAppReport(modelRepository: source.repository, modelRevision: source.revision,
                sourceIdentity: try source.identity(), sourceShards: source.shards.map {
                    .init(name: $0.name, byteCount: $0.byteCount, sha256: $0.sha256)
                }, memoryPlan: memoryPlan, generation: report)
            let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let file = diagnostics.appendingPathComponent("ornith-qpack-\(UUID().uuidString).json")
            try encoder.encode(value).write(to: file, options: .atomic)
            lastMeasurementURL = file; status = "Mätning sparad i Filer › Flash Local › Diagnostics."
        } catch { self.error = error.localizedDescription }
    }
    private func finish(_ id: UUID) {
        guard activeID == id else { return }
        activeID = nil; cancellation = nil; operation = nil; activity = .idle
    }
    private func saveHistory() {
        do { try JSONEncoder().encode(messages).write(to: chatFile, options: .atomic) }
        catch { self.error = error.localizedDescription }
    }
    static func reviewPreview() -> OrnithPerformanceAppModel {
        let model = OrnithPerformanceAppModel(privateRoot: FileManager.default.temporaryDirectory,
                                             documents: FileManager.default.temporaryDirectory)
        model.isPreview = true
        model.status = "Förhandsvisning · ingen inferens eller uppmätt prestanda."
        model.messages = [REAMMessage(role: .user, text: "Vad är 50+5?"),
                          REAMMessage(role: .assistant, text: "55")]
        return model
    }
}
