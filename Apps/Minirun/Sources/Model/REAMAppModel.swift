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
    init(role: Qwen35ChatMessage.Role, text: String, interrupted: Bool = false) {
        id = UUID(); self.role = role; self.text = text; self.interrupted = interrupted
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
    static let initial = REAMRunSettings(contextTokens: 1024, budgetBytes: 4_000_000_000,
        maximumNewTokens: 128, rawCacheBytes: 64 << 20)
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
    var draft = ""
    let root: URL
    let source: PinnedGGUFDownload
    let settings: REAMRunSettings
    var contextTokens: Int { settings.contextTokens }
    var budgetBytes: UInt64 { settings.budgetBytes }
    var maximumNewTokens: Int { settings.maximumNewTokens }
    @ObservationIgnored private var artifact: VerifiedGGUFArtifact?
    @ObservationIgnored private var operation: Task<Void, Never>?
    @ObservationIgnored private var cancelWorker: (() -> Void)?
    @ObservationIgnored private var cancellation: REAMCancellation?
    @ObservationIgnored private var activeID: UUID?
    @ObservationIgnored private var prepared = false
    @ObservationIgnored private let chatFile: URL?
    var isBusy: Bool { activity != .idle }
    var canRunNumerics: Bool { artifact != nil && !isBusy && !isPreview }
    var canChat: Bool { canRunNumerics && lastNumericsReport?.passedAtTestedPositions != false }
    var modelSize: String { ByteCountFormatter.string(fromByteCount: Int64(source.manifest.fileByteCount), countStyle: .file) }

    init(root: URL, source: PinnedGGUFDownload = .primaryREAM, chatFile: URL? = nil,
         settings: REAMRunSettings = .initial) {
        self.root = root; self.source = source; self.chatFile = chatFile; self.settings = settings
    }
    static func live() -> REAMAppModel {
        let manager = FileManager.default
        let documents = manager.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let support = manager.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return REAMAppModel(root: documents.appendingPathComponent("Models", isDirectory: true),
            chatFile: support.appendingPathComponent("FlashLocal", isDirectory: true).appendingPathComponent("chat.json"))
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
                messages = restored
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
        activity = .stopping; status = "Stoppar och frigör arbetsminnet…"
        cancellation?.cancel(); cancelWorker?()
    }
    nonisolated private static func preflight(_ artifact: VerifiedGGUFArtifact, settings: REAMRunSettings) throws -> VerifiedGGUFArtifact {
        try autoreleasepool {
            let layout = try Qwen4ExpDecoder.Layout(index: artifact.index)
            let tokenizer = try CanonicalQwen35Tokenizer(artifact: artifact)
            guard layout.vocabularySize == tokenizer.vocabularySize else { throw CocoaError(.fileReadCorruptFile) }
            _ = try Qwen4ExpMemoryPlan(config: layout.config, contextTokens: settings.contextTokens,
                budgetBytes: settings.budgetBytes, reservedWorkingBytes: 1_000_000_000)
            try artifact.assertUnchanged(); return artifact
        }
    }
    func verify() {
        guard !isBusy, !isPreview else { return }
        artifact = nil
        let (id, flag) = begin(.verifying, status: "Verifierar hela modellfilens SHA-256…")
        let root = root, manifest = source.manifest, settings = settings
        let worker = Task.detached(priority: .utility) {
            try Self.preflight(VerifiedGGUFArtifact.verify(insideRoot: root, manifest: manifest,
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
        let root = root, source = source, settings = settings
        let channel = AsyncStream<GGUFModelTransfer.Progress>.makeStream(bufferingPolicy: .bufferingNewest(1))
        let worker = Task.detached(priority: .utility) { () -> Result<VerifiedGGUFArtifact, Error> in
            defer { channel.continuation.finish() }
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
        let prompt = messages.map { Qwen35ChatMessage(role: $0.role, content: $0.text) }
        messages.append(.init(role: .assistant, text: "", interrupted: true))
        let responseID = messages.last!.id
        let (id, flag) = begin(.generating, status: "Förbereder lokal generering…")
        generationProgress = nil; summary = nil; lastMeasurementURL = nil; save()
        let context = contextTokens, budget = budgetBytes, limit = maximumNewTokens, cache = settings.rawCacheBytes
        let channel = AsyncThrowingStream<Qwen4ExpTextGeneration.Event, Error>.makeStream(bufferingPolicy: .bufferingOldest(32))
        channel.continuation.onTermination = { if case .cancelled = $0 { flag.cancel() } }
        let worker = Task.detached(priority: .userInitiated) { () -> Result<Qwen4ExpTextGeneration.Summary, Error> in
            let result = Result {
                try autoreleasepool {
                    let engine = try Qwen4ExpTextGeneration(artifact: artifact, contextTokens: context, budgetBytes: budget,
                        rawCacheBytes: cache)
                    var lastProgress = 0.0
                    return try engine.generate(messages: prompt, maximumNewTokens: limit,
                        cancellationRequested: { flag.requested() || Task.isCancelled }) { event in
                            if case .prefill(let p) = event {
                                let now = ProcessInfo.processInfo.systemUptime
                                guard now - lastProgress >= 0.2 || p.promptTokensCompleted == p.promptTokensTotal else { return }
                                lastProgress = now
                            }
                            if case .dropped = channel.continuation.yield(event) {
                                if case .prefill = event { return }
                                throw REAMPresentationError.streamFull
                            }
                        }
                }
            }
            switch result { case .success: channel.continuation.finish(); case .failure(let error): channel.continuation.finish(throwing: error) }
            return result
        }
        cancelWorker = { worker.cancel() }
        operation = Task {
            do {
                for try await event in channel.stream {
                    guard let index = messages.firstIndex(where: { $0.id == responseID }) else { continue }
                    switch event {
                    case .prefill(let p):
                        generationProgress = p
                        if activity != .stopping { status = "Läser prompt: \(p.promptTokensCompleted)/\(p.promptTokensTotal) token" }
                    case .token(_, let text, let p):
                        generationProgress = p; messages[index].text += text
                        if activity != .stopping { status = "Genererar lokalt…" }
                    case .trailingText(let text): messages[index].text += text
                    }
                }
            } catch { /* Worker result below is authoritative, after teardown. */ }
            switch await worker.value {
            case .success(let value):
                summary = value
                if let index = messages.firstIndex(where: { $0.id == responseID }) { messages[index].interrupted = false }
                status = "Klar · \(value.generatedTokenIDs.count) genererade token"
            case .failure(let failure):
                if !flag.requested() { error = failure.localizedDescription }
                status = "Genereringen avbröts."
            }
            save(); finish(id)
        }
    }
    func clearMessages() {
        guard !isBusy else { return }; messages.removeAll(); summary = nil; generationProgress = nil; lastMeasurementURL = nil; save()
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
        let budget = budgetBytes, cache = settings.rawCacheBytes
        let directory = root.deletingLastPathComponent().appendingPathComponent("Diagnostics", isDirectory: true)
        typealias Update = (Qwen4ExpLocalParity.Phase, Int, Int)
        let channel = AsyncStream<Update>.makeStream(bufferingPolicy: .bufferingNewest(1))
        let worker = Task.detached(priority: .userInitiated) { () -> Result<(Qwen4ExpLocalParity.Report, URL), Error> in
            defer { channel.continuation.finish() }
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
    #endif
}
