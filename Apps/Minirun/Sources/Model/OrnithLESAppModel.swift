#if canImport(OrnithLES)
import Foundation
import Observation
import OrnithLES
import MinirunKit
import StorageCore

private struct LESRunExport: Encodable {
    let measurement: LESBenchmarkRun
    let decodeForwardTokensPerSecond: Double
    let prefillTokensPerSecond: Double
    let decodeIOBytesPerForward: Double
    let hasRequired128Tokens: Bool
    let metalDecodeNanoseconds: UInt64?
    let metalDecodeDispatches: UInt64?
    let expertCacheHitRate: Double?
    init(_ run: LESBenchmarkRun) {
        measurement = run; decodeForwardTokensPerSecond = run.decodeTokensPerSecond
        prefillTokensPerSecond = run.prefillTokensPerSecond
        decodeIOBytesPerForward = Double(run.decodeIOBytes) / Double(max(1, run.decodeForwardPasses))
        hasRequired128Tokens = run.hasRequiredLength
        if let before = run.metricsAfterPrefill, let after = run.metricsAfter {
            metalDecodeNanoseconds = after.metalNanoseconds - before.metalNanoseconds
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
    let fullModelNumericsVerified = false
    let promptID = LESBenchmark.promptID
    let contextCapacityTokens = 4096
    let numericalCheck: LESNumericalCheck?
    let measurements: [LESRunExport]
    let failure: String?
    let maximumObservedFootprintBytes: UInt64
    let modelSHA256 = GGUFModelManifest.primaryOrnith.sha256
    let nativeRevision = LESContext.revision
}
private actor LESAppWorker {
    private var artifact: VerifiedGGUFArtifact?
    private var prepared: URL?
    private var check: LESNumericalCheck?
    func release() { artifact = nil; prepared = nil }
    func execute(modelsRoot: URL, privateRoot: URL, existing: VerifiedGGUFArtifact?, control: LESControl,
                 progress: @Sendable (String) -> Void,
                 onCheck: @Sendable (LESNumericalCheck) -> Void,
                 onRun: @Sendable (LESBenchmarkRun) -> Void) throws -> (LESNumericalCheck, [LESBenchmarkRun]) {
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
            prepared = try LESPreparation.prepare(artifact: artifact, privateRoot: privateRoot, control: control)
        }
        guard let prepared else { throw CancellationError() }
        if check == nil {
            check = try LESBenchmark.check(artifact: artifact, prepared: prepared, control: control, progress: progress)
        }
        guard let check else { throw CancellationError() }
        onCheck(check)
        guard check.passed else { return (check, []) }
        var runs: [LESBenchmarkRun] = []
        for backend in LESBenchmarkBackend.allCases {
            let pair = try autoreleasepool {
                try LESBenchmark.runPair(backend: backend, artifact: artifact, prepared: prepared,
                    check: check, control: control, progress: progress)
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
    private(set) var measurementURL: URL?
    var userDeclaredAirplaneMode = false
    let modelsRoot: URL
    let privateRoot: URL
    let documents: URL
    @ObservationIgnored private let worker = LESAppWorker()
    @ObservationIgnored private var operation: Task<Void, Never>?
    @ObservationIgnored private var control: LESControl?
    @ObservationIgnored private var activeID: UUID?
    var canStart: Bool { !isBusy && !isShuttingDown && !isPreview }
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
    func start(existing: VerifiedGGUFArtifact? = nil) {
        guard canStart else { return }
        let flag = LESControl(), id = UUID()
        activeID = id; control = flag; isBusy = true; error = nil; runs = []; measurementURL = nil
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
                numericalCheck = result.0; runs = result.1
                status = !result.0.passed ? "Numeriken avvek. LES-testet stoppades; den vanliga chatten är oförändrad." :
                    runs.allSatisfy(\.hasRequiredLength) ? "Jämförelsen är klar." : "Något svar blev kortare än 128 token. Den körningen är inte jämförbar."
                try save(check: result.0, runs: result.1, airplane: declaredAirplane)
            } catch {
                guard activeID == id else { return }
                self.error = error.localizedDescription; status = "Testet stoppades."
                try? save(check: numericalCheck, runs: runs, airplane: declaredAirplane, failure: error.localizedDescription)
            }
        }
    }
    func stop() { guard isBusy else { return }; status = "Stoppar och frigör motorn…"; control?.cancel() }
    func shutdown() async {
        guard !isShuttingDown else { await operation?.value; return }
        isShuttingDown = true
        stop(); await operation?.value; await worker.release(); isShuttingDown = false
    }
    private func save(check: LESNumericalCheck?, runs: [LESBenchmarkRun], airplane: Bool, failure: String? = nil) throws {
        #if os(iOS)
        let platform = "iOSDevice"
        #else
        let platform = "macOS"
        #endif
        let record = LESAppReport(appVersion: Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "unknown",
            appBuild: Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "unknown",
            sourceRevision: BuildInfo.gitRevision, timestamp: Date(), platform: platform, deviceIdentifier: Sysctl.string("hw.machine"),
            physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory, operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString,
            lowPowerModeEnabled: ProcessInfo.processInfo.isLowPowerModeEnabled, userDeclaredAirplaneMode: airplane,
            numericalCheck: check, measurements: runs.map(LESRunExport.init), failure: failure,
            maximumObservedFootprintBytes: control?.maximumObservedFootprintBytes ?? 0)
        let folder = documents.appendingPathComponent("Diagnostics")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let url = folder.appendingPathComponent("ornith-les-" + UUID().uuidString + ".json")
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]; encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(record).write(to: url, options: .atomic); measurementURL = url
    }
}
#endif
