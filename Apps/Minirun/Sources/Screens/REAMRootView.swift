import MinirunKit
import SwiftUI
#if os(iOS)
import UIKit
import SafariServices
#endif

struct REAMRootView: View {
    @Bindable var model: REAMAppModel
    @Environment(\.scenePhase) private var scenePhase
    private enum Tab: String, Hashable {
        case chats, models, api, settings
    }
    @State private var selectedTab: Tab =
        Tab(rawValue: UserDefaults.standard.string(forKey: "FlashLocal.selectedTab.v1") ?? "") ?? .chats
    @State private var showingDiagnostics = false
    @State private var performanceModel = OrnithPerformanceAppModel.live()

    #if canImport(OrnithLES)
    private var lesModel: OrnithLESAppModel { model.ornithSession }
    private enum Runner: String, CaseIterable, Identifiable {
        case ornith, ream
        var id: String { rawValue }
        var title: String {
            switch self {
            case .ornith: "Ornith 1.5 · 35B-A3B"
            case .ream: "Qwen3.8 · REAM"
            }
        }
    }
    @State private var activeRunner: Runner =
        Runner(rawValue: UserDefaults.standard.string(forKey: "FlashLocal.activeInferenceRunner") ?? "") ?? .ornith
    #endif

    var body: some View {
        TabView(selection: $selectedTab) {
            NavigationStack {
                chatContent
                    .navigationTitle("Chattar")
            }
            .tabItem { Label("Chattar", systemImage: "bubble.left.and.bubble.right") }
            .tag(Tab.chats)
            .accessibilityIdentifier("flash-tab-chats")

            HuggingFaceGGUFBrowser()
                .tabItem { Label("Modeller", systemImage: "square.stack.3d.up") }
                .tag(Tab.models)
                .accessibilityIdentifier("flash-tab-models")

            NavigationStack {
                apiContent
                    .navigationTitle("API")
            }
            .tabItem { Label("API", systemImage: "network") }
            .tag(Tab.api)
            .accessibilityIdentifier("flash-tab-api")

            NavigationStack {
                settingsContent
                    .navigationTitle("Inställningar")
            }
            .tabItem { Label("Inställningar", systemImage: "gearshape") }
            .tag(Tab.settings)
            .accessibilityIdentifier("flash-tab-settings")
        }
        .tint(MRColor.tierPinned)
        .onChange(of: selectedTab) { _, value in
            UserDefaults.standard.set(value.rawValue, forKey: "FlashLocal.selectedTab.v1")
        }
        #if canImport(OrnithLES)
        .task {
            if activeRunner == .ream { model.activateForChatModel(.ream) }
        }
        .onChange(of: scenePhase) { _, phase in
            switch activeRunner {
            case .ream:
                if phase == .background { model.releaseInferenceResources() }
                else if phase == .active { model.refreshStorage() }
            case .ornith:
                if phase == .background { Task { await lesModel.enterBackground() } }
                else if phase == .active { lesModel.enterForeground() }
                else if phase == .inactive { lesModel.prepareForInactivity() }
            }
        }
        #else
        .task { model.prepare() }
        .onChange(of: scenePhase) { _, phase in
            if phase == .background {
                model.stopAPI()
                model.releaseInferenceResources()
            } else if phase == .active {
                model.refreshStorage()
            }
        }
        #endif
    }

    @ViewBuilder private var chatContent: some View {
        #if canImport(OrnithLES)
        if model.isSwitchingInferenceModel {
            ProgressView("Frigör föregående modell…")
        } else if activeRunner == .ornith {
            OrnithLESChatPage(model: lesModel,
                activate: { lesModel.activateChat(existing: model.verifiedOrnithForAlternateRunner()) })
        } else {
            REAMChatPage(model: model)
        }
        #else
        REAMChatPage(model: model)
        #endif
    }

    @ViewBuilder private var apiContent: some View {
        #if canImport(OrnithLES)
        if activeRunner == .ornith {
            OrnithAPIPage(model: lesModel,
                activate: { lesModel.activateChat(existing: model.verifiedOrnithForAlternateRunner()) })
        } else {
            REAMLocalAPIPage(model: model)
        }
        #else
        REAMLocalAPIPage(model: model)
        #endif
    }

    private var settingsContent: some View {
        Form {
            Section("Aktiv modell") {
                #if canImport(OrnithLES)
                Picker("Körmodell", selection: Binding(
                    get: { activeRunner },
                    set: { selectRunner($0) }
                )) {
                    ForEach(Runner.allCases) { runner in Text(runner.title).tag(runner) }
                }
                .disabled(model.isSwitchingInferenceModel)
                .accessibilityIdentifier("iphone-model-selector")
                #else
                Picker("Körmodell", selection: Binding(
                    get: { model.selectedModel },
                    set: { model.selectModel($0) }
                )) {
                    ForEach(LocalChatModel.allCases) { entry in
                        Text(entry.title).tag(entry)
                    }
                }
                .disabled(!model.canSelectModel)
                .accessibilityIdentifier("iphone-model-selector")
                #endif
                Text("Community-GGUF visas i Modeller. Nedladdning innebär inte att motorn stöder modellens arkitektur.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Lokal inferens") {
                LabeledContent("Vald modell", value: model.modelTitle)
                LabeledContent("Kontext i standardmotorn", value: "\(model.contextTokens.formatted()) token")
                LabeledContent("RAM-gräns", value: String(format: "%.1f GB", Double(model.budgetBytes) / 1_000_000_000))
                if model.supportsThinking {
                    Toggle("Tänkande", isOn: Binding(
                        get: { model.thinkingEnabled },
                        set: { model.setThinkingEnabled($0) }
                    ))
                }
                #if canImport(OrnithLES)
                if activeRunner == .ornith {
                    Button("Avancerade Ornith-kontroller och jämförelser") {
                        showingDiagnostics = true
                    }
                }
                #endif
            }
            Section("Information") {
                Text("Chatt, modellbibliotek och API har egna flikar och behålls synliga även utan experimentella ramverk.")
                    .font(.caption).foregroundStyle(.secondary)
                Text("API kan bara svara när en kompatibel lokal motor är verifierad och appen aktiv. iOS kan pausa appen i bakgrunden.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        #if canImport(OrnithLES)
        .sheet(isPresented: $showingDiagnostics) {
            OrnithLESView(model: lesModel, existing: model.verifiedOrnithForAlternateRunner())
        }
        #endif
    }

    #if canImport(OrnithLES)
    @MainActor
    private func selectRunner(_ target: Runner) {
        guard !model.isSwitchingInferenceModel, target != activeRunner else { return }
        model.isSwitchingInferenceModel = true
        Task { @MainActor in
            switch activeRunner {
            case .ornith: await lesModel.shutdown()
            case .ream: await model.pauseForAlternateRunner()
            }
            if target == .ream { model.activateForChatModel(.ream) }
            activeRunner = target
            UserDefaults.standard.set(target.rawValue, forKey: "FlashLocal.activeInferenceRunner")
            model.isSwitchingInferenceModel = false
        }
    }
    #endif
}

#if os(iOS)
/// Only exact, SHA-256-pinned GGUF files are admitted by the current engines.
/// Other files in the same repositories are not automatically compatible.
private struct HuggingFaceGGUFBrowser: View {
    @State private var selectedPage: Page?
    @State private var searchQuery = "dolphin"
    @State private var searchResults: [CommunityGGUFRepository] = []
    @State private var searchingModels = false
    @State private var repository = "dphn/Dolphin3.0-Llama3.1-8B-GGUF"
    @State private var availableFiles: [CommunityGGUFFile] = []
    @State private var installedFiles: [CommunityGGUFFile] = []
    @State private var isSearching = false
    @State private var downloadingID: String?
    @State private var completedFraction: Double = 0
    @State private var communityStatus = ""
    @State private var communityError: String?
    @State private var downloadTask: Task<Void, Never>?

    private var communityRoot: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("FlashLocal/CommunityGGUF", isDirectory: true)
    }

    private func refreshLibrary() {
        installedFiles = CommunityGGUFCatalog.installed(in: communityRoot)
    }

    private func searchCatalog() {
        guard !searchingModels, !isSearching, downloadingID == nil else { return }
        communityError = nil; communityStatus = ""; searchResults = []
        searchingModels = true
        Task { @MainActor in
            defer { searchingModels = false }
            do {
                searchResults = try await CommunityGGUFCatalog.searchModels(query: searchQuery)
                communityStatus = searchResults.isEmpty ? "Inga matchande GGUF-repositories." :
                    "\(searchResults.count) modellrepositorier hittades. Välj ett för att se GGUF-filerna."
            } catch { communityError = error.localizedDescription }
        }
    }

    private func normalizedRepository(_ input: String) -> String {
        let trimmed = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let url = URLComponents(string: trimmed), url.scheme == "https",
              url.host == "huggingface.co", url.query == nil, url.fragment == nil else { return trimmed }
        let parts = url.path.split(separator: "/")
        guard parts.count >= 2 else { return trimmed }
        return String(parts[0]) + "/" + String(parts[1])
    }

    private func discover() {
        guard !isSearching, downloadingID == nil else { return }
        let name = normalizedRepository(repository)
        communityError = nil; communityStatus = ""; availableFiles = []; isSearching = true
        Task { @MainActor in
            defer { isSearching = false }
            do {
                availableFiles = try await CommunityGGUFCatalog.discover(repository: name)
                communityStatus = availableFiles.isEmpty
                    ? "Inga GGUF-filer med verifierbart LFS-SHA256 hittades."
                    : "\(availableFiles.count) verifierbara GGUF-filer hittades."
            } catch {
                communityError = error.localizedDescription
            }
        }
    }

    private func download(_ entry: CommunityGGUFFile) {
        guard downloadingID == nil, !isSearching else { return }
        communityError = nil; communityStatus = "Förbereder nedladdning…"
        completedFraction = 0; downloadingID = entry.id
        let root = communityRoot
        downloadTask = Task { @MainActor in
            defer { downloadingID = nil; downloadTask = nil; refreshLibrary() }
            do {
                let source = try entry.pinnedSource()
                try LocalModelStorage.prepare(root: root)
                // Recover from a completed download whose index write was
                // interrupted: always re-verify before recording the entry.
                let fileURL = root.appendingPathComponent(entry.installedName)
                let verified: VerifiedGGUFArtifact
                if FileManager.default.fileExists(atPath: fileURL.path) {
                    verified = try await Task.detached(priority: .utility) {
                        try VerifiedGGUFArtifact.verify(insideRoot: root, manifest: source.manifest)
                    }.value
                } else {
                    let transfer = try GGUFModelTransfer()
                    verified = try await transfer.download(insideRoot: root, source: source,
                        onProgress: { progress in
                            Task { @MainActor in
                                guard downloadingID == entry.id else { return }
                                completedFraction = progress.totalBytes == 0 ? 0 :
                                    Double(progress.completedBytes) / Double(progress.totalBytes)
                                communityStatus = progress.phase == .verifying
                                    ? "Verifierar SHA-256 och GGUF…"
                                    : "Hämtar \(entry.name)…"
                            }
                        })
                }
                try Task.checkCancellation()
                try CommunityGGUFCatalog.recordVerified(entry, artifact: verified, in: root)
                completedFraction = 1
                communityStatus = "\(entry.name) är nedladdad och verifierad. Körning kräver en kompatibel motor."
            } catch is CancellationError {
                communityStatus = "Hämtningen avbruten. Delvis nedladdade filer kan återupptas."
            } catch {
                communityError = error.localizedDescription
            }
        }
    }

    private struct Page: Identifiable {
        let id: String
        let url: URL
    }
    private struct ModelLink: Identifiable {
        let id: String
        let title: String
        let fileName: String
        let downloadURL: URL
    }
    private var admitted: [ModelLink] {
        [
            .init(id: "ornith", title: "Ornith 1.5 · 35B-A3B",
                  fileName: PinnedGGUFDownload.primaryOrnith.manifest.fileName,
                  downloadURL: PinnedGGUFDownload.primaryOrnith.url),
            .init(id: "ream", title: "Qwen3.8 · REAM",
                  fileName: PinnedGGUFDownload.primaryREAM.manifest.fileName,
                  downloadURL: PinnedGGUFDownload.primaryREAM.url)
        ]
    }
    // Derive repository pages from the very same pinned URLs used by the
    // downloader, so the catalog cannot silently point at a different repo.
    private func repositoryPage(_ downloadURL: URL) -> URL {
        guard let marker = downloadURL.path.range(of: "/resolve/"),
              var components = URLComponents(url: downloadURL, resolvingAgainstBaseURL: false) else {
            return downloadURL
        }
        components.path = String(downloadURL.path[..<marker.lowerBound])
        components.query = nil
        components.fragment = nil
        return components.url ?? downloadURL
    }

    var body: some View {
        NavigationStack {
            List {
                Section("Fasta modellfiler i appen") {
                    ForEach(admitted) { entry in
                        Button {
                            selectedPage = .init(id: entry.id, url: repositoryPage(entry.downloadURL))
                        } label: {
                            VStack(alignment: .leading, spacing: 5) {
                                Text(entry.title).font(.headline)
                                Text(entry.fileName).font(.caption.monospaced())
                                    .foregroundStyle(.secondary)
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .accessibilityIdentifier("hf-pinned-\(entry.id)")
                    }
                }
                Section("Sök modeller på Hugging Face") {
                    TextField("Sök namn, utvecklare eller modellfamilj", text: $searchQuery)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .submitLabel(.search)
                        .onSubmit { searchCatalog() }
                        .accessibilityIdentifier("hf-model-search-input")
                    Button("Sök bland modeller", systemImage: "magnifyingglass") {
                        searchCatalog()
                    }
                    .disabled(searchingModels || isSearching || downloadingID != nil)
                    .accessibilityIdentifier("hf-model-search-button")
                    if searchingModels { ProgressView("Söker Hugging Face…") }
                    ForEach(searchResults) { result in
                        Button {
                            repository = result.id
                            discover()
                        } label: {
                            VStack(alignment: .leading, spacing: 5) {
                                Text(result.name).font(.headline)
                                Text(result.author).font(.caption).foregroundStyle(.secondary)
                                Text("\(result.downloads.formatted()) hämtningar")
                                    .font(.caption2).foregroundStyle(.secondary)
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .disabled(isSearching || downloadingID != nil)
                    }
                }
                Section("Välj GGUF-filer från ett repository") {
                    TextField("ägare/modell eller Hugging Face-länk", text: $repository)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .accessibilityIdentifier("hf-community-repository")
                    Button("Sök GGUF-filer") { discover() }
                        .disabled(isSearching || downloadingID != nil)
                        .accessibilityIdentifier("hf-community-search")
                    if isSearching { ProgressView("Läser verifierbara GGUF-filer…") }
                    ForEach(availableFiles) { entry in
                        VStack(alignment: .leading, spacing: 6) {
                            Text(entry.path).font(.subheadline).textSelection(.enabled)
                            Text(ByteCountFormatter.string(fromByteCount: Int64(entry.byteCount), countStyle: .file))
                                .font(.caption).foregroundStyle(.secondary)
                            if installedFiles.contains(where: { $0.sha256 == entry.sha256 }) {
                                Label("Nedladdad", systemImage: "checkmark.circle")
                                    .font(.caption).foregroundStyle(.secondary)
                            } else {
                                Button("Hämta den här GGUF-filen") { download(entry) }
                                    .disabled(downloadingID != nil || isSearching)
                            }
                        }
                    }
                    if downloadingID != nil {
                        ProgressView(value: completedFraction)
                        Button("Avbryt nedladdning", role: .cancel) { downloadTask?.cancel() }
                    }
                    if !communityStatus.isEmpty {
                        Text(communityStatus).font(.caption).foregroundStyle(.secondary)
                    }
                    if let communityError {
                        Text(communityError).font(.caption).foregroundStyle(.red)
                            .textSelection(.enabled)
                    }
                    Text("Filerna hämtas till intern lagring med återupptagbara 8 MiB-intervall och kontrollerad SHA-256. En nedladdad fil är inte automatiskt körbar. Endast kompatibla modeller kan aktiveras i en separat verifierad inferensmotor.")
                        .font(.caption).foregroundStyle(.secondary)
                    Button("Utforska fler på Hugging Face") {
                        selectedPage = .init(id: "explore",
                            url: URL(string: "https://huggingface.co/models?library=gguf")!)
                    }
                }
                if !installedFiles.isEmpty {
                    Section("Nedladdade community-modeller") {
                        ForEach(installedFiles) { entry in
                            VStack(alignment: .leading, spacing: 4) {
                                Text(entry.name).font(.subheadline)
                                Text(entry.repository).font(.caption).foregroundStyle(.secondary)
                            }
                        }
                    }
                }
            }
            .navigationTitle("Hugging Face")
            .onAppear { refreshLibrary() }
            .sheet(item: $selectedPage) { page in
                HuggingFaceSafariPage(url: page.url).ignoresSafeArea()
            }
        }
    }
}

private struct HuggingFaceSafariPage: UIViewControllerRepresentable {
    let url: URL
    func makeUIViewController(context: Context) -> SFSafariViewController {
        SFSafariViewController(url: url)
    }
    func updateUIViewController(_ controller: SFSafariViewController, context: Context) {}
}
#else
private struct HuggingFaceGGUFBrowser: View {
    var body: some View {
        VStack(spacing: 16) {
            Text("Modellbibliotek").font(.title2.bold())
            Link("Utforska GGUF på Hugging Face",
                 destination: URL(string: "https://huggingface.co/models?library=gguf")!)
            Text("Lokal modellimport och nedladdning från biblioteket sker i iPhone-appen.")
                .font(.caption).foregroundStyle(.secondary)
        }.padding()
    }
}
#endif

/// Pure page content, separable from navigation/lifecycle for light/dark QA.
struct REAMChatPage: View {
    @Bindable var model: REAMAppModel
    var discard: () -> Void = {}
    var openPerformance: (() -> Void)? = nil
    var openLES: (() -> Void)? = nil
    var body: some View {
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(spacing: 0) {
                    if let openLES {
                        Button("Öppna Ornith snabbchatt", systemImage: "speedometer", action: openLES)
                            .buttonStyle(.borderedProminent).disabled(!model.canOpenAlternateRunner)
                            .accessibilityIdentifier("open-ornith-les").padding(.bottom, 12)
                    }
                    if let openPerformance {
                        Button(action: openPerformance) {
                            HStack(spacing: 12) {
                                Image(systemName: "cpu").font(.title2)
                                VStack(alignment: .leading, spacing: 4) {
                                    Text("Ornith Performance").font(.headline)
                                    Text("Öppna modell och optimeringskontroll").font(.caption)
                                }
                                Spacer()
                                Image(systemName: "chevron.right")
                            }.frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(!model.canOpenAlternateRunner)
                        .accessibilityIdentifier("open-ornith-performance")
                        .padding(.bottom, 16)
                    }
                    modelCard
                    Divider()
                    LazyVStack(alignment: .leading, spacing: 20) {
                        if model.messages.isEmpty {
                            VStack(alignment: .leading, spacing: 12) {
                                Image(systemName: "internaldrive").font(.largeTitle).foregroundStyle(MRColor.tierPinned)
                                Text("Din modell. På din iPhone.").font(.title2.bold())
                                Text("Hämta eller lägg modellfilen i Filer › På min iPhone › Flash Local › Models. Verifiera filen innan du chattar.")
                                    .foregroundStyle(.secondary)
                            }.padding(.vertical, 28)
                        }
                        ForEach(model.messages) { message in
                            REAMMessageRow(message: message, isGenerating: model.isGenerating(message))
                                .id(message.id)
                        }
                        Color.clear.frame(height: 1).id("latest")
                    }.padding(20)
                    }
                }
                .onChange(of: model.messages.last?.text) { _, _ in proxy.scrollTo("latest", anchor: .bottom) }
                .onChange(of: model.messages.last?.reasoning) { _, _ in proxy.scrollTo("latest", anchor: .bottom) }
            }
            Divider()
            composer
        }
        .background(MRColor.abyss)
        .tint(MRColor.tierPinned)
    }

    private var modelCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(model.modelTitle).font(.subheadline.bold())
                    Text("\(model.modelSize) · intern lagring · \(String(format: "%.1f", Double(model.budgetBytes) / 1_000_000_000)) GB RAM-gräns")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Spacer(minLength: 8)
                if model.canChat { Image(systemName: "checkmark.shield.fill").foregroundStyle(MRColor.tierPinned).accessibilityLabel("Modellfil verifierad") }
            }
            Text(model.status).font(.caption).accessibilityIdentifier("ream-status")
            if model.supportsThinking {
                Toggle("Tänkande", isOn: Binding(get: { model.thinkingEnabled }, set: model.setThinkingEnabled))
                    .font(.subheadline).accessibilityIdentifier("ornith-thinking-toggle")
                    .disabled(model.isBusy || model.isPreview || model.alternateRunnerActive)
                Text("Tänkandet använder en del av svarets \(model.maximumNewTokens) token.")
                    .font(.caption2).foregroundStyle(.secondary)
            }
            if model.isPreview { Text("Förhandsvisning av layout – ingen inferens.").font(.caption.bold()) }
            if let error = model.error {
                Text(error).font(.caption).foregroundStyle(.red).textSelection(.enabled)
            }
            if model.isBusy {
                HStack {
                    if let progress = model.progress, progress.phase == .downloading, progress.totalBytes > 0 {
                        ProgressView(value: Double(progress.completedBytes), total: Double(progress.totalBytes))
                        Text(ByteCountFormatter.string(fromByteCount: Int64(progress.completedBytes), countStyle: .file)).font(.caption.monospacedDigit())
                    } else { ProgressView().controlSize(.small) }
                    Spacer()
                    Button("Stoppa", role: .cancel) { model.stop() }.disabled(model.activity == .stopping)
                }
            } else if !model.canChat {
                HStack {
                    if model.hasFile {
                        Button("Verifiera modellfil") { model.verify() }.buttonStyle(.borderedProminent)
                    } else {
                        Button(model.partialBytes > 0 ? "Återuppta hämtning" : "Hämta modell") { model.download() }
                            .buttonStyle(.borderedProminent)
                    }
                    if model.partialBytes > 0 { Button("Börja om", role: .destructive, action: discard) }
                }.disabled(model.isPreview)
                Text("Wi-Fi rekommenderas. Modellen ingår inte i appen och raderas om appen avinstalleras.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if let p = model.generationProgress {
                Text("RAM \(ram(p.footprintBytes)) · läst totalt \(ByteCountFormatter.string(fromByteCount: Int64(p.ioBytesRead), countStyle: .file)) · \(p.generatedTokens) token")
                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                Text("Filsumman räknar även när samma vikter läses igen.")
                    .font(.caption2).foregroundStyle(.secondary)
            }
            if let summary = model.summary, let speed = summary.decodeTokensPerSecond {
                Text(String(format: "%.2f token/s · observerad RAM-topp %@", speed,
                    ram(summary.maximumObservedFootprintBytes)))
                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            }
            if model.summary != nil {
                Button("Spara mätning i Filer", systemImage: "chart.bar.doc.horizontal") { model.saveMeasurement() }
                    .disabled(model.isBusy)
                Text("Sparar mätvärden och promptens/svarets token-ID:n lokalt. Inget skickas iväg.")
                    .font(.caption2).foregroundStyle(.secondary)
            }
            Text("Utvecklingsversion · \(model.contextTokens.formatted()) token kontextgräns. Modellens fullständiga numerik och 65K kontext är ännu inte verifierade på iPhone.")
                .font(.caption2).foregroundStyle(.secondary)
        }.padding(16)
    }
    private var composer: some View {
        HStack(alignment: .bottom, spacing: 12) {
            TextField("Skriv ett meddelande…", text: $model.draft, axis: .vertical)
                .lineLimit(1...6).textFieldStyle(.plain).submitLabel(.send)
                .onSubmit { model.send() }.disabled(!model.canChat)
                .accessibilityLabel("Meddelande")
            Button { model.send() } label: { Image(systemName: "arrow.up.circle.fill").font(.title) }
                .accessibilityLabel("Skicka meddelande")
                .disabled(!model.canChat || model.draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        }.padding(16)
    }
    private func ram(_ bytes: UInt64) -> String {
        String(format: "%.2f GB (%.2f GiB)", Double(bytes) / 1_000_000_000, Double(bytes) / 1_073_741_824)
    }
}

struct REAMMessageRow: View {
    let message: REAMMessage
    let isGenerating: Bool
    @State private var showsReasoning = false
    private var thinking: Bool { message.isThinking == true }
    private var hasReasoning: Bool { !(message.reasoning ?? "").isEmpty }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(message.role == .user ? "Du" : "Flash Local")
                .font(.caption.weight(.semibold)).foregroundStyle(.secondary)
            if message.role == .assistant && (thinking || hasReasoning) {
                DisclosureGroup(isExpanded: $showsReasoning) {
                    Text((message.reasoning ?? "…").trimmingCharacters(in: .whitespacesAndNewlines))
                        .font(.callout).foregroundStyle(.secondary).textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading).padding(.top, 6)
                } label: {
                    HStack(spacing: 8) {
                        if thinking { ProgressView().controlSize(.small) }
                        else { Image(systemName: "brain") }
                        Text(thinking ? "Tänker…" : "Resonemang").font(.subheadline.weight(.semibold))
                    }
                }
                .accessibilityIdentifier("chat-reasoning")
                .padding(12).background(.quaternary, in: RoundedRectangle(cornerRadius: 12))
                .onChange(of: thinking, initial: true) { _, active in showsReasoning = active }
            }
            if !message.text.isEmpty {
                if message.role == .assistant {
                    AnswerText(source: message.text.trimmingCharacters(in: .whitespacesAndNewlines))
                        .textSelection(.enabled)
                } else {
                    Text(verbatim: message.text.trimmingCharacters(in: .whitespacesAndNewlines))
                        .textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
                }
            } else if isGenerating && !thinking && !hasReasoning {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text("Förbereder svaret…").font(.callout).foregroundStyle(.secondary)
                }
            } else if message.interrupted && !thinking && !hasReasoning {
                Text("…")
            }
            if message.interrupted && !isGenerating {
                Text(hasReasoning && message.text.isEmpty ? "Avbrutet resonemang" : "Avbrutet svar")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
    }
}

#if DEBUG
#Preview("REAM chat") { REAMChatPage(model: .reviewPreview()).frame(width: 430, height: 900) }
#endif

enum REAMBundledNotices {
    static let text: String = {
        do {
            let files = ["QwenModelLicense", "OpenSourceNotices", "SwiftletNotices"]
            return try files.map { name in
                guard let url = Bundle.main.url(forResource: name, withExtension: "txt") else {
                    throw CocoaError(.fileNoSuchFile)
                }
                return try String(contentsOf: url, encoding: .utf8)
            }.joined(separator: "\n\n---\n\n")
        } catch { return "Licensfilerna kunde inte läsas. Rapportera detta byggfel innan modellen används." }
    }()
}
