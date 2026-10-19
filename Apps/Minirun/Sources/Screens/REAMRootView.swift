import MinirunKit
import SwiftUI
#if os(iOS)
import UIKit
#endif

struct REAMRootView: View {
    @Bindable var model: REAMAppModel
    @Environment(\.scenePhase) private var scenePhase
    private enum Confirmation { case clear, discard }
    @State private var confirmation: Confirmation?
    private enum AlternateRunner: String, Identifiable {
        case qpack
        #if canImport(OrnithLES)
        case les
        #endif
        var id: String { rawValue }
    }
    @State private var alternateRunner: AlternateRunner?
    @State private var performanceModel = OrnithPerformanceAppModel.live()
    #if canImport(OrnithLES)
    @State private var lesModel = OrnithLESAppModel.live()
    #endif
    var body: some View {
        NavigationStack {
            REAMChatPage(model: model, discard: { confirmation = .discard }, openPerformance: openPerformance, openLES: lesAction)
                .navigationTitle("Flash Local")
                .toolbar {
                    ToolbarItem(placement: .automatic) {
                        Button { confirmation = .clear } label: { Image(systemName: "trash") }
                            .accessibilityLabel("Rensa chatt")
                            .disabled(model.isBusy || model.messages.isEmpty)
                    }
                    ToolbarItem(placement: .automatic) {
                        Menu {
                            Picker("Modell", selection: Binding(get: { model.selectedModel }, set: { model.selectModel($0) })) {
                                ForEach(LocalChatModel.allCases) { target in Text(target.title).tag(target) }
                            }.disabled(!model.canSelectModel)
                            Button("Ornith QPack (testläge)") {
                                openPerformance()
                            }.disabled(!model.canOpenAlternateRunner)
                            #if canImport(OrnithLES)
                            Button("Ornith snabbchatt och motortest") { openLES() }
                                .disabled(!model.canOpenAlternateRunner)
                            #endif
                            NavigationLink("Om Flash Local") { AboutView() }
                            Button("Kontrollera numerik (kort test)") { model.checkNumerics() }
                                .disabled(!model.canRunNumerics)
                            NavigationLink("Licenser på enheten") {
                                ScrollView {
                                    Text(REAMBundledNotices.text).font(.footnote).textSelection(.enabled).padding()
                                        .frame(maxWidth: .infinity, alignment: .leading)
                                }.navigationTitle("Licenser")
                            }
                        } label: { Image(systemName: "info.circle") }
                        .accessibilityLabel("Information och licenser")
                    }
                }
                .confirmationDialog(confirmation == .clear ? "Rensa den här chatten?" : "Radera den påbörjade hämtningen?",
                    isPresented: Binding(get: { confirmation != nil }, set: { if !$0 { confirmation = nil } }), titleVisibility: .visible) {
                    if confirmation == .clear {
                        Button("Rensa chatten", role: .destructive) { model.clearMessages() }
                    } else {
                        Button("Radera och börja om", role: .destructive) { model.discardPartial() }
                    }
                } message: {
                    if confirmation == .discard { Text("Redan hämtade delar tas bort. En färdig modellfil påverkas inte.") }
                }
        }
        .sheet(item: $alternateRunner, onDismiss: {
            Task {
                await performanceModel.shutdown()
                #if canImport(OrnithLES)
                await lesModel.shutdown()
                #endif
                model.resumeAfterAlternateRunner()
            }
        }) { runner in
            switch runner {
            case .qpack: OrnithPerformanceView(model: performanceModel)
            #if canImport(OrnithLES)
            case .les: OrnithLESView(model: lesModel, existing: model.verifiedOrnithForAlternateRunner())
            #endif
            }
        }
        .tint(MRColor.tierPinned)
        .task { model.prepare() }
        .onChange(of: scenePhase) { _, phase in
            if phase == .background { model.releaseInferenceResources() }
            else if phase == .active { model.refreshStorage() }
        }
        #if os(iOS)
        .onChange(of: model.isBusy, initial: true) { _, value in UIApplication.shared.isIdleTimerDisabled = value }
        .onDisappear { UIApplication.shared.isIdleTimerDisabled = false; model.releaseInferenceResources() }
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.didReceiveMemoryWarningNotification)) { _ in
            model.releaseInferenceResources()
        }
        #endif
    }
    private func openPerformance() {
        guard model.canOpenAlternateRunner else { return }
        Task { await model.pauseForAlternateRunner(); alternateRunner = .qpack }
    }
    private var lesAction: (() -> Void)? {
        #if canImport(OrnithLES)
        return { openLES() }
        #else
        return nil
        #endif
    }
    #if canImport(OrnithLES)
    private func openLES() {
        guard model.canOpenAlternateRunner else { return }
        Task { await model.pauseForAlternateRunner(); alternateRunner = .les }
    }
    #endif
}

/// Pure page content, separable from navigation/lifecycle for light/dark QA.
struct REAMChatPage: View {
    @Bindable var model: REAMAppModel
    var discard: () -> Void = {}
    var openPerformance: () -> Void = {}
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
                Text(message.text.trimmingCharacters(in: .whitespacesAndNewlines))
                    .textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
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
