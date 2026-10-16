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
    var body: some View {
        NavigationStack {
            REAMChatPage(model: model, discard: { confirmation = .discard })
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
}

/// Pure page content, separable from navigation/lifecycle for light/dark QA.
struct REAMChatPage: View {
    @Bindable var model: REAMAppModel
    var discard: () -> Void = {}
    var body: some View {
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(spacing: 0) {
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
                            VStack(alignment: .leading, spacing: 7) {
                                Text(message.role == .user ? "Du" : "Flash Local")
                                    .font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                                Text(message.text.isEmpty && message.interrupted ? "…" : message.text)
                                    .textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
                                if message.interrupted && !model.isBusy {
                                    Text("Avbrutet svar").font(.caption).foregroundStyle(.secondary)
                                }
                            }.id(message.id)
                        }
                        Color.clear.frame(height: 1).id("latest")
                    }.padding(20)
                    }
                }
                .onChange(of: model.messages.last?.text) { _, _ in proxy.scrollTo("latest", anchor: .bottom) }
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
                Text("RAM \(ram(p.footprintBytes)) · läst \(ByteCountFormatter.string(fromByteCount: Int64(p.ioBytesRead), countStyle: .file)) · \(p.generatedTokens) token")
                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
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

#if DEBUG
#Preview("REAM chat") { REAMChatPage(model: .reviewPreview()).frame(width: 430, height: 900) }
#endif

enum REAMBundledNotices {
    static let text: String = {
        do {
            let files = ["QwenModelLicense", "OpenSourceNotices"]
            return try files.map { name in
                guard let url = Bundle.main.url(forResource: name, withExtension: "txt") else {
                    throw CocoaError(.fileNoSuchFile)
                }
                return try String(contentsOf: url, encoding: .utf8)
            }.joined(separator: "\n\n---\n\n")
        } catch { return "Licensfilerna kunde inte läsas. Rapportera detta byggfel innan modellen används." }
    }()
}
