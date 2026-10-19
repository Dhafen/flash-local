#if canImport(OrnithLES)
import SwiftUI
import OrnithLES
import MinirunKit
#if os(iOS)
import UIKit
#endif

struct OrnithLESView: View {
    @Bindable var model: OrnithLESAppModel
    var existing: VerifiedGGUFArtifact?
    @Environment(\.dismiss) private var dismiss
    @Environment(\.scenePhase) private var scenePhase
    private enum Pane: String, CaseIterable { case chat = "Snabbchatt", comparison = "Jämförelse" }
    @State private var pane: Pane = .chat
    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                Picker("Ornith", selection: $pane) {
                    ForEach(Pane.allCases, id: \.self) { Text($0.rawValue).tag($0) }
                }.pickerStyle(.segmented).padding(.horizontal).padding(.bottom, 8).disabled(model.isBusy)
                switch pane {
                case .chat: OrnithLESChatPage(model: model, activate: { model.activateChat(existing: existing) })
                case .comparison: OrnithLESPage(model: model, start: { model.start(existing: existing) })
                }
            }
                .navigationTitle("Ornith")
                .toolbar { ToolbarItem(placement: .cancellationAction) {
                    Button("Stäng") { Task { await model.shutdown(); dismiss() } }
                } }
        }
        .interactiveDismissDisabled()
        .onChange(of: scenePhase) { _, value in if value == .background { Task { await model.shutdown() } } }
        #if os(iOS)
        .onChange(of: model.isBusy, initial: true) { _, busy in UIApplication.shared.isIdleTimerDisabled = busy }
        .onDisappear { UIApplication.shared.isIdleTimerDisabled = false }
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.didReceiveMemoryWarningNotification)) { _ in
            Task { await model.shutdown() }
        }
        #endif
    }
}
struct OrnithLESPage: View {
    @Bindable var model: OrnithLESAppModel
    var start: () -> Void = {}
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("Samma Ornith. Fyra motorlägen.").font(.title2.bold())
                Text("Befintlig Q4_K_M · 4 096 kontextkapacitet · högst 5,5 GB processminne")
                    .font(.caption).foregroundStyle(.secondary)
                Text("Version \(Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "–") (\(Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "–"))")
                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                Text(model.status).font(.callout).accessibilityIdentifier("ornith-les-status")
                if let error = model.error { Text(error).font(.caption).foregroundStyle(.red).textSelection(.enabled) }
                if model.isBusy {
                    HStack { ProgressView(); Spacer(); Button("Stoppa", role: .cancel) { model.stop() } }
                } else {
                    Button("Kör kontroll och jämförelse") { start() }
                        .buttonStyle(.borderedProminent).disabled(!model.canStart)
                    Toggle("Jag har aktiverat flygplansläge", isOn: $model.userDeclaredAirplaneMode)
                        .disabled(!model.canStart)
                }
                Text("Ingen ny modell hämtas. Fasta vikter kopieras till en mindre lokal fil; experterna läses direkt ur din GGUF. Testet ändrar inte den vanliga chatten.")
                    .font(.footnote).foregroundStyle(.secondary)
                Text("Först jämförs hela modellens logits med CPU-referenser vid fyra positioner. Kontrollen räknar även direkt från originalvikterna utan extra aktiveringskvantisering. Därefter körs samma fasta prompt med vanlig motor, expertcache, LES-förläsning och OS-cache. Varje läge körs två gånger med minst 128 genererade token. Låt appen vara öppen; det kan ta flera minuter.")
                    .font(.footnote).foregroundStyle(.secondary)
                if let check = model.numericalCheck {
                    Text(check.nativeBenchmarkAdmitted ? "Kort kontroll mot originalvikterna godkänd." : "Referenskontrollen avvek. Ingen ny motor aktiverades.")
                        .font(.callout.bold()).foregroundStyle(check.nativeBenchmarkAdmitted ? Color.primary : Color.red)
                    if check.nativeBenchmarkAdmitted && check.nativePrefillTokens == 1 {
                        Text("Prompten behandlas en token åt gången eftersom batchkontrollen inte godkändes.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    if check.lookahead.activeLookaheadDepth == 0 {
                        Text("RAM-utrymmet räckte till expertcache men inte till förläsning i kontrollen.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
                ForEach(Array(model.runs.enumerated()), id: \.offset) { _, run in
                    VStack(alignment: .leading, spacing: 5) {
                        Text("\(run.backend.rawValue) · \(run.applicationCache)").font(.headline)
                        if let plan = run.memoryPlan, plan.requestedLookaheadDepth > 0 {
                            Text("Förläsning \(plan.lookaheadDepth)/\(plan.requestedLookaheadDepth) · top-8 oförändrat")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        Text(String(format: "%.2f token/s · %.1f prompttoken/s · %.2f GB", run.decodeTokensPerSecond,
                            run.prefillTokensPerSecond, Double(run.maximumObservedFootprintBytes) / 1e9))
                            .font(.subheadline.monospacedDigit())
                        Text(run.hasRequiredLength ? "\(run.generatedTokens) token · termiskt tillstånd \(run.maximumThermalState)" : "För kort: \(run.generatedTokens)/128 token")
                            .font(.caption).foregroundStyle(.secondary)
                    }.frame(maxWidth: .infinity, alignment: .leading).padding(.vertical, 8)
                }
                if let url = model.measurementURL { ShareLink("Dela mätrapport", item: url).buttonStyle(.bordered) }
                Text("Tom appcache innebär inte tom OS-cache. iOS ger termiskt tillstånd, inte temperatur i grader. Fyra kontrollpositioner verifierar inte all modellkvalitet eller lång kontext.")
                    .font(.caption).foregroundStyle(.secondary)
            }.padding(20).frame(maxWidth: .infinity, alignment: .leading)
        }.background(MRColor.abyss).tint(MRColor.tierPinned)
    }
}
/// A separate experimental conversation; the ordinary chat keeps its engine
/// and saved history. The parent sheet owns inference cancellation/teardown.
struct OrnithLESChatPage: View {
    @Bindable var model: OrnithLESAppModel
    var activate: () -> Void = {}
    var body: some View {
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(alignment: .leading, spacing: 18) {
                        header
                        ForEach(model.chatMessages) { message in
                            REAMMessageRow(message: message, isGenerating: model.isBusy && message.id == model.chatMessages.last?.id)
                                .id(message.id)
                        }
                    }.padding(20).frame(maxWidth: .infinity, alignment: .leading)
                }
                .onChange(of: model.chatMessages.count) { _, _ in
                    if let id = model.chatMessages.last?.id { proxy.scrollTo(id, anchor: .bottom) }
                }
            }
            Divider()
            HStack(alignment: .bottom, spacing: 12) {
                TextField("Skriv ett meddelande…", text: $model.chatDraft, axis: .vertical)
                    .lineLimit(1...6).textFieldStyle(.plain).submitLabel(.send)
                    .onSubmit { model.sendChat() }.disabled(!model.canSendChat)
                    .accessibilityLabel("Meddelande i snabbchatten")
                Button { model.sendChat() } label: { Image(systemName: "arrow.up.circle.fill").font(.title) }
                    .accessibilityLabel("Skicka i snabbchatten")
                    .disabled(!model.canSendChat || model.chatDraft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }.padding(16)
        }.background(MRColor.abyss).tint(MRColor.tierPinned)
    }
    private var header: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Ornith snabbchatt").font(.title2.bold())
            Text("Experimentell motor · din befintliga GGUF · 4 096 token kontextgräns")
                .font(.caption).foregroundStyle(.secondary)
            Text(model.status).font(.callout).accessibilityIdentifier("ornith-chat-status")
            if let error = model.error { Text(error).font(.caption).foregroundStyle(.red).textSelection(.enabled) }
            if model.isBusy {
                HStack { ProgressView(); Spacer(); Button("Stoppa", role: .cancel) { model.stop() } }
            } else if !model.chatReady {
                Button("Aktivera snabbchatten") { activate() }.buttonStyle(.borderedProminent).disabled(!model.canStart)
                Text("Första starten kontrollerar numeriken mot originalvikterna. Ingen modell hämtas.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Toggle("Tänkande", isOn: $model.chatThinkingEnabled).disabled(!model.canStart)
                .accessibilityIdentifier("ornith-chat-thinking")
            if let progress = model.chatProgress, model.isBusy {
                Text(String(format: "Appminne %.2f GB · läsanrop %.2f GB · %d token", Double(progress.footprintBytes) / 1e9,
                    Double(progress.ioBytesRead) / 1e9, progress.generatedTokens)).font(.caption.monospacedDigit())
            }
            if let summary = model.chatSummary {
                if let speed = summary.decodeTokensPerSecond {
                    Text(String(format: "%.2f token/s · prompt %.2f s · %d återanvända token", speed,
                        summary.prefillSeconds, summary.reusedPromptTokens)).font(.caption.monospacedDigit())
                }
                Button("Spara mätning i Filer", systemImage: "chart.bar.doc.horizontal") { model.saveChatMeasurement() }
                    .disabled(!model.canStart)
                Text("Token-ID:n och mätvärden sparas lokalt när du trycker. Inget skickas iväg.")
                    .font(.caption2).foregroundStyle(.secondary)
            }
            if let url = model.chatMeasurementURL { ShareLink("Dela chattmätning", item: url) }
            if !model.chatMessages.isEmpty {
                Button("Rensa snabbchatten", role: .destructive) { model.clearChat() }.disabled(!model.canStart)
            }
            Text("Kort numerikkontroll krävs. Full modellkvalitet och lång kontext är ännu inte verifierade. Samtalet hålls separat i den här vyn.")
                .font(.caption2).foregroundStyle(.secondary)
            Text("OS-filcache ingår inte i det redovisade appminnet. Återläsningar kan komma från cachen.")
                .font(.caption2).foregroundStyle(.secondary)
        }
    }
}
#endif
