#if canImport(OrnithLES)
import SwiftUI
import OrnithLES
import MinirunKit
#if os(iOS)
import UIKit
#elseif os(macOS)
import AppKit
#endif

struct OrnithLESView: View {
    @Bindable var model: OrnithLESAppModel
    var existing: VerifiedGGUFArtifact?
    @Environment(\.dismiss) private var dismiss
    @Environment(\.scenePhase) private var scenePhase
    private enum Pane: String, CaseIterable { case chat = "Snabbchatt", comparison = "Jämförelse", api = "API" }
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
                case .api: OrnithAPIPage(model: model, activate: { model.activateChat(existing: existing) })
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
        .onChange(of: model.shouldKeepAwake, initial: true) { _, awake in UIApplication.shared.isIdleTimerDisabled = awake }
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
                        .buttonStyle(.borderedProminent).disabled(!model.canBenchmark)
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
            Text("Experimentell motor · din befintliga GGUF · \(model.contextCapacityTokens.formatted()) token kontextgräns")
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
            Toggle("Tänkande", isOn: $model.chatThinkingEnabled).disabled(!model.canStart || model.isAPIRunning)
                .accessibilityIdentifier("ornith-chat-thinking")
            OrnithPromptControls(model: model)
            OrnithPrefillControls(model: model, activate: activate)
            OrnithGenerationControls(options: $model.chatOptions).disabled(!model.canStart || model.isAPIRunning)
            if let progress = model.chatProgress, model.isBusy {
                Text("Kontext: \(progress.promptTokensTotal) prompt + högst \(model.chatOptions.maximumNewTokens) nya / \(model.contextCapacityTokens.formatted()) token")
                    .font(.caption.monospacedDigit())
                Text(String(format: "Appminne %.2f GB · läsanrop %.2f GB · %d token", Double(progress.footprintBytes) / 1e9,
                    Double(progress.ioBytesRead) / 1e9, progress.generatedTokens)).font(.caption.monospacedDigit())
            }
            if let summary = model.chatSummary {
                Text("Evaluation \(summary.evaluationBatchTokens) · fysisk gräns \(summary.physicalBatchTokens) · uppmätt högst \(summary.metricsAfter.maximumPhysicalBatchTokens ?? 0) token per fysisk batch")
                    .font(.caption.monospacedDigit())
                if let speed = summary.decodeTokensPerSecond {
                    Text(String(format: "%.2f token/s · prompt %.2f s · %d återanvända token", speed,
                        summary.prefillSeconds, summary.reusedPromptTokens)).font(.caption.monospacedDigit())
                }
                Text(String(format: "Prompt: %.2f token/s · kontext %d + %d / %d", Double(summary.evaluatedPromptTokens) / max(summary.prefillSeconds, 1e-9), summary.promptTokenIDs.count, summary.generatedTokenIDs.count, model.contextCapacityTokens))
                    .font(.caption.monospacedDigit())
                Text("\(summary.reasoningTokens) tänktoken · \(summary.forcedThinkingEndTokens) token för budgetstyrt avslut")
                    .font(.caption2).foregroundStyle(.secondary)
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

struct OrnithPromptControls: View {
    @Bindable var model: OrnithLESAppModel
    @State private var expanded: Bool
    init(model: OrnithLESAppModel, expanded: Bool = false) { self.model = model; self._expanded = State(initialValue: expanded) }
    var body: some View {
        DisclosureGroup("Systemprompt och arbetsprofil", isExpanded: $expanded) {
            VStack(alignment: .leading, spacing: 12) {
                Menu("Välj startprofil") {
                    ForEach(LocalChatProfile.allCases) { profile in Button(profile.title) { model.applyProfile(profile) } }
                }
                TextEditor(text: $model.systemPrompt).frame(minHeight: 130).font(.callout)
                    .accessibilityLabel("Redigerbar systemprompt")
                    .overlay(RoundedRectangle(cornerRadius: 6).stroke(.secondary.opacity(0.3)))
                Text("Profilen fyller i en redigerbar instruktion. Den ändrar inga vikter eller genereringsvärden. Högst 32 KiB; texten räknas in i kontexten.")
                    .font(.caption).foregroundStyle(.secondary)
                Text("Ändringar gäller nästa chattfråga och kan kräva att prompten läses om. API:t tar en kopia vid start; klientens egen systemroll har företräde.")
                    .font(.caption).foregroundStyle(.secondary)
            }.padding(.top, 8)
        }.disabled(!model.canStart || model.isAPIRunning)
    }
}

struct OrnithPrefillControls: View {
    @Bindable var model: OrnithLESAppModel
    var activate: () -> Void = {}
    @State private var expanded: Bool
    init(model: OrnithLESAppModel, activate: @escaping () -> Void = {}, expanded: Bool = false) {
        self.model = model; self.activate = activate; self._expanded = State(initialValue: expanded)
    }
    var body: some View {
        DisclosureGroup("Promptinläsning och batch", isExpanded: $expanded) {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Text("Evaluation batch")
                    Spacer()
                    Picker("Evaluation batch", selection: Binding(get: { model.batchConfiguration.evaluationTokens }, set: { value in
                        var selected = model.batchConfiguration; selected.evaluationTokens = value
                        if selected.physicalTokens > value { selected.physicalTokens = 0 }
                        model.batchConfiguration = selected
                    })) { ForEach([1, 4, 8, 16, 32], id: \.self) { Text("\($0) token").tag($0) } }.labelsHidden()
                }
                HStack {
                    Text("Fysisk batch")
                    Spacer()
                    Picker("Fysisk batch", selection: Binding(get: { model.batchConfiguration.physicalTokens }, set: { value in
                        var selected = model.batchConfiguration; selected.physicalTokens = value
                        selected.evaluationTokens = max(selected.evaluationTokens, value); model.batchConfiguration = selected
                    })) {
                        Text("Auto · kontrollerad 1/4").tag(0)
                        ForEach([1, 4, 8, 16], id: \.self) { Text("\($0) token").tag($0) }
                    }.labelsHidden()
                }
                Text("Evaluation styr prompttoken per anrop. Fysisk batch styr samtidig beräkning. Större batch använder mer RAM och är inte alltid snabbare; modellens top-8 ändras inte.")
                    .font(.caption).foregroundStyle(.secondary)
                if model.batchSettingsChanged {
                    Text("Nytt batchval väntar. Ladda om motorn innan chatten eller API:t används.").font(.caption)
                    Button("Kontrollera och ladda om") { activate() }.buttonStyle(.borderedProminent)
                } else if !model.chatReady {
                    Text("Valet kontrolleras mot originalvikterna vid aktivering. Avvikelse stoppar den valda batchen; Auto behåller den tidigare kontrollen.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                if model.chatReady, let proof = model.activePrefillCheck {
                    Text(proof.passed ? "Vald batch har klarat den korta logit- och expertkontrollen." : "Batchkontrollen avvek.")
                        .font(.caption).foregroundStyle(proof.passed ? Color.primary : Color.red)
                }
                Text("Snabbchatten använder 8 192 token för prompt, tänkande och svar. Jämförelsens fasta benchmark kör fortfarande 4 096 token. Lång kontext behöver verifieras på enheten.")
                    .font(.caption).foregroundStyle(.secondary)
                Button("Jämför vald batch med Auto") { model.startPrefillBenchmark() }.buttonStyle(.bordered)
                    .disabled(!model.canBenchmark || model.batchConfiguration == LESPrefillConfiguration())
                Text("Fyra körningar: Auto, vald, vald, Auto. Samma fasta prompt, minst 128 token per prov. Egna profiler och samplerinställningar används inte. Håll appen öppen; OS-cachen är okontrollerad.")
                    .font(.caption).foregroundStyle(.secondary)
                if let report = model.prefillBenchmark {
                    ForEach(Array(report.runs.enumerated()), id: \.offset) { index, run in
                        Text(String(format: "%d · %@ · prompt %.2f token/s · svar %.2f token/s · %.2f GB", index + 1,
                            run.mode, Double(run.generation.evaluatedPromptTokens) / max(run.generation.prefillSeconds, 1e-9),
                            run.generation.decodeTokensPerSecond ?? 0, Double(run.generation.maximumObservedFootprintBytes) / 1e9))
                            .font(.caption.monospacedDigit())
                    }
                    Text(report.comparable ? "Samma token och jämförbart termiskt tillstånd. Upprepa för hållbar prestanda." : "Ingen säker vinnare: token, längd, strömsparläge eller termiskt tillstånd skiljer.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                if model.prefillBenchmark != nil || model.prefillBenchmarkFailure != nil {
                    Button("Spara batchmätning i Filer", systemImage: "chart.bar.doc.horizontal") { model.savePrefillBenchmark() }
                    Text("Rapporten innehåller token-ID:n och mätvärden; den sparas lokalt när du trycker.")
                        .font(.caption2).foregroundStyle(.secondary)
                }
                if let url = model.prefillBenchmarkURL { ShareLink("Dela batchrapport", item: url) }
            }.padding(.top, 8)
        }.disabled(!model.canStart || model.isAPIRunning)
    }
}

struct OrnithGenerationControls: View {
    @Binding var options: LocalGenerationOptions
    @State private var expanded = false
    init(options: Binding<LocalGenerationOptions>, expanded: Bool = false) {
        self._options = options; self._expanded = State(initialValue: expanded)
    }
    var body: some View {
        DisclosureGroup("Generering", isExpanded: $expanded) {
            VStack(alignment: .leading, spacing: 12) {
                integerRow("Max token totalt", value: $options.maximumNewTokens)
                Text("Tänkande och svar delar gränsen. 1–2 048 token per svar; snabbchattens kontext är 8 192 token.")
                    .font(.caption).foregroundStyle(.secondary)
                if options.enableThinking {
                    Toggle("Begränsa tänkandet", isOn: Binding(get: { options.thinkingBudget != nil },
                        set: { options.thinkingBudget = $0 ? min(256, max(0, options.maximumNewTokens - 16)) : nil }))
                    if options.thinkingBudget != nil {
                        integerRow("Tänkbudget (token)", value: Binding(get: { options.thinkingBudget ?? 256 }, set: { options.thinkingBudget = $0 }))
                        Text("0 stänger av tänkandet. Vid gränsen avslutas resonemanget med modellens slutmarkör och svaret fortsätter. Lämna utrymme i totalgränsen. En kort budget kan försämra svaret.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
                HStack {
                    Text("Temperatur")
                    Spacer()
                    TextField("Temperatur", value: $options.temperature, format: .number.precision(.fractionLength(0...2)))
                        .multilineTextAlignment(.trailing).frame(width: 85).textFieldStyle(.roundedBorder)
                }
                Text("0 ger deterministiska svar. Högre värden ger mer variation, upp till 2.")
                    .font(.caption).foregroundStyle(.secondary)
                HStack {
                    Text("Top-p"); Spacer()
                    TextField("Top-p", value: $options.topP, format: .number.precision(.fractionLength(0...2)))
                        .multilineTextAlignment(.trailing).frame(width: 85).textFieldStyle(.roundedBorder)
                }
                HStack {
                    Text("Slumpfrö"); Spacer()
                    TextField("Slumpfrö", value: $options.seed, format: .number.grouping(.never))
                        .multilineTextAlignment(.trailing).frame(width: 110).textFieldStyle(.roundedBorder)
                }
                Text("Top-p och slumpfrö används när temperaturen är större än 0. API:t använder dessa standardvärden vid start; anrop kan ange egna värden.")
                    .font(.caption).foregroundStyle(.secondary)
            }.padding(.top, 8)
        }.accessibilityIdentifier("ornith-generation-controls")
    }
    private func integerRow(_ title: String, value: Binding<Int>) -> some View {
        HStack {
            Text(title); Spacer()
            TextField(title, value: value, format: .number.grouping(.never))
                .multilineTextAlignment(.trailing).frame(width: 100).textFieldStyle(.roundedBorder)
                .accessibilityLabel(title)
        }
    }
}

struct OrnithAPIPage: View {
    @Bindable var model: OrnithLESAppModel
    var activate: () -> Void = {}
    @State private var showKey = false
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("Använd Ornith från datorn").font(.title2.bold())
                Text("OpenAI-kompatibelt API · lokal inferens på iPhone")
                    .font(.subheadline).foregroundStyle(.secondary)
                Text(model.apiStatus).accessibilityIdentifier("ornith-api-status")
                if let error = model.error { Text(error).foregroundStyle(.red).font(.caption) }
                if (!model.chatReady || model.batchSettingsChanged) && !model.isAPIRunning {
                    Text("Förbered samma verifierade motor som används i snabbchatten.").font(.callout)
                    Button("Förbered modellen") { activate() }.buttonStyle(.borderedProminent).disabled(!model.canStart)
                }
                HStack {
                    Text("Port"); Spacer()
                    TextField("Port", value: $model.apiPort, format: .number.grouping(.never))
                        .multilineTextAlignment(.trailing).frame(width: 100).textFieldStyle(.roundedBorder)
                        .disabled(!model.apiAccessKey.isEmpty || !model.canStart)
                }
                if model.apiAccessKey.isEmpty {
                    Button("Starta lokalt API") { model.startAPI() }.buttonStyle(.borderedProminent)
                        .disabled(!model.canStart || !model.chatReady || model.batchSettingsChanged)
                } else {
                    Button("Stoppa API", role: .destructive) { model.stopAPI() }.buttonStyle(.bordered)
                }
                ForEach(model.apiURLs, id: \.self) { url in
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Base URL").font(.caption).foregroundStyle(.secondary)
                        Text(url).font(.callout.monospaced()).textSelection(.enabled)
                        Button("Kopiera adress", systemImage: "doc.on.doc") { copy(url) }
                    }
                }
                if !model.apiAccessKey.isEmpty {
                    Text("Lokal API-nyckel").font(.headline)
                    if showKey { Text(model.apiAccessKey).font(.caption.monospaced()).textSelection(.enabled).privacySensitive() }
                    HStack {
                        Button(showKey ? "Dölj nyckel" : "Visa nyckel") { showKey.toggle() }
                        Spacer()
                        Button("Kopiera nyckel", systemImage: "key") { copy(model.apiAccessKey) }
                    }
                    Text("Nyckeln skapas på nytt när servern startas och sparas inte. Ange den som API key i klienten.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Toggle("Behåll skärmen vaken", isOn: $model.apiKeepAwake)
                Text("Modellnamn").font(.caption).foregroundStyle(.secondary)
                Text(LocalChatRequest.modelID).font(.callout.monospaced()).textSelection(.enabled)
                Text("Samma Wi-Fi eller Ethernet via USB-C-adapter. HTTP används på det betrodda lokala nätverket. Vanlig USB-kabel kräver separat portvidarebefordran eller nätverksdrivrutin på datorn; den vägen är ännu inte verifierad.")
                    .font(.callout)
                Text("Låt appen vara öppen i förgrunden. Låsning, bakgrund eller minnestryck stoppar servern. Chatten och API:t delar motorn och hanterar ett anrop åt gången.")
                    .font(.caption).foregroundStyle(.secondary)
                Text("Stöd: /v1/models, /v1/chat/completions och strömmande textsvar. Resonemang skickas separat som reasoning_content. Strukturerade verktygsanrop och Responses API stöds ännu inte.")
                    .font(.caption).foregroundStyle(.secondary)
                Text("\(model.apiCompletedRequests) slutförda API-anrop").font(.caption.monospacedDigit())
                OrnithGenerationControls(options: $model.chatOptions).disabled(!model.canStart || model.isAPIRunning)
                OrnithPromptControls(model: model)
                OrnithPrefillControls(model: model, activate: activate)
            }.padding(20).frame(maxWidth: .infinity, alignment: .leading)
        }.background(MRColor.abyss).tint(MRColor.tierPinned)
    }
    private func copy(_ value: String) {
        #if os(iOS)
        UIPasteboard.general.string = value
        #else
        NSPasteboard.general.clearContents(); NSPasteboard.general.setString(value, forType: .string)
        #endif
    }
}
#endif
