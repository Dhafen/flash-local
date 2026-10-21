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
                    Button("Stäng") { dismiss() }
                } }
        }
        .interactiveDismissDisabled()
        .onChange(of: scenePhase) { _, value in
            if value == .inactive { model.prepareForInactivity() }
            if value == .background { Task { await model.enterBackground() } }
            if value == .active { model.enterForeground() }
        }
        #if os(iOS)
        .onChange(of: model.shouldKeepAwake, initial: true) { _, awake in UIApplication.shared.isIdleTimerDisabled = awake }
        .onDisappear { UIApplication.shared.isIdleTimerDisabled = false }
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.didReceiveMemoryWarningNotification)) { _ in
            Task { await model.handleMemoryPressure() }
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
            }
            HStack {
                Button("Ladda modell") { activate() }.buttonStyle(.borderedProminent)
                    .disabled(!model.canStart || (model.chatReady && !model.batchSettingsChanged))
                Button("Ladda ur modell", role: .destructive) { Task { await model.unloadModel() } }
                    .buttonStyle(.bordered).disabled(model.isShuttingDown || (!model.chatReady && !model.isBusy))
            }
            Text(model.loadState).font(.caption.bold())
            if let started = model.loadingStartedAt, model.isBusy {
                TimelineView(.periodic(from: started, by: 1)) { tick in
                    Text("Förfluten tid: \(Int(max(0, tick.date.timeIntervalSince(started)))) s · \(model.loadingFootprintText)")
                        .font(.caption.monospacedDigit())
                }
                Text("Fasen ovan rapporteras av motorn. Procent och återstående tid saknas för denna fas.")
                    .font(.caption2).foregroundStyle(.secondary)
            }
            if !model.chatReady {
                Text("Första starten i en ny appversion kontrollerar numeriken mot originalvikterna. Godkänd kontroll återanvänds vid följande starter om modell, motor och iOS är oförändrade. Ingen modell hämtas.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            OrnithPromptControls(model: model)
            OrnithPrefillControls(model: model, activate: activate)
            OrnithGenerationControls(options: $model.chatOptions, contextTokens: model.contextCapacityTokens).disabled(!model.canStart || model.isAPIRunning)
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
                Text("Prompt: \(summary.evaluatedPromptTokens) nya · \(summary.reusedPromptTokens) återanvända · \(summary.promptTokenIDs.count) totalt")
                    .font(.caption.monospacedDigit())
                Text(String(format: "Prompt: %.2f token/s · kontext %d + %d / %d", Double(summary.evaluatedPromptTokens) / max(summary.prefillSeconds, 1e-9), summary.promptTokenIDs.count, summary.generatedTokenIDs.count, model.contextCapacityTokens))
                    .font(.caption.monospacedDigit())
                if let first = summary.timeToFirstTokenSeconds {
                    Text(String(format: "Första token: %.2f s", first)).font(.caption.monospacedDigit())
                }
                Text("\(summary.reasoningTokens) tänktoken · \(summary.forcedThinkingEndTokens) token för budgetstyrt avslut")
                    .font(.caption2).foregroundStyle(.secondary)
                Button("Spara mätning i Filer", systemImage: "chart.bar.doc.horizontal") { model.saveChatMeasurement() }
                    .disabled(!model.canStart)
                Text("Token-ID:n och mätvärden sparas lokalt när du trycker. Inget skickas iväg.")
                    .font(.caption2).foregroundStyle(.secondary)
            }
            if let url = model.chatMeasurementURL { ShareLink("Dela chattmätning", item: url) }
            DisclosureGroup("Kontrollera KV- och prefixcache") {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Tre riktiga körningar: första turen, följdfråga med varm kontext, sedan samma följdfråga med nollställd kontext. Fasta testprompter och 64 tänktoken; din chatt behålls. OS-filcachen är okontrollerad. Motorn behöver laddas efter testet.")
                        .font(.caption).foregroundStyle(.secondary)
                    Button("Kör cachetest och spara rapport") { model.startCacheBenchmark() }.disabled(!model.canBenchmark)
                    if let report = model.cacheBenchmark {
                        Text("Varm: \(report.warm.evaluatedPromptTokens) nya / \(report.warm.reusedPromptTokens) återanvända. Ny kontext: \(report.fresh.evaluatedPromptTokens) nya.")
                        Text(String(format: "Uppmätt skillnad i prompttid: %.2f s · identiska svar: %@", report.observedPrefillSecondsDifference, report.exactOutputsMatch ? "ja" : "nej"))
                    }
                    if let url = model.cacheBenchmarkURL { ShareLink("Dela cachetest", item: url) }
                }
            }.font(.caption)
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
                Picker("Kontext", selection: $model.contextProfile) {
                    ForEach(LocalContextProfile.allCases) { Text($0.title).tag($0) }
                }.accessibilityIdentifier("ornith-context-profile")
                Text("8K är standard. Större fönster är experimentella och kräver ny minneskontroll. En lyckad start verifierar inte svarskvalitet i lång kontext.")
                    .font(.caption).foregroundStyle(.secondary)
                HStack {
                    Text("Evaluation batch")
                    Spacer()
                    Picker("Evaluation batch", selection: Binding(get: { model.batchConfiguration.evaluationTokens }, set: { value in
                        var selected = model.batchConfiguration; selected.evaluationTokens = value
                        if selected.physicalTokens > value { selected.physicalTokens = 0 }
                        model.batchConfiguration = selected
                    })) { ForEach([1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048], id: \.self) { Text("\($0) token").tag($0) } }.labelsHidden()
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
                Toggle("Använd sparad, uppmätt batchprofil automatiskt", isOn: $model.automaticBatch)
                Text("En godkänd jämförelse sparar vinnaren för samma telefon, modell, motor och kontext. Ingen benchmark körs vid vanlig laddning. Utan mätning behålls den kontrollerade batchen. Fysisk batch över 16 stöds ännu inte av denna streaminggraf; 128/256/512 visas inte som fungerande.")
                    .font(.caption).foregroundStyle(.secondary)
                if model.batchSettingsChanged {
                    Text("Nytt kontext- eller batchval väntar. Ladda om motorn innan chatten eller API:t används.").font(.caption)
                    Button("Kontrollera och ladda om") { activate() }.buttonStyle(.borderedProminent)
                } else if !model.chatReady {
                    Text("Valet kontrolleras mot originalvikterna vid aktivering. Avvikelse stoppar den valda batchen; Auto behåller den tidigare kontrollen.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                if model.chatReady, let proof = model.activePrefillCheck {
                    Text(proof.passed ? "Vald batch har klarat den korta logit- och expertkontrollen." : "Batchkontrollen avvek.")
                        .font(.caption).foregroundStyle(proof.passed ? Color.primary : Color.red)
                }
                Text("Aktiv snabbchatt: \(model.contextCapacityTokens.formatted()) token inklusive prompt, tänkande och svar. Batchjämförelsen använder valt kontextfönster. Lång kontext behöver verifieras på enheten.")
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
    var contextTokens: Int = LocalContextProfile.standard.rawValue
    @State private var expanded = false
    init(options: Binding<LocalGenerationOptions>, contextTokens: Int = LocalContextProfile.standard.rawValue, expanded: Bool = false) {
        self._options = options; self.contextTokens = contextTokens; self._expanded = State(initialValue: expanded)
    }
    var body: some View {
        DisclosureGroup("Generering", isExpanded: $expanded) {
            VStack(alignment: .leading, spacing: 12) {
                Picker("Tänkande", selection: Binding(
                    get: { OrnithThinkingPreset.matching(options) ?? .medium },
                    set: { options = $0.applying(to: options) })) {
                    ForEach(OrnithThinkingPreset.allCases) { preset in
                        Text(preset.title).tag(preset)
                    }
                }.pickerStyle(.segmented)
                    .accessibilityIdentifier("ornith-thinking-preset")
                Text("Off: inget tänkande · Small: 256 · Medium: 512 · Ultra: 2 048 tänktoken. Svarstaket sätts automatiskt (1 536–4 096 token). Aktiv kontext: \(contextTokens.formatted()) token inklusive prompt och svar.")
                    .font(.caption).foregroundStyle(.secondary)
                HStack {
                    Text("Temperatur")
                    Spacer()
                    OrnithDecimalField(title: "Temperatur", value: $options.temperature, minimum: 0, maximum: 2)
                }
                Text("0 ger deterministiska svar. Högre värden ger mer variation, upp till 2.")
                    .font(.caption).foregroundStyle(.secondary)
                HStack {
                    Text("Top-p"); Spacer()
                    OrnithDecimalField(title: "Top-p", value: $options.topP, minimum: 0, maximum: 1, excludingMinimum: true)
                }
                HStack {
                    Text("Slumpfrö"); Spacer()
                    TextField("Slumpfrö", value: $options.seed, format: .number.grouping(.never))
                        .multilineTextAlignment(.trailing).frame(width: 110).textFieldStyle(.roundedBorder)
                }
                Text("Top-p och slumpfrö används när temperaturen är större än 0. API:t använder dessa standardvärden vid start; anrop kan ange egna värden.")
                    .font(.caption).foregroundStyle(.secondary)
                Text("Giltiga värden sparas direkt, även när du byter tänkläge eller stänger appen. Både komma och punkt fungerar.")
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

/// Keep an incomplete keyboard draft (e.g. `0,`) separate from the persisted
/// numeric value. The formatted-value TextField used to fight decimal edits.
private struct OrnithDecimalField: View {
    let title: String
    @Binding var value: Double
    let minimum: Double
    let maximum: Double
    var excludingMinimum = false
    @Environment(\.locale) private var locale
    @State private var draft = ""
    @FocusState private var focused: Bool
    private var parsed: Double? {
        LocalSamplingInput.decimal(draft, minimum: minimum, maximum: maximum, excludingMinimum: excludingMinimum)
    }
    private var formatted: String { value.formatted(.number.grouping(.never).precision(.fractionLength(0...8)).locale(locale)) }
    var body: some View {
        VStack(alignment: .trailing, spacing: 4) {
            TextField(title, text: $draft)
                .focused($focused).onSubmit { focused = false }
                .multilineTextAlignment(.trailing).frame(width: 100).textFieldStyle(.roundedBorder)
                .accessibilityLabel(title)
                .accessibilityIdentifier(title == "Temperatur" ? "ornith-temperature" : "ornith-top-p")
                #if os(iOS)
                .keyboardType(.decimalPad)
                .toolbar { if focused { ToolbarItemGroup(placement: .keyboard) { Spacer(); Button("Klar") { focused = false } } } }
                #endif
            if focused && parsed == nil {
                Text("Ange \(excludingMinimum ? ">" : "")\(minimum.formatted())–\(maximum.formatted())")
                    .font(.caption2).foregroundStyle(.secondary)
            }
        }
        .onAppear { draft = formatted }
        .onChange(of: draft) { _, _ in if focused, let parsed, parsed != value { value = parsed } }
        .onChange(of: value) { _, _ in if !focused { draft = formatted } }
        .onChange(of: focused) { _, active in if !active { draft = formatted } }
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
                if model.isBusy && !model.isAPIRunning {
                    Text(model.status).font(.callout)
                    ProgressView().accessibilityLabel("Förbereder modellen")
                    Button("Stoppa förberedelsen") { model.stop() }
                }
                if let error = model.error { Text(error).foregroundStyle(.red).font(.caption) }
                if (!model.chatReady || model.batchSettingsChanged) && !model.isAPIRunning {
                    Text("Förbered samma verifierade motor som används i snabbchatten.").font(.callout)
                    Button("Ladda modell") { activate() }.buttonStyle(.borderedProminent).disabled(!model.canStart)
                }
                HStack {
                    Text("Port"); Spacer()
                    TextField("Port", value: $model.apiPort, format: .number.grouping(.never))
                        .multilineTextAlignment(.trailing).frame(width: 100).textFieldStyle(.roundedBorder)
                        .disabled(!model.apiAccessKey.isEmpty || !model.canStart)
                }
                if !model.isAPIListening {
                    SecureField("Egen API-nyckel (valfritt)", text: $model.apiKeyDraft)
                        .textFieldStyle(.roundedBorder).privacySensitive()
                        .disabled(!model.canStart)
                        .accessibilityIdentifier("ornith-api-key-input")
                    Text("Tomt fält använder den sparade nyckeln eller skapar en ny. Egen nyckel: 32–128 tecken utan mellanslag. Sparas i enhetens nyckelring.")
                        .font(.caption).foregroundStyle(.secondary)
                    Button("Skapa ny API-nyckel", systemImage: "arrow.clockwise") { model.rotateAPIKey() }
                        .disabled(!model.canStart)
                    Button("Starta API-server") { model.startAPI() }.buttonStyle(.borderedProminent)
                        .disabled(!model.canStart || !model.chatReady || model.batchSettingsChanged)
                } else {
                    Button("Stoppa API-server", role: .destructive) { model.stopAPI() }.buttonStyle(.bordered)
                }
                if model.isAPIRunning, !model.apiURLs.isEmpty {
                    OrnithAgentConnectionControls(urls: model.apiURLs, contextTokens: model.contextCapacityTokens,
                        copyKey: { copy(model.apiAccessKey) })
                } else {
                    Text("Starta API:t så visas färdiga datorkommandon för OpenCode, Codex, Hermes och PentAGI här. Adress, port och modell fylls i automatiskt.")
                        .font(.callout)
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
                    Text("Nyckeln sparas i enhetens nyckelring och fungerar efter omstart. Ange den som API key i klienten. Stoppa API:t för att byta nyckel.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Toggle("Behåll skärmen vaken", isOn: $model.apiKeepAwake)
                Text("Modellnamn").font(.caption).foregroundStyle(.secondary)
                Text(LocalChatRequest.modelID).font(.callout.monospaced()).textSelection(.enabled)
                Text("Samma Wi-Fi eller Ethernet via USB-C-adapter. HTTP används på det betrodda lokala nätverket. Vanlig USB-kabel kräver separat portvidarebefordran eller nätverksdrivrutin på datorn; den vägen är ännu inte verifierad.")
                    .font(.callout)
                Text("Chatten och API:t delar en modell. Vid appbyte behålls en ledig modell och servern pausas; iOS kan suspendera eller avsluta processen. Pågående inferens avbryts säkert. Återgå till appen för att återuppta. Minnestryck kan kräva urladdning.")
                    .font(.caption).foregroundStyle(.secondary)
                Text("Stöd: /v1/models, /v1/chat/completions, strömmande svar och validerade function/tool calls. Datorn utför verktygen och skickar tillbaka resultatet. Resonemang skickas separat som reasoning_content.")
                    .font(.caption).foregroundStyle(.secondary)
                Text("\(model.apiCompletedRequests) slutförda API-anrop").font(.caption.monospacedDigit())
                if let summary = model.apiLastSummary {
                    Text(String(format: "%.2f token/s · prompt %.2f s · minne %.2f GB", summary.decodeTokensPerSecond ?? 0,
                        summary.prefillSeconds, Double(summary.maximumObservedFootprintBytes) / 1e9))
                        .font(.caption.monospacedDigit())
                    Text("Kontext \(summary.promptTokenIDs.count) + \(summary.generatedTokenIDs.count) / \(model.contextCapacityTokens) · \(summary.reusedPromptTokens) återanvända token")
                        .font(.caption.monospacedDigit())
                }
                OrnithGenerationControls(options: $model.chatOptions, contextTokens: model.contextCapacityTokens).disabled(!model.canStart || model.isAPIRunning)
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

struct OrnithAgentConnectionControls: View {
    let urls: [String]
    let contextTokens: Int
    var copyKey: () -> Void = {}
    @State private var client: LocalAgentSetup.Client = .opencode
    @State private var computer: LocalAgentSetup.Computer = .unix
    @State private var selectedURL = ""
    @State private var copied: String?
    init(urls: [String], contextTokens: Int, copyKey: @escaping () -> Void = {},
         initialClient: LocalAgentSetup.Client = .opencode, initialComputer: LocalAgentSetup.Computer = .unix) {
        self.urls = urls; self.contextTokens = contextTokens; self.copyKey = copyKey
        self._client = State(initialValue: initialClient); self._computer = State(initialValue: initialComputer)
    }
    private var setup: LocalAgentSetup? {
        try? LocalAgentSetup(client: client, computer: computer,
            baseURL: urls.contains(selectedURL) ? selectedURL : (urls.first ?? ""),
            model: LocalChatRequest.modelID, contextTokens: contextTokens)
    }
    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 14) {
                Text("Koppla datorn till telefonen").font(.headline)
                Picker("Agent", selection: $client) {
                    ForEach(LocalAgentSetup.Client.allCases) { Text($0.title).tag($0) }
                }
                Picker("Dator", selection: $computer) {
                    ForEach(LocalAgentSetup.Computer.allCases) { Text($0.title).tag($0) }
                }
                if urls.count > 1 {
                    Picker("Telefonadress", selection: $selectedURL) {
                        ForEach(urls, id: \.self) { Text($0).tag($0) }
                    }.onAppear { if !urls.contains(selectedURL) { selectedURL = urls.first ?? "" } }
                }
                if let setup {
                    Text("1. Samma nätverk. Låt Flash Local vara öppet. Datorn behöver Python 3.11+, Git och \(client.title) installerat.")
                        .font(.callout)
                    DisclosureGroup("Första gången: hämta anslutningshjälpen") {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Kör detta en gång i datorns terminal. Det hämtar en testad version av hjälpskripten, inga modellvikter. Har du redan hämtat den här versionen går du till steg 2.")
                                .font(.caption).foregroundStyle(.secondary)
                            Text(setup.installCommand).font(.caption.monospaced()).textSelection(.enabled)
                            Button("Kopiera installation", systemImage: "doc.on.doc") { copy(setup.installCommand, label: "Installationen kopierad") }
                        }.padding(.top, 6)
                    }
                    if let reason = setup.unavailableReason {
                        Text(reason).font(.callout).foregroundStyle(.orange)
                    } else if let command = setup.startCommand {
                        Text("2. Kör på datorn för att starta \(client.title):").font(.callout.bold())
                        if client == .pentagi {
                            Text("Öppna först terminalen i din befintliga PentAGI-mapp. Docker Compose krävs; hjälpen installerar inte PentAGI.")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        Text(command).font(.caption.monospaced()).textSelection(.enabled)
                        Button("Kopiera startkommando", systemImage: "terminal") { copy(command, label: "Startkommandot kopierat") }
                            .buttonStyle(.borderedProminent).accessibilityIdentifier("ornith-copy-agent-command")
                        Text("3. Klistra in API-nyckeln när datorn frågar. Den visas inte i terminalen och sparas inte av hjälpskriptet.")
                            .font(.callout)
                        Button("Kopiera API-nyckel", systemImage: "key") { copyKey(); copied = "API-nyckeln kopierad" }
                    }
                    Text("\(setup.baseURL) · \(setup.model) · \(contextTokens.formatted()) kontexttoken")
                        .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                } else {
                    Text("Ingen giltig lokal adress ännu. Anslut telefonen till Wi-Fi eller Ethernet och starta om API:t.")
                        .font(.callout)
                }
                if let copied { Text(copied).font(.caption).foregroundStyle(.secondary).accessibilityIdentifier("ornith-agent-copy-status") }
                Text("Hjälpen läser modellens aktuella gränser från telefonen. Codex använder en lokal Responses-adapter på datorn. Agenten utför sina verktyg på datorn; modellens inferens sker på telefonen.")
                    .font(.caption).foregroundStyle(.secondary)
            }.frame(maxWidth: .infinity, alignment: .leading)
        }.accessibilityIdentifier("ornith-agent-connection")
        .onChange(of: client) { _, _ in copied = nil }
        .onChange(of: computer) { _, _ in copied = nil }
        .onChange(of: selectedURL) { _, _ in copied = nil }
    }
    private func copy(_ value: String, label: String) {
        #if os(iOS)
        UIPasteboard.general.string = value
        #else
        NSPasteboard.general.clearContents(); NSPasteboard.general.setString(value, forType: .string)
        #endif
        copied = label
    }
}
#endif
