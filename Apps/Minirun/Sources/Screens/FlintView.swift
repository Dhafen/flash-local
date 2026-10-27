#if canImport(FlintEngine)
import FlintEngine
import MinirunKit
import SwiftUI
#if os(iOS)
import UIKit
#elseif os(macOS)
import AppKit
#endif

struct FlintView: View {
    @Bindable var model: FlintAppModel
    @Environment(\.scenePhase) private var scenePhase
    private enum Pane: String, CaseIterable { case chat = "Chatt", api = "API", measurements = "Prestanda" }
    @State private var pane: Pane = .chat
    var body: some View {
        VStack(spacing: 0) {
            Picker("Flint", selection: $pane) {
                ForEach(Pane.allCases, id: \.self) { Text($0.rawValue).tag($0) }
            }.pickerStyle(.segmented).padding(.horizontal)
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    FlintModelControls(model: model)
                    if pane == .chat { FlintChatPage(model: model) }
                    if pane == .api { FlintAPIControls(model: model) }
                    if pane == .measurements { FlintMeasurementPage(model: model) }
                }.padding()
            }
        }
        .navigationTitle("Flint · experiment")
        .onChange(of: scenePhase, initial: true) { _, value in model.setActive(value == .active) }
        #if os(iOS)
        .onChange(of: model.isBusy || (model.apiListening && model.sceneIsActive), initial: true) { _, awake in
            UIApplication.shared.isIdleTimerDisabled = awake
        }
        .onDisappear { UIApplication.shared.isIdleTimerDisabled = false }
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.didReceiveMemoryWarningNotification)) { _ in
            Task { await model.unloadModel() }
        }
        #endif
    }
}

private struct FlintModelControls: View {
    @Bindable var model: FlintAppModel
    @State private var temperature = ""
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Recursive Flint · 40B-A2.5B").font(.headline)
            Text("Officiell Q4_1 + F32 · 25,99 GB · intern lagring").font(.caption).foregroundStyle(.secondary)
            Text(model.isShuttingDown ? "Laddar ur…" : model.status).accessibilityIdentifier("flint-status")
            if model.isBusy {
                if let fraction = model.fraction {
                    ProgressView(value: fraction).accessibilityLabel("Aktuell laddningsfas")
                    Text(fraction, format: .percent.precision(.fractionLength(0))).font(.caption)
                } else { ProgressView() }
                if let start = model.loadingStartedAt {
                    TimelineView(.periodic(from: start, by: 1)) { context in
                        Text("Förfluten tid: \(Int(context.date.timeIntervalSince(start))) s").font(.caption)
                    }
                }
                Button("Stoppa", role: .cancel) { model.stop() }
            }
            if let error = model.error { Text(error).font(.callout).foregroundStyle(.red).textSelection(.enabled) }
            if let url = model.numericalReportURL { ShareLink("Dela numerikkontroll", item: url) }
            if !model.hasModel {
                Button("Hämta Flint · 25,99 GB") { model.download() }.disabled(!model.canStart)
                Text("Återupptagbar hämtning direkt till appens lagring. Hela filens SHA256 kontrolleras innan den används.")
                    .font(.caption).foregroundStyle(.secondary)
            } else {
                HStack {
                    Button("Ladda modell") { model.load() }.buttonStyle(.borderedProminent).disabled(!model.canStart || model.ready)
                    Button("Ladda ur modell") { Task { await model.unloadModel() } }.buttonStyle(.bordered)
                        .disabled(model.isShuttingDown || (!model.ready && !model.isBusy))
                }
            }
            DisclosureGroup("Inställningar") {
                Picker("Kontextkapacitet", selection: $model.contextProfile) {
                    ForEach([LocalContextProfile.standard, .test16K, .test32K]) { Text($0.title).tag($0) }
                }.disabled(model.ready || model.isBusy || model.apiListening)
                Text("65K ryms inte i nuvarande konservativa minnesbudget. Vald kapacitet är ingen verifiering av kontextkvalitet.")
                    .font(.caption).foregroundStyle(.secondary)
                TextField("Temperatur", text: $temperature).textFieldStyle(.roundedBorder)
                    .disabled(model.isBusy || model.apiListening)
                    .onChange(of: temperature) { _, value in
                        if let number = LocalSamplingInput.decimal(value, minimum: 0, maximum: 2) {
                            model.options.temperature = number; model.savePreferences()
                        }
                    }
                Text("Standard: temperatur 1, top-p 0,95, top-k 20 enligt utgivaren. 0 ger greedy. Flint styr sitt eget resonemang; påtvingad thinking-budget är inte verifierad.")
                    .font(.caption).foregroundStyle(.secondary)
                Text("Systemprompt").font(.subheadline)
                TextEditor(text: $model.systemPrompt).frame(minHeight: 80).disabled(model.isBusy || model.apiListening)
                    .onChange(of: model.systemPrompt) { _, _ in model.savePreferences() }
            }.onAppear { temperature = String(model.options.temperature) }
            Text("Första aktiveringen jämför fyra verkliga logitpositioner och beräknade top-6-expertval mellan CPU och Metal. Godkänd kontroll sparas per modell, appbygge och inställning. Full modellkvalitet och lång kontext behöver fortfarande testas.")
                .font(.caption2).foregroundStyle(.secondary)
        }
    }
}

private struct FlintChatPage: View {
    @Bindable var model: FlintAppModel
    var body: some View {
        LazyVStack(alignment: .leading, spacing: 20) {
            ForEach(model.history) { turn in
                VStack(alignment: .leading, spacing: 12) {
                    Text("Du").font(.caption).foregroundStyle(.secondary)
                    Text(turn.prompt).textSelection(.enabled)
                    Text("Flash Local · Flint").font(.caption).foregroundStyle(.secondary)
                    if !turn.reasoning.isEmpty {
                        DisclosureGroup("Resonemang") { Text(turn.reasoning).font(.callout).textSelection(.enabled) }
                    }
                    AnswerText(source: turn.answer, caret: nil)
                    if !turn.completed { Text(model.isBusy ? "Arbetar…" : "Avbrutet svar").font(.caption).foregroundStyle(.secondary) }
                }.frame(maxWidth: .infinity, alignment: .leading)
            }
            HStack(alignment: .bottom) {
                TextField("Skriv till Flint…", text: $model.prompt, axis: .vertical).lineLimit(1...8)
                Button("Skicka", systemImage: "arrow.up.circle.fill") { model.send() }
                    .labelStyle(.iconOnly).disabled(!model.canSend || model.prompt.isEmpty)
            }
            Button("Rensa Flint-chatten", role: .destructive) { model.clearChat() }.disabled(!model.canStart)
        }
    }
}

private struct FlintMeasurementPage: View {
    @Bindable var model: FlintAppModel
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Verkliga mätvärden från senaste körningen").font(.headline)
            HStack {
                ForEach([1000, 4000, 10000], id: \.self) { count in
                    Button("Testa \(count / 1000)K") { model.benchmark(promptTokens: count) }
                        .disabled(!model.canSend || model.apiListening || count + 256 > model.contextProfile.rawValue)
                }
            }
            Text("En ofarlig fast prompt med exakt uppmätt tokenantal och högst 256 nya token. Kör på sval telefon, upprepa och spara rapporterna. Minst 128 utdata-token krävs för en jämförbar decode-mätning. Chatthistoriken behålls; den gemensamma KV-sessionen används till testet.")
                .font(.caption).foregroundStyle(.secondary)
            if let report = model.report {
                LabeledContent("Nya / återanvända prompttoken", value: "\(report.evaluatedPromptTokens) / \(report.reusedPromptTokens)")
                LabeledContent("Prompttid", value: String(format: "%.2f s", report.prefillSeconds))
                if report.prefillSeconds > 0 { LabeledContent("Prompt", value: String(format: "%.2f token/s", Double(report.evaluatedPromptTokens) / report.prefillSeconds)) }
                if let speed = report.decodeTokensPerSecond { LabeledContent("Generering", value: String(format: "%.2f token/s", speed)) }
                if let time = report.timeToFirstTokenSeconds { LabeledContent("Första modelltoken", value: String(format: "%.2f s", time)) }
                LabeledContent("Observerad RAM-topp", value: String(format: "%.2f GB", Double(report.maximumObservedFootprintBytes) / 1e9))
                LabeledContent("Explicit expertläsning", value: String(format: "%.2f GB", Double(report.explicitExpertBytesRead) / 1e9))
                Button("Spara mätning i Filer") { model.saveReport() }
                if let url = model.reportURL { ShareLink("Dela mätning", item: url) }
            } else { Text("Ingen modellkörning har mätts ännu.").foregroundStyle(.secondary) }
            Text("I/O räknar begärda expertbytes, även återläsningar. Det är inte modellens storlek eller fysisk NAND-trafik. OS-filcache och embeddingars mmap-fel ingår inte. Termiskt tillstånd finns i rapporten; temperatur i grader exponeras inte av iOS.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }
}

private struct FlintAPIControls: View {
    @Bindable var model: FlintAppModel
    @State private var client: LocalAgentSetup.Client = .opencode
    @State private var computer: LocalAgentSetup.Computer = .windows
    private var setup: LocalAgentSetup? {
        guard let url = model.apiURLs.first else { return nil }
        return try? .init(client: client, computer: computer, baseURL: url,
            model: LocalModelCapabilities(family: .flint).modelID, contextTokens: model.contextProfile.rawValue)
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(model.apiStatus).accessibilityIdentifier("flint-api-status")
            HStack {
                Button("Starta API-server") { model.startAPI() }.disabled(!model.canSend || model.apiListening)
                Button("Stoppa API-server") { model.stopAPI() }.disabled(!model.apiListening)
            }
            if !model.apiListening { SecureField("Egen API-nyckel · valfritt", text: $model.apiKeyDraft).textFieldStyle(.roundedBorder) }
            ForEach(model.apiURLs, id: \.self) { Text($0).font(.system(.callout, design: .monospaced)).textSelection(.enabled) }
            if !model.apiAccessKey.isEmpty { Button("Kopiera API-nyckeln") { Self.copy(model.apiAccessKey) } }
            Picker("Dator", selection: $computer) { ForEach(LocalAgentSetup.Computer.allCases) { Text($0.title).tag($0) } }
            Picker("Agentverktyg", selection: $client) { ForEach(LocalAgentSetup.Client.allCases) { Text($0.title).tag($0) } }
            if let setup {
                Text("1. Hämta startskriptet").font(.headline)
                Text(setup.installCommand).font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                Button("Kopiera installation") { Self.copy(setup.installCommand) }
                if let command = setup.startCommand {
                    Text("2. Anslut till telefonen").font(.headline)
                    Text(command).font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                    Button("Kopiera startkommando") { Self.copy(command) }
                }
                if let reason = setup.unavailableReason { Text(reason).foregroundStyle(.orange) }
            }
            Text("Telefon och dator behöver samma lokala nätverk, till exempel Wi-Fi eller Ethernet. Håll Flash Local öppen; iOS kan suspendera servern i bakgrunden. Datorn utför verktyg inom din uttryckligen tillåtna testscope. Telefonen kör inferensen och utför inga säkerhetskommandon.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }
    private static func copy(_ value: String) {
        #if os(iOS)
        UIPasteboard.general.string = value
        #elseif os(macOS)
        NSPasteboard.general.clearContents(); NSPasteboard.general.setString(value, forType: .string)
        #endif
    }
}
#endif
