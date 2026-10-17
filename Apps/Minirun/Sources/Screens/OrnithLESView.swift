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
    var body: some View {
        NavigationStack {
            OrnithLESPage(model: model, start: { model.start(existing: existing) })
                .navigationTitle("Ornith LES")
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
                Text("Först jämförs hela modellens logits med CPU-referensen vid fyra positioner. Därefter körs samma fasta prompt med vanlig motor, expertcache, LES-förläsning och OS-cache. Varje läge körs två gånger med minst 128 genererade token. Låt appen vara öppen; det kan ta flera minuter.")
                    .font(.footnote).foregroundStyle(.secondary)
                if let check = model.numericalCheck {
                    Text(check.passed ? "Kort referenskontroll godkänd." : "Referenskontrollen avvek. Ingen ny motor aktiverades.")
                        .font(.callout.bold()).foregroundStyle(check.passed ? Color.primary : Color.red)
                }
                ForEach(Array(model.runs.enumerated()), id: \.offset) { _, run in
                    VStack(alignment: .leading, spacing: 5) {
                        Text("\(run.backend.rawValue) · \(run.applicationCache)").font(.headline)
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
#endif
