import MinirunKit
import OrnithPerformance
import SwiftUI
#if os(iOS)
import UIKit
#endif

struct OrnithPerformanceView: View {
    @Bindable var model: OrnithPerformanceAppModel
    @Environment(\.dismiss) private var dismiss
    @Environment(\.scenePhase) private var scenePhase
    var body: some View {
        NavigationStack {
            OrnithPerformancePage(model: model)
                .navigationTitle("Ornith Performance")
                .toolbar {
                    ToolbarItem(placement: .cancellationAction) {
                        Button("Stäng") { Task { await model.shutdown(); dismiss() } }
                    }
                    ToolbarItem(placement: .automatic) {
                        Button("Ny chatt", systemImage: "square.and.pencil") { model.clearMessages() }
                            .disabled(model.isBusy || model.messages.isEmpty)
                    }
                }
        }
        .interactiveDismissDisabled()
        .task { model.prepare() }
        .onChange(of: scenePhase) { _, phase in
            if phase == .background { Task { await model.shutdown() } }
        }
        #if os(iOS)
        .onChange(of: model.isBusy, initial: true) { _, busy in UIApplication.shared.isIdleTimerDisabled = busy }
        .onDisappear { UIApplication.shared.isIdleTimerDisabled = false }
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.didReceiveMemoryWarningNotification)) { _ in
            Task { await model.shutdown() }
        }
        #endif
    }
}

/// No network request, model allocation, or invented throughput in rendering.
struct OrnithPerformancePage: View {
    @Bindable var model: OrnithPerformanceAppModel
    var body: some View {
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(alignment: .leading, spacing: 20) {
                        modelCard
                        Divider()
                        ForEach(model.messages) { message in
                            VStack(alignment: .leading, spacing: 6) {
                                Text(message.role == .user ? "Du" : "Flash Local").font(.caption.bold()).foregroundStyle(.secondary)
                                Text(message.text.isEmpty ? "…" : message.text).textSelection(.enabled)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                if message.interrupted && !model.isBusy { Text("Avbrutet svar").font(.caption).foregroundStyle(.secondary) }
                            }.id(message.id)
                        }
                        Color.clear.frame(height: 1).id("latest-qpack")
                    }.padding(20)
                }
                .onChange(of: model.messages.last?.text) { _, _ in proxy.scrollTo("latest-qpack", anchor: .bottom) }
            }
            Divider()
            HStack(alignment: .bottom, spacing: 12) {
                TextField("Skriv ett meddelande…", text: $model.draft, axis: .vertical)
                    .lineLimit(1...6).textFieldStyle(.plain).submitLabel(.send)
                    .onSubmit { model.send() }.disabled(!model.canSend)
                    .accessibilityLabel("Meddelande till Ornith QPack")
                Button { model.send() } label: { Image(systemName: "arrow.up.circle.fill").font(.title) }
                    .disabled(!model.canSend || model.draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    .accessibilityLabel("Skicka till Ornith QPack")
            }.padding(16)
        }.background(MRColor.abyss).tint(MRColor.tierPinned)
    }
    private var modelCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Ornith 1.5 · 35B-A3B · MLX 4-bit").font(.headline)
            Text("Testläge · intern lagring · 8K kontextgräns · 5,5 GB RAM-gräns")
                .font(.caption).foregroundStyle(.secondary)
            Text("Version \(Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "–") (\(Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "–"))")
                .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            Text(model.status).font(.caption).accessibilityIdentifier("ornith-qpack-status")
            if let error = model.error { Text(error).font(.caption).foregroundStyle(.red).textSelection(.enabled) }
            if model.isBusy {
                HStack(spacing: 12) {
                    if let progress = model.progress, model.activity == .installing {
                        VStack(alignment: .leading, spacing: 4) {
                            ProgressView(value: Double(progress.completedSourceBytes), total: Double(max(1, progress.totalSourceBytes)))
                            Text("\(bytes(UInt64(progress.completedSourceBytes))) av \(bytes(UInt64(progress.totalSourceBytes)))")
                                .font(.caption.monospacedDigit())
                        }
                    } else { ProgressView().controlSize(.small) }
                    Spacer()
                    Button("Stoppa", role: .cancel) { model.stop() }.disabled(model.activity == .stopping)
                }
            } else if !model.isReady {
                Button(model.hasInstalledFiles ? "Öppna installerad modell" : "Hämta och installera · 19,51 GB") { model.installAndOpen() }
                    .buttonStyle(.borderedProminent).disabled(model.isPreview)
                Text("Kräver cirka 20 GB ledigt utöver befintliga modeller. Låt appen vara öppen. Efter installation jämförs den optimerade beräkningen med originalet innan den aktiveras.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if model.isReady {
                Text(model.kernelMode == .groupedExperts ? "Motor: optimerad expertberäkning" : "Motor: original")
                    .font(.caption.bold()).accessibilityIdentifier("ornith-kernel-mode")
                if model.kernelMode == .groupedExperts {
                    Button("Använd originalmotor för jämförelse") { model.useOriginalKernels() }
                        .disabled(!model.canCheckKernels)
                } else {
                    Button("Kontrollera och aktivera optimering") { model.checkAndActivateKernels() }
                        .disabled(!model.canCheckKernels)
                }
            }
            if let check = model.kernelCheck {
                Text(check.passed ? "Kort kontroll godkänd: identiska logits vid \(check.comparedLogitVectors) positioner." : "Kontrollen avvek. Originalberäkningen används.")
                    .font(.caption).foregroundStyle(check.passed ? Color.secondary : Color.red)
            }
            if let plan = model.memoryPlan {
                Text("Fasta vikter \(bytes(plan.residentDensePayloadBytes)) · expertcache \(bytes(plan.expertCacheBytes))")
                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            }
            if let report = model.report {
                if let speed = report.decodeForwardTokensPerSecond {
                    Text(String(format: "%.2f token/s · RAM-topp %@", speed, bytes(report.maximumObservedFootprintBytes)))
                        .font(.caption.monospacedDigit())
                }
                Text(String(format: "Prompt %.2f s · %d återanvända token · expertläsning %@", report.prefillSeconds,
                            report.reusedPromptTokens, bytes(report.expertReturnedReadBytes)))
                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            }
            if model.report != nil || model.kernelCheck != nil {
                Button("Spara mätning i Filer", systemImage: "chart.bar.doc.horizontal") { model.saveMeasurement() }
                    .disabled(model.isBusy)
                Text("Sparar kontroll, prestanda, minne och I/O lokalt. Ingen chatttext ingår.")
                    .font(.caption2).foregroundStyle(.secondary)
            }
            Text("Fullmodellens numerik, hastighet och långa kontext är ännu inte verifierade på iPhone.")
                .font(.caption2).foregroundStyle(.secondary)
        }
    }
    private func bytes(_ value: UInt64) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(clamping: value), countStyle: .file)
    }
}
