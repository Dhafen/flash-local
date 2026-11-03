import MinirunKit
import SwiftUI
#if os(iOS)
import UIKit
#elseif os(macOS)
import AppKit
#endif

/// Always-visible API tab for the baseline Ornith runtime. The same local
/// inference actor serves chat and HTTP; no cloud or experimental module.
struct REAMLocalAPIPage: View {
    @Bindable var model: REAMAppModel
    @State private var revealKey = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                Label("Lokal AI-server", systemImage: "network")
                    .font(.title2.bold())
                Text(model.apiStatus)
                    .font(.subheadline)
                    .accessibilityIdentifier("flash-api-status")
                if let error = model.error {
                    Text(error).font(.caption).foregroundStyle(.red).textSelection(.enabled)
                }
                if model.selectedModel != .ornith {
                    Label("Den här versionens lokala API kan köra Ornith. Välj Ornith under Inställningar.", systemImage: "info.circle")
                        .font(.callout)
                }
                GroupBox("Modell") {
                    VStack(alignment: .leading, spacing: 10) {
                        Text(model.modelTitle).font(.subheadline.bold())
                        Text(model.canChat ? "Verifierad och redo" : model.status)
                            .font(.caption).foregroundStyle(.secondary)
                        if !model.canChat && !model.isBusy {
                            Button(model.hasFile ? "Verifiera modellfil" : "Hämta modell") {
                                if model.hasFile { model.verify() } else { model.download() }
                            }
                            .buttonStyle(.bordered)
                        } else if model.isBusy {
                            ProgressView("Modellarbete pågår…")
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                GroupBox("Anslutning") {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Text("Port")
                            Spacer()
                            TextField("Port", value: $model.apiPort, format: .number.grouping(.never))
                                .keyboardType(.numberPad)
                                .multilineTextAlignment(.trailing)
                                .frame(width: 100)
                                .textFieldStyle(.roundedBorder)
                                .disabled(model.isAPIListening)
                        }
                        if !model.isAPIListening {
                            SecureField("Egen API-nyckel (valfritt)", text: $model.apiKeyDraft)
                                .textFieldStyle(.roundedBorder)
                                .privacySensitive()
                            Button("Skapa en ny API-nyckel", systemImage: "arrow.clockwise") {
                                model.rotateAPIKey()
                            }
                            .disabled(model.isBusy)
                            Button("Starta API-server", systemImage: "play.fill") { model.startAPI() }
                                .buttonStyle(.borderedProminent)
                                .disabled(!model.canStartAPI)
                                .accessibilityIdentifier("flash-api-start")
                        } else {
                            Button("Stoppa API-server", role: .destructive) { model.stopAPI() }
                                .buttonStyle(.bordered)
                                .accessibilityIdentifier("flash-api-stop")
                        }
                        if !model.apiURLs.isEmpty {
                            ForEach(model.apiURLs, id: \.self) { url in
                                VStack(alignment: .leading, spacing: 6) {
                                    Text(url).font(.footnote.monospaced()).textSelection(.enabled)
                                    Button("Kopiera API-adress", systemImage: "doc.on.doc") { copy(url) }
                                }
                            }
                            Text("GET /v1/models och POST /v1/chat/completions med Bearer-nyckel. Strömmande SSE stöds.")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        if !model.apiAccessKey.isEmpty {
                            Divider()
                            Text("API-nyckel").font(.subheadline.bold())
                            if revealKey {
                                Text(model.apiAccessKey).font(.caption.monospaced())
                                    .textSelection(.enabled).privacySensitive()
                            }
                            HStack {
                                Button(revealKey ? "Dölj" : "Visa") { revealKey.toggle() }
                                Button("Kopiera nyckel", systemImage: "key") {
                                    copy(model.apiAccessKey)
                                }
                            }
                            Text("Nyckeln lagras i iPhones nyckelring och återanvänds efter omstart.")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                Text("\(model.apiCompletedRequests) slutförda anrop · \(model.contextTokens) token kontext")
                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                Text("Servern fungerar endast medan Flash Local är aktiv i förgrunden på samma lokala Wi-Fi/Ethernet. iOS kan pausa nätverkstrafik i bakgrunden. Vanlig USB-kabel kräver separat portvidarebefordran.")
                    .font(.footnote).foregroundStyle(.secondary)
                Text("Den här standardmotorn stöder bara textbaserad Ornith-inferens och greedy sampling. Avancerade verktygsanrop och fler modeller kräver en verifierad kompatibel motor.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            .padding(20)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func copy(_ value: String) {
        #if os(iOS)
        UIPasteboard.general.string = value
        #elseif os(macOS)
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(value, forType: .string)
        #endif
    }
}
