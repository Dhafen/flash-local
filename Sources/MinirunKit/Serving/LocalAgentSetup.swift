import Foundation

/// Copyable commands contain the live address/model, never an API credential.
/// The pinned client revision has independent transport controls; it discovers
/// actual capabilities from /models before starting any agent.
public struct LocalAgentSetup {
    public enum Client: String, CaseIterable, Identifiable {
        case opencode, codex, hermes, pentagi
        public var id: String { rawValue }
        public var title: String {
            switch self { case .opencode: "OpenCode"; case .codex: "Codex"; case .hermes: "Hermes"; case .pentagi: "PentAGI" }
        }
    }
    public enum Computer: String, CaseIterable, Identifiable {
        case unix, windows
        public var id: String { rawValue }
        public var title: String { self == .unix ? "Kali / Linux / Mac" : "Windows PowerShell" }
    }
    public static let clientRevision = "39c262c646ff3774f60e31c5b6f26b690e4dd01c"
    private var clientDirectory: String { "flash-local-clients-" + String(Self.clientRevision.prefix(8)) }
    public let client: Client
    public let computer: Computer
    public let baseURL: String
    public let model: String
    public let contextTokens: Int
    public init(client: Client, computer: Computer, baseURL: String, model: String, contextTokens: Int) throws {
        guard let url = URLComponents(string: baseURL), url.scheme == "http", url.user == nil, url.password == nil,
              url.query == nil, url.fragment == nil, url.path == "/v1", let host = url.host,
              let port = url.port, (1...65535).contains(port) else {
            throw LocalAPIError.invalid("base_url", "Starta API:t och välj telefonens lokala adress.")
        }
        let octets = host.split(separator: ".", omittingEmptySubsequences: false).compactMap { UInt8($0) }
        guard octets.count == 4, host.split(separator: ".").count == 4,
              octets[0] == 10 || (octets[0] == 172 && (16...31).contains(octets[1])) ||
                (octets[0] == 192 && octets[1] == 168) || (octets[0] == 169 && octets[1] == 254),
              !model.isEmpty, model.utf8.count <= 128,
              model.utf8.allSatisfy({ (48...57).contains($0) || (65...90).contains($0) || (97...122).contains($0) || [45, 46, 95].contains($0) }),
              LocalContextProfile(rawValue: contextTokens) != nil else {
            throw LocalAPIError.invalid("connection", "Ogiltig lokal adress, modell eller kontext.")
        }
        self.client = client; self.computer = computer; self.baseURL = baseURL; self.model = model; self.contextTokens = contextTokens
    }
    public var unavailableReason: String? {
        client == .hermes && contextTokens < 64000
            ? "Den här Hermes-versionen kräver minst 64 000 kontexttoken. Aktiv modell har \(contextTokens.formatted()). 65K-profilen är experimentell och måste testas på telefonen först." : nil
    }
    private var folder: String {
        computer == .unix ? "\"$HOME/\(clientDirectory)\"" : "\"$env:USERPROFILE\\\(clientDirectory)\""
    }
    public var installCommand: String {
        // Chaining makes checkout conditional on a successful, new clone. PS 5
        // has no &&, so use its explicit exit-code check.
        let clone = "git clone --filter=blob:none --no-checkout https://github.com/Dhafen/flash-local.git \(folder)"
        let checkout = "git -C \(folder) checkout --detach \(Self.clientRevision)"
        return computer == .unix ? "\(clone) &&\n\(checkout)" : "\(clone)\nif ($LASTEXITCODE -eq 0) { \(checkout) }"
    }
    public var startCommand: String? {
        guard unavailableReason == nil else { return nil }
        let script = computer == .unix ? "\"$HOME/\(clientDirectory)/scripts/flash_local.py\""
            : "\"$env:USERPROFILE\\\(clientDirectory)\\scripts\\flash_local.py\""
        let python = computer == .unix ? "python3" : "py -3"
        let project = client == .pentagi ? " --project ." : ""
        return "\(python) \(script) --base-url '\(baseURL)' --model '\(model)' run \(client.rawValue)\(project)"
    }
}
