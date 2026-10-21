import Darwin
import Foundation
import Network
import Security

/// Foreground, authenticated HTTP service. Network state is confined to queue;
/// the supplied handler reserves the app's sole inference owner before headers.
public final class LocalModelAPIServer: @unchecked Sendable {
    public enum State: Sendable { case ready(port: UInt16, addresses: [String]), stopped, failed(String) }
    public typealias Handler = @Sendable (LocalChatRequest) async throws -> LocalChatGeneration
    private let queue = DispatchQueue(label: "flash-local.api", qos: .userInitiated)
    private let key: String
    private let handler: Handler
    private let onState: @Sendable (State) -> Void
    private let loopbackOnly: Bool
    private let defaults: LocalGenerationOptions
    private let capabilities: LocalModelCapabilities
    private var listener: NWListener?
    private var sessions: [UUID: LocalAPISession] = [:]
    private var stopped = false

    public static func makeAccessKey() throws -> String {
        var bytes = [UInt8](repeating: 0, count: 32)
        guard SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes) == errSecSuccess else {
            throw LocalAPIError(status: 500, message: "Could not create a local access key")
        }
        return bytes.map { String(format: "%02x", $0) }.joined()
    }
    public init(key: String, port: UInt16 = 8080, loopbackOnly: Bool = false, defaults: LocalGenerationOptions = .init(),
                capabilities: LocalModelCapabilities = .init(),
                handler: @escaping Handler, onState: @escaping @Sendable (State) -> Void) throws {
        try LocalAPIKeyStore.validate(key)
        try defaults.validate()
        self.key = key; self.handler = handler; self.onState = onState; self.loopbackOnly = loopbackOnly; self.defaults = defaults
        self.capabilities = capabilities
        let parameters = NWParameters.tcp
        parameters.prohibitedInterfaceTypes = [.cellular]
        parameters.includePeerToPeer = false
        let host: NWEndpoint.Host = .ipv4(loopbackOnly ? IPv4Address.loopback : IPv4Address.any)
        parameters.requiredLocalEndpoint = .hostPort(host: host, port: NWEndpoint.Port(rawValue: port)!)
        let listener = try NWListener(using: parameters)
        self.listener = listener
        listener.stateUpdateHandler = { [weak self] state in
            guard let self, !self.stopped else { return }
            switch state {
            case .ready:
                guard let port = self.listener?.port?.rawValue else { return }
                self.onState(.ready(port: port, addresses: Self.localIPv4Addresses(loopbackOnly: loopbackOnly)))
            case .failed:
                self.onState(.failed("API-servern kunde inte startas. Kontrollera port och behörighet till lokalt nätverk."))
                self.stopOnQueue(notify: false)
            default: break
            }
        }
        listener.newConnectionHandler = { [weak self] connection in
            guard let self, !self.stopped, self.sessions.count < 4,
                  case .hostPort(let host, _) = connection.endpoint,
                  Self.isLocalIPv4(String(describing: host), loopbackOnly: loopbackOnly) else {
                connection.cancel(); return
            }
            let id = UUID()
            let session = LocalAPISession(connection: connection, queue: self.queue, key: self.key, defaults: self.defaults, capabilities: self.capabilities, handler: self.handler) { [weak self] in
                self?.sessions.removeValue(forKey: id)
            }
            self.sessions[id] = session; session.start()
        }
        listener.start(queue: queue)
    }
    public func stop() { queue.async { self.stopOnQueue() } }
    private func stopOnQueue(notify: Bool = true) {
        guard !stopped else { return }; stopped = true
        listener?.cancel(); listener = nil
        let active = Array(sessions.values); sessions.removeAll()
        active.forEach { $0.finish() }; if notify { onState(.stopped) }
    }
    deinit {
        let ownedListener = listener, ownedSessions = Array(sessions.values)
        queue.async { ownedListener?.cancel(); ownedSessions.forEach { $0.finish() } }
    }

    static func isLocalIPv4(_ value: String, loopbackOnly: Bool = false) -> Bool {
        let pieces = value.split(separator: ".", omittingEmptySubsequences: false)
        guard pieces.count == 4 else { return false }
        let numbers = pieces.compactMap { UInt8($0) }
        guard numbers.count == 4 else { return false }
        if numbers[0] == 127 { return true }
        if loopbackOnly { return false }
        return numbers[0] == 10 || (numbers[0] == 172 && (16...31).contains(numbers[1])) ||
            (numbers[0] == 192 && numbers[1] == 168) || (numbers[0] == 169 && numbers[1] == 254)
    }
    public static func localIPv4Addresses(loopbackOnly: Bool = false) -> [String] {
        if loopbackOnly { return ["127.0.0.1"] }
        var list: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&list) == 0 else { return [] }; defer { freeifaddrs(list) }
        var values = Set<String>(), cursor = list
        while let item = cursor {
            defer { cursor = item.pointee.ifa_next }
            guard let address = item.pointee.ifa_addr, address.pointee.sa_family == UInt8(AF_INET),
                  item.pointee.ifa_flags & UInt32(IFF_UP) != 0 else { continue }
            var buffer = [CChar](repeating: 0, count: Int(INET_ADDRSTRLEN))
            address.withMemoryRebound(to: sockaddr_in.self, capacity: 1) { pointer in
                var ip = pointer.pointee.sin_addr
                if inet_ntop(AF_INET, &ip, &buffer, socklen_t(buffer.count)) != nil {
                    let value = String(cString: buffer)
                    if isLocalIPv4(value), !value.hasPrefix("127.") { values.insert(value) }
                }
            }
        }
        return values.sorted()
    }
}

private final class LocalAPISession: @unchecked Sendable {
    private let connection: NWConnection, queue: DispatchQueue, key: String
    private let handler: LocalModelAPIServer.Handler
    private let defaults: LocalGenerationOptions
    private let capabilities: LocalModelCapabilities
    private let onClose: () -> Void
    private var parser = LocalHTTPParser()
    private var finished = false
    private var timer: DispatchWorkItem?
    private var generation: LocalChatGeneration?
    private var task: Task<Void, Never>?
    init(connection: NWConnection, queue: DispatchQueue, key: String, defaults: LocalGenerationOptions, capabilities: LocalModelCapabilities,
         handler: @escaping LocalModelAPIServer.Handler, onClose: @escaping () -> Void) {
        self.connection = connection; self.queue = queue; self.key = key; self.defaults = defaults; self.handler = handler; self.onClose = onClose
        self.capabilities = capabilities
    }
    func start() {
        connection.stateUpdateHandler = { [weak self] state in
            guard let self else { return }
            switch state { case .failed, .cancelled: self.finish(); default: break }
        }
        connection.start(queue: queue)
        let timeout = DispatchWorkItem { [weak self] in self?.finish() }; timer = timeout
        queue.asyncAfter(deadline: .now() + 15, execute: timeout)
        receive()
    }
    private func receive() {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 16384) { [weak self] data, _, complete, error in
            guard let self, !self.finished else { return }
            do {
                if let request = try self.parser.append(data ?? Data()) {
                    self.timer?.cancel()
                    let timeout = DispatchWorkItem { [weak self] in self?.finish() }; self.timer = timeout
                    let seconds = min(10800, max(1800, self.capabilities.context.rawValue / 8192 * 1800))
                    self.queue.asyncAfter(deadline: .now() + .seconds(seconds), execute: timeout)
                    self.respond(request)
                    // A closed or pipelined client must not leave inference running.
                    self.connection.receive(minimumIncompleteLength: 1, maximumLength: 1) { [weak self] data, _, complete, error in
                        if complete || error != nil || data?.isEmpty == false { self?.finish() }
                    }
                } else if complete || error != nil { self.finish() }
                else { self.receive() }
            } catch { self.respondError(error as? LocalAPIError ?? .init(status: 400, message: "Invalid request")) }
        }
    }
    private func respond(_ request: LocalHTTPRequest) {
        guard LocalAPIWire.authenticated(request.headers["authorization"], key: key) else {
            respondError(.init(status: 401, message: "A valid local Bearer key is required")); return
        }
        guard request.headers["origin"] == nil else {
            respondError(.init(status: 403, message: "Browser-origin requests are not enabled")); return
        }
        if request.method == "GET", request.path == "/v1/models" || request.path == "/health" {
            let object: [String: Any] = request.path == "/health" ? ["status": "ok", "inference": "local", "model": LocalChatRequest.modelID] :
                ["object": "list", "data": [["id": LocalChatRequest.modelID, "object": "model", "created": 0, "owned_by": "local", "flash_local": capabilities.json]]]
            task = Task { [weak self] in
                guard let self else { return }
                do { try await self.sendJSON(object) } catch { }
                self.queue.async { self.finish(cancelInference: false) }
            }
            return
        }
        guard request.method == "POST", request.path == "/v1/chat/completions" else {
            respondError(.init(status: 404, message: "Use /v1/models or /v1/chat/completions")); return
        }
        guard request.headers["content-type"]?.lowercased().split(separator: ";").first == "application/json" else {
            respondError(.init(status: 415, message: "Content-Type must be application/json")); return
        }
        do {
            let prompt = try LocalChatRequest(data: request.body, defaults: defaults)
            task = Task { [weak self] in guard let self else { return }; await self.generate(prompt) }
        } catch { respondError(error as? LocalAPIError ?? .init(status: 400, message: "Invalid JSON request")) }
    }
    private func generate(_ request: LocalChatRequest) async {
        var headersSent = false
        do {
            try Task.checkCancellation()
            let source = try await handler(request)
            // Install cancellation on the network queue, including disconnects
            // occurring while the main actor was reserving the inference owner.
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                queue.async {
                    if self.finished { source.cancel(); continuation.resume(throwing: CancellationError()) }
                    else { self.generation = source; continuation.resume() }
                }
            }
            let id = "chatcmpl-" + UUID().uuidString.lowercased(), created = Int(Date().timeIntervalSince1970)
            if request.stream {
                try await send(LocalAPIWire.head(type: "text/event-stream")); headersSent = true
                try await send(LocalAPIWire.chunk(id: id, created: created, delta: ["role": "assistant", "content": ""]))
            }
            var answer = "", reasoning = "", usage: LocalCompletionUsage?
            for try await event in source.events {
                try Task.checkCancellation()
                switch event {
                case .delta(let content, let thought):
                    answer += content; reasoning += thought
                    guard answer.utf8.count + reasoning.utf8.count <= 4 << 20 else { throw LocalAPIError(status: 500, message: "Output limit exceeded") }
                    if request.stream {
                        var delta: [String: Any] = [:]
                        // Tool envelopes are withheld until the model finishes
                        // and every argument validates. Reasoning still streams.
                        if !content.isEmpty, !request.toolChoice.enabled { delta["content"] = content }
                        if !thought.isEmpty { delta["reasoning_content"] = thought }
                        if !delta.isEmpty { try await send(LocalAPIWire.chunk(id: id, created: created, delta: delta)) }
                    }
                case .complete(let result): usage = result
                }
            }
            guard let usage else { throw LocalAPIError(status: 500, message: "Generation ended without usage") }
            let result = try LocalToolOutput.parse(answer, tools: request.tools, choice: request.toolChoice,
                parallel: request.parallelToolCalls)
            guard result.calls.isEmpty || usage.finishReason == "stop" else {
                throw LocalAPIError(status: 502, message: "Function generation reached its limit before a normal completion", parameter: "tool_calls")
            }
            let finish = result.calls.isEmpty ? usage.finishReason : "tool_calls"
            await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                queue.async { self.generation = nil; continuation.resume() }
            }
            if request.stream {
                if request.toolChoice.enabled, !result.content.isEmpty {
                    try await send(LocalAPIWire.chunk(id: id, created: created, delta: ["content": result.content]))
                }
                if !result.calls.isEmpty {
                    let calls = result.calls.enumerated().map { index, call -> [String: Any] in
                        var value = call.json; value["index"] = index; return value
                    }
                    try await send(LocalAPIWire.chunk(id: id, created: created, delta: ["tool_calls": calls]))
                }
                try await send(LocalAPIWire.chunk(id: id, created: created, delta: [:], finish: finish))
                if request.includeUsage {
                    try await send(LocalAPIWire.sse(["id": id, "object": "chat.completion.chunk", "created": created,
                        "model": LocalChatRequest.modelID, "choices": [], "usage": usage.json]))
                }
                try await send(Data("data: [DONE]\n\n".utf8))
            } else {
                var message: [String: Any] = ["role": "assistant", "content": result.content.isEmpty && !result.calls.isEmpty ? NSNull() : result.content as Any]
                if !result.calls.isEmpty { message["tool_calls"] = result.calls.map(\.json) }
                if !reasoning.isEmpty { message["reasoning_content"] = reasoning }
                try await sendJSON(["id": id, "object": "chat.completion", "created": created, "model": LocalChatRequest.modelID,
                    "choices": [["index": 0, "message": message, "finish_reason": finish]], "usage": usage.json])
            }
            queue.async { self.generation = nil; self.finish(cancelInference: false) }
        } catch {
            let failure = error as? LocalAPIError ?? .init(status: 500, message: "Local generation stopped; see the phone status")
            if !headersSent { try? await sendJSONData(failure.encoded(), status: failure.status) }
            else { try? await send(LocalAPIWire.sse((try? JSONSerialization.jsonObject(with: failure.encoded()) as? [String: Any]) ?? [:])) }
            queue.async { self.finish() }
        }
    }
    private func respondError(_ error: LocalAPIError) {
        task = Task { [weak self] in
            guard let self else { return }; try? await sendJSONData(error.encoded(), status: error.status)
            queue.async { self.finish() }
        }
    }
    private func sendJSON(_ object: [String: Any]) async throws { try await sendJSONData(LocalAPIWire.json(object)) }
    private func sendJSONData(_ body: Data, status: Int = 200) async throws {
        var bytes = LocalAPIWire.head(status: status, length: body.count); bytes.append(body); try await send(bytes)
    }
    private func send(_ data: Data) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            queue.async {
                guard !self.finished else { continuation.resume(throwing: CancellationError()); return }
                self.connection.send(content: data, completion: .contentProcessed { error in
                    if let error { continuation.resume(throwing: error) } else { continuation.resume() }
                })
            }
        }
    }
    func finish(cancelInference: Bool = true) {
        guard !finished else { return }; finished = true
        timer?.cancel(); timer = nil
        if cancelInference { generation?.cancel() }
        generation = nil; task?.cancel(); task = nil
        connection.stateUpdateHandler = nil; connection.cancel(); onClose()
    }
}
