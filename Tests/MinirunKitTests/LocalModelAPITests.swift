import Foundation
import XCTest
@testable import MinirunKit

final class LocalModelAPITests: XCTestCase {
    private let key = String(repeating: "test-key-", count: 8)
    private func body(_ overrides: [String: Any] = [:]) throws -> Data {
        var value: [String: Any] = ["model": LocalChatRequest.modelID, "messages": [["role": "user", "content": "Hej"]]]
        value.merge(overrides) { _, new in new }
        return try JSONSerialization.data(withJSONObject: value)
    }
    func testSDKTextRequestAndExplicitGenerationControls() throws {
        let value = try LocalChatRequest(data: body(["stream": true, "stream_options": ["include_usage": true],
            "max_completion_tokens": 256, "thinking_budget": 32, "temperature": 0.7, "top_p": 0.9, "seed": 7]))
        XCTAssertTrue(value.stream); XCTAssertTrue(value.includeUsage)
        XCTAssertEqual(value.options.maximumNewTokens, 256); XCTAssertEqual(value.options.thinkingBudget, 32)
        XCTAssertEqual(value.options.temperature, 0.7); XCTAssertEqual(value.options.seed, 7)
        XCTAssertEqual(value.messages.first?.content, "Hej")
        var defaults = LocalGenerationOptions(); defaults.maximumNewTokens = 1024; defaults.thinkingBudget = 128
        let inherited = try LocalChatRequest(data: body(), defaults: defaults)
        XCTAssertEqual(inherited.options, defaults)
        XCTAssertFalse(try LocalChatRequest(data: body(["thinking_budget": 0])).options.thinkingEnabled)
    }
    func testUnsupportedToolsAndInvalidParametersFailBeforeInference() throws {
        let invalid: [[String: Any]] = [["temperature": true], ["temperature": -1], ["top_p": 0], ["max_tokens": 0],
            ["max_tokens": 2049], ["max_tokens": 1.5], ["thinking_budget": 512], ["n": 2],
            ["max_tokens": 64, "max_completion_tokens": 65], ["stop": ["END"]], ["seed": -1],
            ["tools": [["type": "function"]]], ["response_format": ["type": "json_object"]],
            ["messages": [["role": "user", "content": "one"], ["role": "user", "content": "two"]]],
            ["messages": [["role": "user", "content": [["type": "image_url", "image_url": "x"]]]]],
            ["model": "other"], ["unknown_parameter": 1]]
        for request in invalid { XCTAssertThrowsError(try LocalChatRequest(data: body(request)), "\(request.keys.sorted())") }
    }
    func testTextPartsSystemAndAssistantReasoningAreRetained() throws {
        let request = try LocalChatRequest(data: body(["messages": [
            ["role": "system", "content": "local"], ["role": "user", "content": "question"],
            ["role": "assistant", "content": "answer", "reasoning_content": "reasoning"],
            ["role": "user", "content": [["type": "text", "text": "next"], ["type": "text", "text": " question"]]]]]))
        XCTAssertEqual(request.messages.count, 4); XCTAssertEqual(request.messages[2].reasoning, "reasoning")
        XCTAssertEqual(request.messages.last?.content, "next question")
    }
    func testFramingHandlesSplitUnicodeBodiesAndRejectsSmuggling() throws {
        let data = try body(["messages": [["role": "user", "content": "Hej 👋"]]])
        var parser = LocalHTTPParser()
        let header = Data("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: \(data.count)\r\n\r\n".utf8)
        XCTAssertNil(try parser.append(header + data.prefix(3)))
        XCTAssertEqual(try parser.append(Data(data.dropFirst(3)))?.body, data)
        for headers in ["Content-Length: 0\r\nContent-Length: 0", "Transfer-Encoding: chunked", "Content-Length: -1", "Content-Length: 999999"] {
            var parser = LocalHTTPParser()
            XCTAssertThrowsError(try parser.append(Data("POST / HTTP/1.1\r\nHost: local\r\n\(headers)\r\n\r\n".utf8)))
        }
        var extra = LocalHTTPParser()
        XCTAssertThrowsError(try extra.append(Data("GET / HTTP/1.1\r\nHost: local\r\n\r\nextra".utf8)))
        var large = LocalHTTPParser()
        XCTAssertThrowsError(try large.append(Data(repeating: 65, count: 16385)))
    }
    func testKeyAndNetworkBoundary() throws {
        XCTAssertTrue(LocalAPIWire.authenticated("Bearer " + key, key: key))
        XCTAssertFalse(LocalAPIWire.authenticated("Bearer " + String(key.dropLast()) + "X", key: key))
        XCTAssertFalse(LocalAPIWire.authenticated(nil, key: key))
        XCTAssertEqual(try LocalModelAPIServer.makeAccessKey().count, 64)
        XCTAssertNotEqual(try LocalModelAPIServer.makeAccessKey(), try LocalModelAPIServer.makeAccessKey())
        for ip in ["127.0.0.1", "10.0.0.8", "172.16.2.3", "192.168.1.10", "169.254.2.8"] {
            XCTAssertTrue(LocalModelAPIServer.isLocalIPv4(ip))
        }
        for ip in ["8.8.8.8", "0.0.0.0", "172.32.1.1", "192.168.1.999", "::1", "example.com"] {
            XCTAssertFalse(LocalModelAPIServer.isLocalIPv4(ip))
        }
        XCTAssertFalse(LocalModelAPIServer.isLocalIPv4("192.168.1.1", loopbackOnly: true))
    }
    func testErrorEnvelopeAndStreamingNullFieldsAreValidJSON() throws {
        let data = LocalAPIError(status: 401, message: "Use a local key").encoded()
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let error = try XCTUnwrap(object["error"] as? [String: Any])
        XCTAssertTrue(error["param"] is NSNull); XCTAssertEqual(error["type"] as? String, "authentication_error")
        let invalid = LocalAPIError.invalid("temperature", "Invalid value").encoded()
        let parsed = try XCTUnwrap(JSONSerialization.jsonObject(with: invalid) as? [String: [String: Any]])
        XCTAssertEqual(parsed["error"]?["param"] as? String, "temperature")
        let wire = try LocalAPIWire.chunk(id: "chatcmpl-test", created: 1, delta: ["content": "hej"])
        let json = String(decoding: wire, as: UTF8.self).dropFirst(6).trimmingCharacters(in: .whitespacesAndNewlines)
        let chunk = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
        XCTAssertTrue(chunk["usage"] is NSNull)
    }
    func testGreedySeededSamplingAndNucleusDistribution() throws {
        var greedy = LocalTokenSampler(seed: 1), options = LocalGenerationOptions()
        XCTAssertEqual(try greedy.select([-1, 4, 4, 2], options: options), 1)
        XCTAssertThrowsError(try greedy.select([.nan], options: options))
        options.temperature = 1; options.topP = 0.5
        for _ in 0..<100 { XCTAssertEqual(try greedy.select([0, 1, 4], options: options), 2) }
        options.topP = 1
        var a = LocalTokenSampler(seed: 42), b = LocalTokenSampler(seed: 42), counts = [0, 0, 0]
        let logits: [Float] = [Float(log(0.1)), Float(log(0.3)), Float(log(0.6))]
        for _ in 0..<10000 {
            let value = try a.select(logits, options: options)
            XCTAssertEqual(value, try b.select(logits, options: options)); counts[value] += 1
        }
        for (count, expected) in zip(counts, [0.1, 0.3, 0.6]) { XCTAssertEqual(Double(count) / 10000, expected, accuracy: 0.02) }
    }
    private func listen(handler: @escaping LocalModelAPIServer.Handler) async throws -> (LocalModelAPIServer, URL) {
        let ready = expectation(description: "real TCP listener ready"), port = APITestBox()
        let server = try LocalModelAPIServer(key: key, port: 0, loopbackOnly: true, handler: handler) { state in
            if case .ready(let value, _) = state { port.set(Int(value)); ready.fulfill() }
        }
        await fulfillment(of: [ready], timeout: 10)
        return (server, try XCTUnwrap(URL(string: "http://127.0.0.1:\(port.get())")))
    }
    private func request(_ url: URL, method: String = "GET", json: Data? = nil, authorized: Bool = true) -> URLRequest {
        var request = URLRequest(url: url); request.httpMethod = method; request.timeoutInterval = 10
        request.httpBody = json
        if authorized { request.setValue("Bearer " + key, forHTTPHeaderField: "Authorization") }
        if json != nil { request.setValue("application/json", forHTTPHeaderField: "Content-Type") }
        return request
    }
    func testRealTCPModelsAuthJSONAndSSE() async throws {
        let calls = APITestBox(), cancelled = APITestBox()
        let (server, base) = try await listen { request in
            calls.increment()
            let events = AsyncThrowingStream<LocalChatEvent, Error>.makeStream()
            events.continuation.yield(.delta(content: "", reasoning: "Räknar…"))
            events.continuation.yield(.delta(content: "Hej 👋\n", reasoning: ""))
            events.continuation.yield(.complete(.init(promptTokens: 14, completionTokens: 8, reasoningTokens: 3, forcedTokens: 0, finishReason: "stop")))
            events.continuation.finish()
            return .init(events: events.stream, cancel: { cancelled.increment() })
        }
        defer { server.stop() }
        let session = URLSession(configuration: .ephemeral); defer { session.invalidateAndCancel() }
        let (_, unauthenticated) = try await session.data(for: request(base.appendingPathComponent("v1/models"), authorized: false))
        XCTAssertEqual((unauthenticated as? HTTPURLResponse)?.statusCode, 401); XCTAssertEqual(calls.get(), 0)
        let (models, modelResponse) = try await session.data(for: request(base.appendingPathComponent("v1/models")))
        XCTAssertEqual((modelResponse as? HTTPURLResponse)?.statusCode, 200)
        XCTAssertTrue(String(decoding: models, as: UTF8.self).contains(LocalChatRequest.modelID)); XCTAssertEqual(calls.get(), 0)
        let endpoint = base.appendingPathComponent("v1/chat/completions")
        let (json, response) = try await session.data(for: request(endpoint, method: "POST", json: body()))
        XCTAssertEqual((response as? HTTPURLResponse)?.statusCode, 200)
        let parsed = try XCTUnwrap(JSONSerialization.jsonObject(with: json) as? [String: Any])
        let choice = try XCTUnwrap((parsed["choices"] as? [[String: Any]])?.first)
        let message = try XCTUnwrap(choice["message"] as? [String: Any])
        XCTAssertEqual(message["content"] as? String, "Hej 👋\n"); XCTAssertEqual(message["reasoning_content"] as? String, "Räknar…")
        let (stream, streamingResponse) = try await session.data(for: request(endpoint, method: "POST", json: body(["stream": true, "stream_options": ["include_usage": true]])))
        XCTAssertEqual((streamingResponse as? HTTPURLResponse)?.statusCode, 200)
        let text = String(decoding: stream, as: UTF8.self)
        XCTAssertTrue(text.hasSuffix("data: [DONE]\n\n")); XCTAssertTrue(text.contains("chat.completion.chunk"))
        XCTAssertTrue(text.contains("reasoning_content")); XCTAssertTrue(text.contains("completion_tokens"))
        let chunks = text.components(separatedBy: "\n\n").filter { $0.hasPrefix("data: {") }
        let ids = try chunks.map { try XCTUnwrap(JSONSerialization.jsonObject(with: Data($0.dropFirst(6).utf8)) as? [String: Any])["id"] as? String }
        XCTAssertEqual(Set(ids.compactMap { $0 }).count, 1)
        XCTAssertEqual(calls.get(), 2); XCTAssertEqual(cancelled.get(), 0)
        print("ORNITH_LOCAL_API realTCP=true auth=true json=true sse=true reasoningSeparated=true syntheticOnly=true phoneReachability=false")
    }
    func testBusyStatusAndDisconnectCancelTheGeneration() async throws {
        let (busy, base) = try await listen { _ in throw LocalAPIError(status: 429, message: "Busy") }
        let session = URLSession(configuration: .ephemeral); defer { session.invalidateAndCancel(); busy.stop() }
        let (_, response) = try await session.data(for: request(base.appendingPathComponent("v1/chat/completions"), method: "POST", json: body()))
        XCTAssertEqual((response as? HTTPURLResponse)?.statusCode, 429)
        XCTAssertEqual((response as? HTTPURLResponse)?.value(forHTTPHeaderField: "Retry-After"), "2")
        let began = expectation(description: "generation reserved"), cancelled = expectation(description: "disconnect reaches owner")
        let (server, url) = try await listen { _ in
            let events = AsyncThrowingStream<LocalChatEvent, Error>.makeStream()
            began.fulfill()
            return .init(events: events.stream, cancel: { events.continuation.finish(throwing: CancellationError()); cancelled.fulfill() })
        }
        defer { server.stop() }
        let task = session.dataTask(with: request(url.appendingPathComponent("v1/chat/completions"), method: "POST", json: try body(["stream": true]))) { _, _, _ in }
        task.resume(); await fulfillment(of: [began], timeout: 10); task.cancel()
        await fulfillment(of: [cancelled], timeout: 10)
    }
}

private final class APITestBox: @unchecked Sendable {
    private let lock = NSLock()
    private var value = 0
    func set(_ next: Int) { lock.lock(); value = next; lock.unlock() }
    func increment() { lock.lock(); value += 1; lock.unlock() }
    func get() -> Int { lock.lock(); defer { lock.unlock() }; return value }
}
