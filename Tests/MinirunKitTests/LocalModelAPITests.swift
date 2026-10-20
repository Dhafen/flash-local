import Foundation
import XCTest
@testable import MinirunKit

final class LocalModelAPITests: XCTestCase {
    private let key = String(repeating: "test-key-", count: 8)
    func testFixedReasoningPresets() throws {
        let expected: [(OrnithThinkingPreset, Int, Int, Bool)] = [
            (.off, 0, 2048, false),
            (.small, 256, 1536, true),
            (.medium, 512, 2048, true),
            (.ultra, 2048, 4096, true)
        ]
        for (preset, budget, maximum, enabled) in expected {
            let options = preset.options
            XCTAssertEqual(options.thinkingBudget, budget)
            XCTAssertEqual(options.maximumNewTokens, maximum)
            XCTAssertEqual(options.enableThinking, enabled)
            XCTAssertEqual(OrnithThinkingPreset.matching(options), preset)
            XCTAssertNoThrow(try options.validate())
        }
        XCTAssertEqual(LocalGenerationOptions().maximumNewTokens, 2048)
        XCTAssertEqual(LocalGenerationOptions().thinkingBudget, 512)
    }
    private func body(_ overrides: [String: Any] = [:]) throws -> Data {
        var value: [String: Any] = ["model": LocalChatRequest.modelID, "messages": [["role": "user", "content": "Hej"]]]
        value.merge(overrides) { _, new in new }
        return try JSONSerialization.data(withJSONObject: value)
    }
    private func sumTool() -> [String: Any] {
        ["type": "function", "function": ["name": "local_sum", "description": "Sum two supplied integers without network access",
            "strict": true, "parameters": ["type": "object", "properties": [
                "a": ["type": "integer", "minimum": -100, "maximum": 100], "b": ["type": "integer"]],
                "required": ["a", "b"], "additionalProperties": false]]]
    }
    func testFunctionDefinitionsChoicesAndSchemaAdmission() throws {
        let value = try LocalChatRequest(data: body(["tools": [sumTool()], "tool_choice": "required", "parallel_tool_calls": false]))
        XCTAssertEqual(value.toolChoice, .required); XCTAssertFalse(value.parallelToolCalls)
        XCTAssertEqual(value.tools.map(\.name), ["local_sum"])
        let named = try LocalChatRequest(data: body(["tools": [sumTool()],
            "tool_choice": ["type": "function", "function": ["name": "local_sum"]]]))
        XCTAssertEqual(named.toolChoice, .named("local_sum"))
        XCTAssertThrowsError(try LocalChatRequest(data: body(["tools": [sumTool(), sumTool()]])))
        XCTAssertThrowsError(try LocalChatRequest(data: body(["tool_choice": "required"])))
        XCTAssertThrowsError(try LocalChatRequest(data: body(["tools": [sumTool()],
            "tool_choice": ["type": "function", "function": ["name": "undeclared"]]])))
        let invalidSchemas: [[String: Any]] = [
            ["type": "object", "patternProperties": [:]],
            ["type": "object", "properties": ["x": ["$ref": "https://example.invalid/schema"]]],
            ["type": "object", "properties": ["x": ["type": "imaginary"]]]
        ]
        for schema in invalidSchemas {
            let tool: [String: Any] = ["type": "function", "function": ["name": "test", "parameters": schema]]
            XCTAssertThrowsError(try LocalChatRequest(data: body(["tools": [tool]])))
        }
    }
    func testToolHistoryRequiresEveryMatchingResultExactlyOnce() throws {
        let call: [String: Any] = ["id": "call_one", "type": "function",
            "function": ["name": "local_sum", "arguments": "{\"a\":50,\"b\":5}"]]
        let user: [String: Any] = ["role": "user", "content": "Use the sum tool"]
        let assistant: [String: Any] = ["role": "assistant", "content": NSNull(), "tool_calls": [call]]
        let result: [String: Any] = ["role": "tool", "tool_call_id": "call_one", "content": "55"]
        let valid = try LocalChatRequest(data: body(["tools": [sumTool()], "messages": [user, assistant, result]]))
        XCTAssertEqual(valid.messages[1].toolCalls.first?.function.arguments, "{\"a\":50,\"b\":5}")
        XCTAssertEqual(valid.messages.last?.toolCallID, "call_one")
        for messages in [[user, assistant], [user, result], [user, assistant, result, result],
            [user, assistant, ["role": "tool", "tool_call_id": "other", "content": "55"]],
            [user, ["role": "assistant", "content": "", "tool_calls": [call, call]], result]] {
            XCTAssertThrowsError(try LocalChatRequest(data: body(["tools": [sumTool()], "messages": messages])))
        }
    }
    func testSchemaValidatesNestedReferencesUnionsAndExactJSONTypes() throws {
        let tool: [String: Any] = ["type": "function", "function": ["name": "nested", "parameters": [
            "type": "object", "$defs": ["item": ["type": "integer", "minimum": 1]],
            "properties": ["items": ["type": "array", "items": ["$ref": "#/$defs/item"], "maxItems": 3],
                "flag": ["enum": [true]], "optional": ["anyOf": [["type": "null"], ["type": "string", "maxLength": 5]]]],
            "required": ["items", "flag"], "additionalProperties": false]]]
        let definition = try LocalChatRequest(data: body(["tools": [tool]])).tools[0]
        XCTAssertNoThrow(try definition.validate(arguments: "{\"items\":[1,2],\"flag\":true,\"optional\":null}"))
        for value in ["{\"items\":[0],\"flag\":true}", "{\"items\":[true],\"flag\":true}",
            "{\"items\":[1],\"flag\":1}", "{\"items\":[1],\"flag\":true,\"extra\":0}",
            "{\"items\":[1,2,3,4],\"flag\":true}", "{\"items\":[1],\"flag\":true,\"optional\":\"too long\"}"] {
            XCTAssertThrowsError(try definition.validate(arguments: value))
        }
    }
    func testNativeXMLCallsBecomeValidatedArgumentsWithoutChangingStringValues() throws {
        let tools = try LocalChatRequest(data: body(["tools": [sumTool()]])).tools
        let xml = "<tool_call>\n<function=local_sum>\n<parameter=a>\n50\n</parameter>\n<parameter=b>\n5\n</parameter>\n</function>\n</tool_call>"
        let output = try LocalToolOutput.parse(xml, tools: tools, choice: .required, parallel: false)
        XCTAssertTrue(output.content.isEmpty); XCTAssertEqual(output.calls.count, 1)
        XCTAssertEqual(output.calls[0].function.name, "local_sum")
        let arguments = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(output.calls[0].function.arguments.utf8)) as? [String: Int])
        XCTAssertEqual(arguments, ["a": 50, "b": 5])
        let stringTool: [String: Any] = ["type": "function", "function": ["name": "echo", "parameters": ["type": "object",
            "properties": ["text": ["type": "string"]], "required": ["text"], "additionalProperties": false]]]
        let echo = try LocalChatRequest(data: body(["tools": [stringTool]])).tools
        for text in ["55", "true", "  source code\n    indented\n"] {
            let xml = "<tool_call><function=echo><parameter=text>\n" + text + "\n</parameter></function></tool_call>"
            let result = try LocalToolOutput.parse(xml, tools: echo, choice: .auto, parallel: true)
            let arguments = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(result.calls[0].function.arguments.utf8)) as? [String: String])
            XCTAssertEqual(arguments["text"], text)
        }
    }
    func testMalformedUnknownAndTruncatedCallsNeverBecomeActions() throws {
        let tools = try LocalChatRequest(data: body(["tools": [sumTool()]])).tools
        for text in ["<tool_call>{\"name\":\"local_sum\",\"arguments\":{\"a\":50,\"b\":5}}",
            "<tool_call>{\"name\":\"undeclared\",\"arguments\":{}}</tool_call>",
            "<tool_call>{\"name\":\"local_sum\",\"arguments\":{\"a\":true,\"b\":5}}</tool_call>",
            "<tool_call>{\"name\":\"local_sum\",\"arguments\":{\"a\":500,\"b\":5}}</tool_call>",
            "<tool_call>{\"name\":\"local_sum\",\"arguments\":{\"a\":1}}</tool_call>",
            "<tool_call>{\"name\":\"local_sum\",\"arguments\":{\"a\":1,\"b\":2}}</tool_call> suffix",
            "<tool_call><function=local_sum><parameter=a>1</parameter><parameter=a>2</parameter></function></tool_call>"] {
            XCTAssertThrowsError(try LocalToolOutput.parse(text, tools: tools, choice: .auto, parallel: true))
        }
        XCTAssertThrowsError(try LocalToolOutput.parse("No call", tools: tools, choice: .required, parallel: true))
        let output = try LocalToolOutput.parse("  plain text\n", tools: tools, choice: .auto, parallel: false)
        XCTAssertEqual(output.content, "  plain text\n"); XCTAssertTrue(output.calls.isEmpty)
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
        let reset = try LocalChatRequest(data: body(["thinking_budget": NSNull(), "stream_options": NSNull()]), defaults: defaults)
        XCTAssertNil(reset.options.thinkingBudget); XCTAssertFalse(reset.includeUsage)
        XCTAssertFalse(try LocalChatRequest(data: body(["thinking_budget": 0])).options.thinkingEnabled)
        // An inactive saved cap must not block a short non-thinking request.
        let short = try LocalChatRequest(data: body(["max_tokens": 16, "enable_thinking": false]), defaults: defaults)
        XCTAssertFalse(short.options.thinkingEnabled); XCTAssertEqual(short.options.maximumNewTokens, 16)
        XCTAssertEqual(short.options.thinkingBudget, 128)
    }
    func testUnsupportedToolsAndInvalidParametersFailBeforeInference() throws {
        let invalid: [[String: Any]] = [["temperature": true], ["temperature": -1], ["top_p": 0], ["max_tokens": 0],
            ["max_tokens": 4097], ["max_tokens": 1.5], ["thinking_budget": 4097], ["n": 2],
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
            ["role": "assistant", "content": "answer", "reasoning_content": "reasoning", "tool_calls": NSNull(), "function_call": NSNull()],
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
        for invalid in ["", String(repeating: "a", count: 31), String(repeating: "a", count: 129),
                        String(repeating: "a", count: 32) + " ", String(repeating: "å", count: 32)] {
            XCTAssertThrowsError(try LocalAPIKeyStore.validate(invalid))
        }
        XCTAssertNoThrow(try LocalAPIKeyStore.validate(String(repeating: "A", count: 32)))
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
    func testKeychainCredentialSurvivesNewOwnerAndExplicitRotation() throws {
        let service = "com.dhafen.flashlocal.tests." + UUID().uuidString
        let store = LocalAPIKeyStore(service: service)
        defer { try? store.delete() }
        XCTAssertNil(try store.load())
        let first = try store.loadOrCreate()
        XCTAssertEqual(try LocalAPIKeyStore(service: service).loadOrCreate(), first)
        let replacement = try LocalModelAPIServer.makeAccessKey()
        try store.save(replacement)
        XCTAssertEqual(try store.load(), replacement); XCTAssertNotEqual(first, replacement)
        XCTAssertThrowsError(try store.save("too short"))
        XCTAssertEqual(try store.load(), replacement)
        try store.delete(); XCTAssertNil(try store.load())
        print("LOCAL_API_CREDENTIAL realKeychain=true newOwnerRetains=true explicitRotation=true invalidWriteAtomic=true noCredentialLogged=true")
    }
    func testInferenceMeasurementsAreOptionalAndNeverInventedByTheWireLayer() throws {
        let usage = LocalCompletionUsage(promptTokens: 20, completionTokens: 128, reasoningTokens: 0, forcedTokens: 0, finishReason: "length")
        XCTAssertNil(usage.json["flash_local"])
        let metrics = LocalInferenceMetrics(modelSHA256: String(repeating: "a", count: 64), contextTokens: 8192,
            evaluatedPromptTokens: 18, reusedPromptTokens: 2, decodeForwardPasses: 128,
            prefillSeconds: 2, decodeSeconds: 32, elapsedSeconds: 34, prefillIOBytes: 1024,
            decodeIOBytes: 4096, peakMemoryBytes: 2_000_000_000, thermalStateAtStart: 0, maximumThermalState: 1)
        let measured = LocalCompletionUsage(promptTokens: 20, completionTokens: 128, reasoningTokens: 0, forcedTokens: 0,
            finishReason: "length", metrics: metrics)
        let data = try JSONSerialization.data(withJSONObject: measured.json)
        let decoded = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let values = try XCTUnwrap(decoded["flash_local"] as? [String: Any])
        XCTAssertEqual(values["decode_forward_passes"] as? Int, 128)
        XCTAssertEqual(values["decode_seconds"] as? Double, 32)
        XCTAssertEqual(values["reused_prompt_tokens"] as? Int, 2)
        XCTAssertNil(values["gpu_seconds"])
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
    func testRealTCPToolCallResultAndStreamingRoundTrip() async throws {
        let calls = APITestBox()
        let (server, base) = try await listen { request in
            calls.increment()
            let events = AsyncThrowingStream<LocalChatEvent, Error>.makeStream()
            if request.messages.last?.role == "tool" {
                guard request.messages.last?.content == "55", request.messages[1].toolCalls.count == 1 else {
                    throw LocalAPIError(status: 500, message: "Synthetic tool result was not preserved")
                }
                events.continuation.yield(.delta(content: "The tool returned 55.", reasoning: ""))
            } else {
                events.continuation.yield(.delta(content: "", reasoning: "Synthetic provider chooses the supplied sum tool."))
                for part in ["<tool_", "call>\n<function=local_sum>\n<parameter=a>\n50\n</parameter>\n",
                    "<parameter=b>\n5\n</parameter>\n</function>\n</tool_call>"] {
                    events.continuation.yield(.delta(content: part, reasoning: ""))
                }
            }
            events.continuation.yield(.complete(.init(promptTokens: 20, completionTokens: 40, reasoningTokens: 4, forcedTokens: 0, finishReason: "stop")))
            events.continuation.finish()
            return .init(events: events.stream, cancel: {})
        }
        defer { server.stop() }
        let session = URLSession(configuration: .ephemeral); defer { session.invalidateAndCancel() }
        let endpoint = base.appendingPathComponent("v1/chat/completions")
        let user: [String: Any] = ["role": "user", "content": "Use local_sum for 50 plus 5."]
        let firstBody = try body(["tools": [sumTool()], "tool_choice": "required", "parallel_tool_calls": false, "messages": [user]])
        let (first, response) = try await session.data(for: request(endpoint, method: "POST", json: firstBody))
        XCTAssertEqual((response as? HTTPURLResponse)?.statusCode, 200)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: first) as? [String: Any])
        let choice = try XCTUnwrap((object["choices"] as? [[String: Any]])?.first)
        XCTAssertEqual(choice["finish_reason"] as? String, "tool_calls")
        let message = try XCTUnwrap(choice["message"] as? [String: Any])
        XCTAssertTrue(message["content"] is NSNull)
        let functionCall = try XCTUnwrap((message["tool_calls"] as? [[String: Any]])?.first)
        let function = try XCTUnwrap(functionCall["function"] as? [String: String])
        XCTAssertEqual(function["name"], "local_sum")
        let args = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(try XCTUnwrap(function["arguments"]).utf8)) as? [String: Int])
        // This is a real HTTP/client tool round trip with a declared synthetic
        // generation provider. It is not a model-capability or phone result.
        let result = try XCTUnwrap(args["a"]) + XCTUnwrap(args["b"])
        let toolResult: [String: Any] = ["role": "tool", "tool_call_id": try XCTUnwrap(functionCall["id"] as? String), "content": String(result)]
        let second = try body(["tools": [sumTool()], "messages": [user, message, toolResult]])
        let (followup, followupResponse) = try await session.data(for: request(endpoint, method: "POST", json: second))
        XCTAssertEqual((followupResponse as? HTTPURLResponse)?.statusCode, 200)
        XCTAssertTrue(String(decoding: followup, as: UTF8.self).contains("The tool returned 55."))
        let (stream, streamResponse) = try await session.data(for: request(endpoint, method: "POST",
            json: body(["tools": [sumTool()], "stream": true, "stream_options": ["include_usage": true]])))
        XCTAssertEqual((streamResponse as? HTTPURLResponse)?.statusCode, 200)
        let text = String(decoding: stream, as: UTF8.self)
        XCTAssertTrue(text.hasSuffix("data: [DONE]\n\n")); XCTAssertFalse(text.contains("<tool_call>"))
        let chunks = try text.components(separatedBy: "\n\n").filter { $0.hasPrefix("data: {") }.map {
            try XCTUnwrap(JSONSerialization.jsonObject(with: Data($0.dropFirst(6).utf8)) as? [String: Any])
        }
        let choices = chunks.compactMap { ($0["choices"] as? [[String: Any]])?.first }
        let streamed = choices.compactMap { ($0["delta"] as? [String: Any])?["tool_calls"] as? [[String: Any]] }.flatMap { $0 }
        XCTAssertEqual(streamed.count, 1); XCTAssertEqual(streamed.first?["index"] as? Int, 0)
        XCTAssertTrue(choices.contains { $0["finish_reason"] as? String == "tool_calls" })
        XCTAssertEqual(calls.get(), 3)
        print("LOCAL_API_TOOL_ROUNDTRIP realTCP=true structuredArguments=true matchedToolResult=true json=true sse=true syntheticProvider=true trainedInference=false phoneReachability=false")
    }
    func testRealTCPInvalidToolHistoryAndModelArgumentsCannotBecomeActions() async throws {
        let calls = APITestBox()
        let (server, base) = try await listen { _ in
            calls.increment()
            let events = AsyncThrowingStream<LocalChatEvent, Error>.makeStream()
            events.continuation.yield(.delta(content: "<tool_call>{\"name\":\"local_sum\",\"arguments\":{\"a\":true,\"b\":5}}</tool_call>", reasoning: ""))
            events.continuation.yield(.complete(.init(promptTokens: 1, completionTokens: 1, reasoningTokens: 0, forcedTokens: 0, finishReason: "stop")))
            events.continuation.finish()
            return .init(events: events.stream, cancel: {})
        }
        defer { server.stop() }
        let session = URLSession(configuration: .ephemeral); defer { session.invalidateAndCancel() }
        let endpoint = base.appendingPathComponent("v1/chat/completions")
        let invalid = try body(["tools": [sumTool()], "messages": [["role": "tool", "tool_call_id": "orphan", "content": "55"]]])
        let (_, invalidResponse) = try await session.data(for: request(endpoint, method: "POST", json: invalid))
        XCTAssertEqual((invalidResponse as? HTTPURLResponse)?.statusCode, 400); XCTAssertEqual(calls.get(), 0)
        let (bad, badResponse) = try await session.data(for: request(endpoint, method: "POST", json: body(["tools": [sumTool()]])))
        XCTAssertEqual((badResponse as? HTTPURLResponse)?.statusCode, 502)
        let parsed = try XCTUnwrap(JSONSerialization.jsonObject(with: bad) as? [String: Any])
        XCTAssertEqual((parsed["error"] as? [String: Any])?["type"] as? String, "server_error")
        XCTAssertNil(parsed["choices"])
        let (stream, _) = try await session.data(for: request(endpoint, method: "POST", json: body(["tools": [sumTool()], "stream": true])))
        let text = String(decoding: stream, as: UTF8.self)
        XCTAssertTrue(text.contains("schema-valid")); XCTAssertFalse(text.contains("\"tool_calls\":["))
        XCTAssertFalse(text.contains("data: [DONE]")); XCTAssertEqual(calls.get(), 2)
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
    func testAToolEnvelopeAtTheTokenLimitCannotBeReportedAsACompletedAction() async throws {
        let (server, base) = try await listen { _ in
            let events = AsyncThrowingStream<LocalChatEvent, Error>.makeStream()
            events.continuation.yield(.delta(content: "<tool_call>{\"name\":\"local_sum\",\"arguments\":{\"a\":50,\"b\":5}}</tool_call>", reasoning: ""))
            events.continuation.yield(.complete(.init(promptTokens: 10, completionTokens: 10, reasoningTokens: 0, forcedTokens: 0, finishReason: "length")))
            events.continuation.finish()
            return .init(events: events.stream, cancel: {})
        }
        defer { server.stop() }
        let session = URLSession(configuration: .ephemeral); defer { session.invalidateAndCancel() }
        let (data, response) = try await session.data(for: request(base.appendingPathComponent("v1/chat/completions"), method: "POST", json: body(["tools": [sumTool()]])))
        XCTAssertEqual((response as? HTTPURLResponse)?.statusCode, 502)
        let decoded = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertNil(decoded["choices"]); XCTAssertNotNil(decoded["error"])
    }
}

private final class APITestBox: @unchecked Sendable {
    private let lock = NSLock()
    private var value = 0
    func set(_ next: Int) { lock.lock(); value = next; lock.unlock() }
    func increment() { lock.lock(); value += 1; lock.unlock() }
    func get() -> Int { lock.lock(); defer { lock.unlock() }; return value }
}
