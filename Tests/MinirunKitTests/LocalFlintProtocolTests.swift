import Foundation
import XCTest
@testable import MinirunKit

final class LocalFlintProtocolTests: XCTestCase {
    private let capabilities = LocalModelCapabilities(family: .flint)
    func testCompetingRuntimeFamiliesCannotLoadTogether() throws {
        var first: LocalEngineLease? = try .acquire(.ornith)
        var diagnostic: LocalEngineLease? = try .acquire(.ornith)
        XCTAssertNotNil(first); XCTAssertNotNil(diagnostic)
        XCTAssertThrowsError(try LocalEngineLease.acquire(.flint))
        first = nil
        XCTAssertThrowsError(try LocalEngineLease.acquire(.flint))
        diagnostic = nil
        var next: LocalEngineLease? = try .acquire(.flint)
        XCTAssertNotNil(next); XCTAssertThrowsError(try LocalEngineLease.acquire(.ornith))
        next = nil
        XCTAssertNoThrow(try LocalEngineLease.acquire(.ornith))
    }
    private func request(_ extra: [String: Any] = [:]) throws -> LocalChatRequest {
        var value: [String: Any] = ["model": capabilities.modelID, "messages": [["role": "user", "content": "Use the local fixture"]]]
        value.merge(extra) { _, new in new }
        return try LocalChatRequest(data: JSONSerialization.data(withJSONObject: value), capabilities: capabilities)
    }
    private func tool(_ properties: [String: Any], required: [String]? = nil) throws -> [LocalToolDefinition] {
        try request(["tools": [["type": "function", "function": ["name": "local_fixture", "parameters": [
            "type": "object", "properties": properties, "required": required ?? Array(properties.keys), "additionalProperties": false]]]]]).tools
    }
    func testFlintIdentityDefaultsAndUnsupportedReasoningControls() throws {
        let parsed = try request()
        XCTAssertEqual(parsed.requestedModelID, capabilities.modelID)
        XCTAssertEqual(parsed.options.temperature, 1); XCTAssertEqual(parsed.options.topP, 0.95)
        XCTAssertNil(parsed.options.thinkingBudget)
        XCTAssertEqual(capabilities.json["reasoning_presets"] as? [String], [])
        for extra in [["model": LocalChatRequest.modelID], ["reasoning_effort": "high"], ["enable_thinking": false],
                      ["thinking_budget": 256], ["chat_template_kwargs": ["enable_thinking": true]]] as [[String: Any]] {
            XCTAssertThrowsError(try request(extra))
        }
        let legacy = try LocalChatRequest(data: JSONSerialization.data(withJSONObject: ["model": LocalChatRequest.modelID,
            "messages": [["role": "user", "content": "fixture"]]]))
        XCTAssertEqual(legacy.options, LocalGenerationOptions())
    }
    func testActualFlintNamedArgumentsAndLocalToolResultRoundTrip() throws {
        let tools = try tool(["a": ["type": "integer"], "b": ["type": "integer"]])
        let result = try LocalFlintToolOutput.parse("<tool_call>local_fixture<arg_key>a</arg_key><arg_value>50</arg_value><arg_key>b</arg_key><arg_value>5</arg_value></tool_call>",
            tools: tools, choice: .required, parallel: false)
        let call = try XCTUnwrap(result.calls.first)
        let args = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(call.function.arguments.utf8)) as? [String: Int])
        // Owned, side-effect-free fixture tool executes on this test client.
        let observation = try XCTUnwrap(args["a"]) + XCTUnwrap(args["b"])
        XCTAssertEqual(observation, 55)
        let replay = try request(["tools": tools.map(\.json), "messages": [["role": "user", "content": "50+5"],
            ["role": "assistant", "content": NSNull(), "reasoning_content": "Use the local sum.", "tool_calls": [call.json]],
            ["role": "tool", "tool_call_id": call.id, "content": String(observation)]]])
        XCTAssertEqual(replay.messages.count, 3); XCTAssertEqual(replay.messages.last?.content, "55")
        XCTAssertEqual(replay.messages[1].reasoning, "Use the local sum.")
    }
    func testStringsKeepWhitespaceAndNumericText() throws {
        let tools = try tool(["source": ["type": "string"]])
        for value in ["00042", "null", "\n  def f():\n    return 55\n", "Hej 👋", "true", "{\"key\":1}"] {
            let result = try LocalFlintToolOutput.parse("<tool_call>local_fixture<arg_key>source</arg_key><arg_value>\(value)</arg_value></tool_call>", tools: tools, choice: .auto, parallel: false)
            let arguments = try JSONSerialization.jsonObject(with: Data(XCTUnwrap(result.calls.first).function.arguments.utf8)) as? [String: String]
            XCTAssertEqual(arguments?["source"], value)
        }
    }
    func testNullableAndStructuredValuesUseDeclaredJSONTypes() throws {
        let tools = try tool(["maybe": ["type": ["string", "null"]], "enabled": ["type": "boolean"],
            "items": ["type": "array", "items": ["type": "integer"]]])
        let result = try LocalFlintToolOutput.parse("<tool_call>local_fixture<arg_key>maybe</arg_key><arg_value>null</arg_value><arg_key>enabled</arg_key><arg_value>true</arg_value><arg_key>items</arg_key><arg_value>[1,2]</arg_value></tool_call>", tools: tools, choice: .required, parallel: false)
        let args = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(XCTUnwrap(result.calls.first).function.arguments.utf8)) as? [String: Any])
        XCTAssertTrue(args["maybe"] is NSNull); XCTAssertEqual(args["enabled"] as? Bool, true); XCTAssertEqual(args["items"] as? [Int], [1, 2])
    }
    func testMalformedUndeclaredDuplicateAndSchemaInvalidCallsFailClosed() throws {
        let tools = try tool(["n": ["type": "integer"]])
        let valid = "<tool_call>local_fixture<arg_key>n</arg_key><arg_value>1</arg_value></tool_call>"
        let cases = ["", "plain answer", "<tool_call>local_fixture", "</tool_call>",
            valid.replacingOccurrences(of: "local_fixture", with: "unknown"),
            valid.replacingOccurrences(of: ">1<", with: ">true<"),
            valid.replacingOccurrences(of: ">1<", with: ">True<"),
            valid.replacingOccurrences(of: "<arg_value>1</arg_value>", with: "<arg_value>1</arg_value><arg_key>n</arg_key><arg_value>2</arg_value>"),
            valid.replacingOccurrences(of: ">n<", with: ">extra<"), valid + " trailing", valid + "<tool_call>",
            valid.replacingOccurrences(of: "1</arg_value>", with: "<arg_key>x</arg_key></arg_value>"),
            String(repeating: valid, count: 17)]
        for value in cases {
            XCTAssertThrowsError(try LocalFlintToolOutput.parse(value, tools: tools, choice: .required, parallel: true)) { error in
                XCTAssertEqual((error as? LocalAPIError)?.status, 502)
            }
        }
        XCTAssertThrowsError(try LocalFlintToolOutput.parse(valid + valid, tools: tools, choice: .auto, parallel: false))
        XCTAssertThrowsError(try LocalFlintToolOutput.parse(valid, tools: tools, choice: .named("different"), parallel: true))
        XCTAssertEqual(try LocalFlintToolOutput.parse(valid, tools: tools, choice: .none, parallel: false).calls.count, 0)
    }
    func testLargeFlintLANRequestRemainsBoundedAndDoesNotRelaxOrnith() throws {
        // A long text prompt can be much bigger than 256 KiB even when
        // tokenization stays within the model's selected context window.
        let content = String(repeating: "a", count: 350_000)
        func body(model: String) throws -> Data {
            try JSONSerialization.data(withJSONObject: [
                "model": model, "messages": [["role": "user", "content": content]]
            ])
        }
        let flint = try body(model: capabilities.modelID)
        XCTAssertGreaterThan(flint.count, 262144)
        XCTAssertLessThan(flint.count, 2 << 20)
        let parsed = try LocalChatRequest(data: flint, capabilities: capabilities)
        XCTAssertEqual(parsed.messages.first?.content, content)

        // Verify the actual incremental HTTP parser, not just JSON decoding.
        let header = Data(("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: \(flint.count)\r\n\r\n").utf8)
        var request = header; request.append(flint)
        var flintParser = LocalHTTPParser(maximumBodyBytes: 2 << 20)
        let split = 16_384
        XCTAssertNil(try flintParser.append(Data(request.prefix(split))))
        let completed = try XCTUnwrap(flintParser.append(Data(request.dropFirst(split))))
        XCTAssertEqual(completed.body, flint)

        var ordinaryParser = LocalHTTPParser()
        XCTAssertThrowsError(try ordinaryParser.append(request)) {
            XCTAssertEqual(($0 as? LocalAPIError)?.status, 413)
        }
        let ordinaryBody = try body(model: LocalChatRequest.modelID)
        XCTAssertThrowsError(try LocalChatRequest(data: ordinaryBody, capabilities: .init())) {
            XCTAssertEqual(($0 as? LocalAPIError)?.status, 400)
        }
        var oversized = LocalHTTPParser(maximumBodyBytes: 2 << 20)
        let tooLongHeader = Data("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2097153\r\n\r\n".utf8)
        XCTAssertThrowsError(try oversized.append(tooLongHeader)) {
            XCTAssertEqual(($0 as? LocalAPIError)?.status, 413)
        }
    }
    func testSSEUsesSelectedModelAndKeepsLegacyDefault() throws {
        let selected = String(decoding: try LocalAPIWire.chunk(id: "fixture", created: 0, model: capabilities.modelID, delta: ["content": "55"]), as: UTF8.self)
        XCTAssertTrue(selected.contains(capabilities.modelID)); XCTAssertFalse(selected.contains(LocalChatRequest.modelID))
        let legacy = String(decoding: try LocalAPIWire.chunk(id: "fixture", created: 0, delta: [:]), as: UTF8.self)
        XCTAssertTrue(legacy.contains(LocalChatRequest.modelID))
    }
}
