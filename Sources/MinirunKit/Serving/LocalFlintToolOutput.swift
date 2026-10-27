import Foundation

/// Rec/Flint's named-argument protocol, distinct from Qwen's function tags.
/// Only complete, declared and schema-valid calls reach the OpenAI wire layer.
/// This parses model output; it never executes a tool on the phone.
public enum LocalFlintToolOutput {
    public static func parse(_ text: String, tools: [LocalToolDefinition], choice: LocalToolChoice,
                             parallel: Bool) throws -> (content: String, calls: [LocalToolCall]) {
        guard choice.enabled, !tools.isEmpty else { return (text, []) }
        guard text.utf8.count <= 4 << 20 else { throw malformed() }
        var remaining = text[...], content = "", calls: [LocalToolCall] = []
        while let open = remaining.range(of: "<tool_call>") {
            content += remaining[..<open.lowerBound]
            let tail = remaining[open.upperBound...]
            guard let close = tail.range(of: "</tool_call>") else { throw malformed() }
            let payload = String(tail[..<close.lowerBound])
            guard payload.utf8.count <= 65536 else { throw malformed() }
            let nameEnd = payload.range(of: "<arg_key>")?.lowerBound ?? payload.endIndex
            let name = String(payload[..<nameEnd]).trimmingCharacters(in: .whitespacesAndNewlines)
            guard LocalToolCall.validName(name), let tool = tools.first(where: { $0.name == name }) else { throw malformed() }
            if case .named(let expected) = choice, expected != name { throw malformed() }
            var body = payload[nameEnd...], arguments: [String: Any] = [:]
            while !body.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                body = body.drop(while: { $0.isWhitespace })
                guard body.hasPrefix("<arg_key>"), let keyEnd = body.range(of: "</arg_key>") else { throw malformed() }
                let key = String(body.dropFirst("<arg_key>".count).prefix(upTo: keyEnd.lowerBound))
                guard LocalToolCall.validName(key), arguments[key] == nil, arguments.count < 128 else { throw malformed() }
                let valueStart = body[keyEnd.upperBound...].drop(while: { $0.isWhitespace })
                guard valueStart.hasPrefix("<arg_value>"), let valueEnd = valueStart.range(of: "</arg_value>") else { throw malformed() }
                let raw = String(valueStart.dropFirst("<arg_value>".count).prefix(upTo: valueEnd.lowerBound))
                // This delimiter protocol cannot unambiguously encode its own
                // control tags inside values. Reject rather than invent data.
                guard !["<arg_", "</arg_", "<tool_call", "</tool_call"].contains(where: raw.contains) else { throw malformed() }
                do { arguments[key] = try tool.flintArgument(raw, name: key) } catch { throw malformed() }
                body = valueStart[valueEnd.upperBound...]
            }
            let encoded = try JSONSerialization.data(withJSONObject: arguments, options: [.sortedKeys, .fragmentsAllowed])
            let json = String(decoding: encoded, as: UTF8.self)
            do { try tool.validate(arguments: json) } catch { throw malformed() }
            calls.append(.init(id: "call_" + UUID().uuidString.replacingOccurrences(of: "-", with: ""),
                type: "function", function: .init(name: name, arguments: json)))
            guard calls.count <= 16, parallel || calls.count == 1 else { throw malformed() }
            remaining = tail[close.upperBound...]
        }
        if !calls.isEmpty, !remaining.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { throw malformed() }
        content += remaining
        guard !["<tool_call", "</tool_call", "<arg_key", "<arg_value", "</arg_key", "</arg_value"].contains(where: content.contains) else {
            throw malformed()
        }
        if calls.isEmpty, choice == .required { throw malformed() }
        if calls.isEmpty, case .named = choice { throw malformed() }
        // Preserve content exactly for correct publisher-template replay.
        return (calls.isEmpty ? text : content, calls)
    }
    private static func malformed() -> LocalAPIError {
        .init(status: 502, message: "Flint did not produce a complete, declared, schema-valid function call", parameter: "tool_calls")
    }
}
