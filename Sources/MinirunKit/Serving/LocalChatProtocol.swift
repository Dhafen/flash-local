import CoreFoundation
import Foundation

public struct LocalAPIError: Error, LocalizedError, Sendable {
    public let status: Int
    public let message: String
    public let parameter: String?
    public init(status: Int, message: String, parameter: String? = nil) {
        self.status = status; self.message = message; self.parameter = parameter
    }
    public static func invalid(_ parameter: String, _ message: String) -> Self {
        .init(status: 400, message: message, parameter: parameter)
    }
    public var errorDescription: String? { message }
    public func encoded() -> Data {
        let object: [String: Any] = ["error": ["message": message, "type": status == 401 ? "authentication_error" : "invalid_request_error",
            "param": parameter as Any? ?? NSNull(), "code": status]]
        return (try? JSONSerialization.data(withJSONObject: object)) ?? Data()
    }
}

public struct LocalChatRequest: Sendable {
    public static let modelID = "ornith-1.5-35b-a3b"
    public struct Message: Sendable {
        public let role, content: String
        public let reasoning: String?
    }
    public let messages: [Message]
    public let options: LocalGenerationOptions
    public let stream, includeUsage: Bool

    public init(data: Data, defaults: LocalGenerationOptions = .init()) throws {
        guard data.count <= 262144, let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw LocalAPIError.invalid("body", "Expected a JSON object, at most 256 KiB")
        }
        let allowed: Set<String> = ["model", "messages", "stream", "stream_options", "max_tokens", "max_completion_tokens",
            "temperature", "top_p", "seed", "enable_thinking", "thinking_budget", "chat_template_kwargs", "n",
            "frequency_penalty", "presence_penalty", "stop", "tools", "tool_choice", "response_format"]
        if let key = object.keys.sorted().first(where: { !allowed.contains($0) }) {
            throw LocalAPIError.invalid(key, "This parameter is not supported by the local text API")
        }
        guard object["model"] as? String == Self.modelID else {
            throw LocalAPIError(status: 404, message: "Unknown model; use GET /v1/models", parameter: "model")
        }
        func numeric(_ key: String, from source: [String: Any]? = nil) throws -> Double? {
            guard let value = (source ?? object)[key], !(value is NSNull) else { return nil }
            guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(), number.doubleValue.isFinite else {
                throw LocalAPIError.invalid(key, "Expected a finite number")
            }
            return number.doubleValue
        }
        func integer(_ key: String) throws -> Int? {
            guard let value = try numeric(key) else { return nil }
            guard value.rounded(.towardZero) == value, value >= 0, value <= Double(Int32.max) else {
                throw LocalAPIError.invalid(key, "Expected a bounded nonnegative integer")
            }
            return Int(value)
        }
        func boolean(_ key: String, from source: [String: Any]? = nil, fallback: Bool) throws -> Bool {
            guard let value = (source ?? object)[key], !(value is NSNull) else { return fallback }
            guard let number = value as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() else {
                throw LocalAPIError.invalid(key, "Expected a boolean")
            }
            return number.boolValue
        }
        for key in ["frequency_penalty", "presence_penalty"] {
            if let value = try numeric(key), value != 0 { throw LocalAPIError.invalid(key, "Only zero is currently supported") }
        }
        if let count = try integer("n"), count != 1 { throw LocalAPIError.invalid("n", "One completion at a time") }
        for key in ["stop", "tools"] {
            if let value = object[key], !(value is NSNull), (value as? [Any])?.isEmpty != true {
                throw LocalAPIError.invalid(key, "Custom stop sequences and structured tools are not implemented")
            }
        }
        if let value = object["tool_choice"], !(value is NSNull), value as? String != "none" {
            throw LocalAPIError.invalid("tool_choice", "Structured tools are not implemented")
        }
        if let value = object["response_format"], !(value is NSNull) {
            guard let format = value as? [String: String], format == ["type": "text"] else {
                throw LocalAPIError.invalid("response_format", "Only text is supported")
            }
        }
        var options = defaults
        let oldMaximum = try integer("max_tokens"), newMaximum = try integer("max_completion_tokens")
        if let a = oldMaximum, let b = newMaximum, a != b {
            throw LocalAPIError.invalid("max_tokens", "Conflicting token limits")
        }
        options.maximumNewTokens = newMaximum ?? oldMaximum ?? defaults.maximumNewTokens
        options.temperature = try numeric("temperature") ?? defaults.temperature
        options.topP = try numeric("top_p") ?? defaults.topP
        if let seed = try integer("seed") { options.seed = UInt64(seed) }
        options.thinkingBudget = try integer("thinking_budget") ?? defaults.thinkingBudget
        options.enableThinking = try boolean("enable_thinking", fallback: defaults.enableThinking)
        if let value = object["chat_template_kwargs"] {
            guard let kwargs = value as? [String: Any], Set(kwargs.keys).isSubset(of: ["enable_thinking"]) else {
                throw LocalAPIError.invalid("chat_template_kwargs", "Only enable_thinking is supported")
            }
            options.enableThinking = try boolean("enable_thinking", from: kwargs, fallback: options.enableThinking)
        }
        try options.validate()
        self.options = options
        stream = try boolean("stream", fallback: false)
        if let value = object["stream_options"] {
            guard let values = value as? [String: Any], Set(values.keys).isSubset(of: ["include_usage", "include_obfuscation"]) else {
                throw LocalAPIError.invalid("stream_options", "Unsupported stream options")
            }
            includeUsage = try boolean("include_usage", from: values, fallback: false)
        } else { includeUsage = false }
        guard let raw = object["messages"] as? [[String: Any]], !raw.isEmpty, raw.count <= 4096 else {
            throw LocalAPIError.invalid("messages", "Expected a nonempty text conversation")
        }
        var parsed: [Message] = [], expected = "user", size = 0
        for (index, value) in raw.enumerated() {
            guard let role = value["role"] as? String,
                  (index == 0 && role == "system") || role == expected else {
                throw LocalAPIError.invalid("messages", "Alternate user and assistant after an optional system message")
            }
            let content: String
            if let text = value["content"] as? String { content = text }
            else if let parts = value["content"] as? [[String: Any]], parts.allSatisfy({ $0["type"] as? String == "text" && $0["text"] is String }) {
                content = parts.compactMap { $0["text"] as? String }.joined()
            } else { throw LocalAPIError.invalid("messages", "Only text content is supported") }
            let reasoning = value["reasoning_content"] as? String
            guard reasoning == nil || role == "assistant" else { throw LocalAPIError.invalid("messages", "Only assistants have reasoning_content") }
            if value["tool_calls"] != nil || value["function_call"] != nil {
                throw LocalAPIError.invalid("messages", "Tool-call history is not implemented")
            }
            size += content.utf8.count + (reasoning?.utf8.count ?? 0)
            guard size <= 262144 else { throw LocalAPIError(status: 413, message: "Conversation is too large") }
            parsed.append(.init(role: role, content: content, reasoning: reasoning))
            if role != "system" { expected = role == "user" ? "assistant" : "user" }
        }
        guard parsed.last?.role == "user" else { throw LocalAPIError.invalid("messages", "Finish the prompt with a user message") }
        messages = parsed
    }
}

public struct LocalCompletionUsage: Sendable {
    public let promptTokens, completionTokens, reasoningTokens, forcedTokens: Int
    public let finishReason: String
    public init(promptTokens: Int, completionTokens: Int, reasoningTokens: Int, forcedTokens: Int, finishReason: String) {
        self.promptTokens = promptTokens; self.completionTokens = completionTokens
        self.reasoningTokens = reasoningTokens; self.forcedTokens = forcedTokens; self.finishReason = finishReason
    }
    var json: [String: Any] { ["prompt_tokens": promptTokens, "completion_tokens": completionTokens,
        "total_tokens": promptTokens + completionTokens,
        "completion_tokens_details": ["reasoning_tokens": reasoningTokens], "forced_thinking_end_tokens": forcedTokens] }
}
public enum LocalChatEvent: Sendable {
    case delta(content: String, reasoning: String)
    case complete(LocalCompletionUsage)
}
public struct LocalChatGeneration: Sendable {
    public let events: AsyncThrowingStream<LocalChatEvent, Error>
    public let cancel: @Sendable () -> Void
    public init(events: AsyncThrowingStream<LocalChatEvent, Error>, cancel: @escaping @Sendable () -> Void) {
        self.events = events; self.cancel = cancel
    }
}
