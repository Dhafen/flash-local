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
        let kind = status == 401 ? "authentication_error" : status >= 500 ? "server_error" : "invalid_request_error"
        let object: [String: Any] = ["error": ["message": message, "type": kind,
            "param": parameter as Any? ?? NSNull(), "code": status]]
        return (try? JSONSerialization.data(withJSONObject: object)) ?? Data()
    }
}

public struct LocalChatRequest: Sendable {
    public static let modelID = "ornith-1.5-35b-a3b"
    public struct Message: Sendable {
        public let role, content: String
        public let reasoning: String?
        public let toolCalls: [LocalToolCall]
        public let toolCallID: String?
    }
    public let messages: [Message]
    public let options: LocalGenerationOptions
    public let stream, includeUsage: Bool
    public let tools: [LocalToolDefinition]
    public let toolChoice: LocalToolChoice
    public let parallelToolCalls: Bool
    public let requestedModelID: String

    public init(data: Data, defaults: LocalGenerationOptions? = nil, capabilities: LocalModelCapabilities = .init()) throws {
        let defaults = defaults ?? capabilities.defaultOptions
        // Flint can legitimately receive a 65K-token prompt over the LAN.
        // Preserve the existing 256 KiB limit for Ornith; Flint gets a bounded
        // 2 MiB request matching its native template/tokenizer input limits.
        let maximumRequestBytes = capabilities.family == .flint ? 2 << 20 : 262144
        guard data.count <= maximumRequestBytes, let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            let limit = capabilities.family == .flint ? "2 MiB" : "256 KiB"
            throw LocalAPIError.invalid("body", "Expected a JSON object, at most \(limit)")
        }
        let allowed: Set<String> = ["model", "messages", "stream", "stream_options", "max_tokens", "max_completion_tokens",
            "temperature", "top_p", "seed", "enable_thinking", "thinking_budget", "chat_template_kwargs", "n",
            "frequency_penalty", "presence_penalty", "stop", "tools", "tool_choice", "parallel_tool_calls", "response_format", "reasoning_effort"]
        if let key = object.keys.sorted().first(where: { !allowed.contains($0) }) {
            throw LocalAPIError.invalid(key, "This parameter is not supported by the local text API")
        }
        guard object["model"] as? String == capabilities.modelID else {
            throw LocalAPIError(status: 404, message: "Unknown model; use GET /v1/models", parameter: "model")
        }
        requestedModelID = capabilities.modelID
        if capabilities.family == .flint {
            for key in ["reasoning_effort", "thinking_budget", "enable_thinking", "chat_template_kwargs"] {
                if let value = object[key], !(value is NSNull) {
                    throw LocalAPIError.invalid(key, "Flint exposes native reasoning output; forced reasoning budgets are not supported")
                }
            }
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
        for key in ["stop"] {
            if let value = object[key], !(value is NSNull), (value as? [Any])?.isEmpty != true {
                throw LocalAPIError.invalid(key, "Custom stop sequences are not implemented")
            }
        }
        var definitions: [LocalToolDefinition] = []
        if let value = object["tools"], !(value is NSNull) {
            guard let values = value as? [[String: Any]], values.count <= 32 else {
                throw LocalAPIError.invalid("tools", "Expected at most 32 function definitions")
            }
            definitions = try values.map(LocalToolDefinition.parse)
            guard Set(definitions.map(\.name)).count == definitions.count else {
                throw LocalAPIError.invalid("tools", "Function names must be unique")
            }
        }
        tools = definitions
        var choice: LocalToolChoice = definitions.isEmpty ? .none : .auto
        if let value = object["tool_choice"], !(value is NSNull) {
            if let text = value as? String {
                switch text {
                case "none": choice = .none
                case "auto": choice = definitions.isEmpty ? .none : .auto
                case "required": choice = .required
                default: throw LocalAPIError.invalid("tool_choice", "Expected none, auto, required, or a named function")
                }
            } else if let value = value as? [String: Any], Set(value.keys) == ["type", "function"],
                      value["type"] as? String == "function", let function = value["function"] as? [String: String],
                      Set(function.keys) == ["name"], let name = function["name"], definitions.contains(where: { $0.name == name }) {
                choice = .named(name)
            } else { throw LocalAPIError.invalid("tool_choice", "Named functions must be declared in tools") }
        }
        guard !choice.enabled || !definitions.isEmpty else { throw LocalAPIError.invalid("tool_choice", "Declare tools before requiring a call") }
        toolChoice = choice
        parallelToolCalls = try boolean("parallel_tool_calls", fallback: true)
        if let value = object["response_format"], !(value is NSNull) {
            guard let format = value as? [String: String], format == ["type": "text"] else {
                throw LocalAPIError.invalid("response_format", "Only text is supported")
            }
        }
        var options = defaults
        if let effort = object["reasoning_effort"], !(effort is NSNull) {
            let preset: OrnithThinkingPreset
            switch effort as? String {
            case "none": preset = .off
            case "minimal", "low": preset = .small
            case "medium": preset = .medium
            case "high", "xhigh", "max": preset = .ultra
            default: throw LocalAPIError.invalid("reasoning_effort", "Expected none, minimal, low, medium, high, xhigh, or max")
            }
            options.enableThinking = preset.options.enableThinking
            options.thinkingBudget = preset.options.thinkingBudget
            options.maximumNewTokens = preset.options.maximumNewTokens
        }
        let oldMaximum = try integer("max_tokens"), newMaximum = try integer("max_completion_tokens")
        if let a = oldMaximum, let b = newMaximum, a != b {
            throw LocalAPIError.invalid("max_tokens", "Conflicting token limits")
        }
        options.maximumNewTokens = newMaximum ?? oldMaximum ?? options.maximumNewTokens
        options.temperature = try numeric("temperature") ?? defaults.temperature
        options.topP = try numeric("top_p") ?? defaults.topP
        if let seed = try integer("seed") { options.seed = UInt64(seed) }
        if object.keys.contains("thinking_budget") {
            options.thinkingBudget = try integer("thinking_budget")
        }
        options.enableThinking = try boolean("enable_thinking", fallback: options.enableThinking)
        if let value = object["chat_template_kwargs"], !(value is NSNull) {
            guard let kwargs = value as? [String: Any], Set(kwargs.keys).isSubset(of: ["enable_thinking"]) else {
                throw LocalAPIError.invalid("chat_template_kwargs", "Only enable_thinking is supported")
            }
            options.enableThinking = try boolean("enable_thinking", from: kwargs, fallback: options.enableThinking)
        }
        if !object.keys.contains("thinking_budget"), oldMaximum != nil || newMaximum != nil,
           options.enableThinking, let inherited = options.thinkingBudget {
            // Apply only after both thinking overrides. An inactive saved cap
            // stays intact; a short thinking request fits its inherited cap.
            options.thinkingBudget = min(inherited, max(0, options.maximumNewTokens - 9))
        }
        try options.validate()
        self.options = options
        stream = try boolean("stream", fallback: false)
        if let value = object["stream_options"], !(value is NSNull) {
            guard let values = value as? [String: Any], Set(values.keys).isSubset(of: ["include_usage", "include_obfuscation"]) else {
                throw LocalAPIError.invalid("stream_options", "Unsupported stream options")
            }
            includeUsage = try boolean("include_usage", from: values, fallback: false)
        } else { includeUsage = false }
        guard let raw = object["messages"] as? [[String: Any]], !raw.isEmpty, raw.count <= 4096 else {
            throw LocalAPIError.invalid("messages", "Expected a nonempty text conversation")
        }
        var parsed: [Message] = [], expected = "user", size = 0
        var pending = Set<String>(), seen = Set<String>()
        for (index, value) in raw.enumerated() {
            guard let inputRole = value["role"] as? String else { throw LocalAPIError.invalid("messages", "A message role is required") }
            let role = inputRole == "developer" && index == 0 ? "system" : inputRole
            guard (index == 0 && role == "system") || role == expected else {
                throw LocalAPIError.invalid("messages", "Alternate conversation turns and resolve each assistant tool call before continuing")
            }
            var calls: [LocalToolCall] = []
            if let field = value["tool_calls"], !(field is NSNull) {
                guard role == "assistant", let values = field as? [[String: Any]], values.count <= 16 else {
                    throw LocalAPIError.invalid("messages", "Only assistant messages may contain bounded function calls")
                }
                calls = try values.map(LocalToolCall.parse)
                for call in calls {
                    guard seen.insert(call.id).inserted else { throw LocalAPIError.invalid("messages", "Tool call IDs must be unique") }
                }
            }
            let content: String
            if let text = value["content"] as? String { content = text }
            else if let parts = value["content"] as? [[String: Any]], parts.allSatisfy({ $0["type"] as? String == "text" && $0["text"] is String }) {
                content = parts.compactMap { $0["text"] as? String }.joined()
            } else if role == "assistant", !calls.isEmpty, value["content"] == nil || value["content"] is NSNull { content = "" }
            else { throw LocalAPIError.invalid("messages", "Only text content is supported") }
            let reasoning = value["reasoning_content"] as? String
            if let rawReasoning = value["reasoning_content"], !(rawReasoning is NSNull), !(rawReasoning is String) {
                throw LocalAPIError.invalid("messages", "reasoning_content must be text")
            }
            guard reasoning == nil || role == "assistant" else { throw LocalAPIError.invalid("messages", "Only assistants have reasoning_content") }
            if let field = value["function_call"], !(field is NSNull) {
                throw LocalAPIError.invalid("messages", "Use tool_calls instead of deprecated function_call")
            }
            let toolCallID = value["tool_call_id"] as? String
            if role == "tool" {
                guard let toolCallID, pending.remove(toolCallID) != nil else {
                    throw LocalAPIError.invalid("messages", "Tool results must match an outstanding call ID exactly once")
                }
            } else if value["tool_call_id"] != nil { throw LocalAPIError.invalid("messages", "Only tool results have tool_call_id") }
            size += content.utf8.count + (reasoning?.utf8.count ?? 0) + calls.reduce(0) { $0 + $1.function.arguments.utf8.count }
            guard size <= maximumRequestBytes else { throw LocalAPIError(status: 413, message: "Conversation is too large") }
            parsed.append(.init(role: role, content: content, reasoning: reasoning, toolCalls: calls, toolCallID: toolCallID))
            switch role {
            case "user": expected = "assistant"
            case "assistant":
                pending = Set(calls.map(\.id)); expected = pending.isEmpty ? "user" : "tool"
            case "tool": expected = pending.isEmpty ? "assistant" : "tool"
            default: break
            }
        }
        guard pending.isEmpty, parsed.last?.role == "user" || parsed.last?.role == "tool" else {
            throw LocalAPIError.invalid("messages", "Finish with a user turn or all matching tool results")
        }
        messages = parsed
    }

    public var toolInstructions: String {
        guard toolChoice.enabled else { return "" }
        var text = ""
        if toolChoice == .required { text = "For this response, call at least one of the declared functions." }
        if case .named(let name) = toolChoice { text = "For this response, call the declared function \(name)." }
        if !parallelToolCalls { text += " Emit at most one function call in this response." }
        return text.trimmingCharacters(in: .whitespaces)
    }
}

/// Measurements from the actual inference owner. Missing measurements remain
/// absent; HTTP timing or a synthetic provider never substitutes GPU evidence.
public struct LocalInferenceMetrics: Sendable {
    public let modelSHA256: String
    public let contextTokens, evaluatedPromptTokens, reusedPromptTokens, decodeForwardPasses: Int
    public let prefillSeconds, decodeSeconds, elapsedSeconds: Double
    public let prefillIOBytes, decodeIOBytes, peakMemoryBytes: UInt64
    public let thermalStateAtStart, maximumThermalState: Int
    public let platform, deviceIdentifier, appBuild, sourceRevision, nativeRevision: String?
    public let memoryBudgetBytes: UInt64?
    public let timeToFirstTokenSeconds: Double?
    public init(modelSHA256: String, contextTokens: Int, evaluatedPromptTokens: Int, reusedPromptTokens: Int,
                decodeForwardPasses: Int, prefillSeconds: Double, decodeSeconds: Double, elapsedSeconds: Double,
                prefillIOBytes: UInt64, decodeIOBytes: UInt64, peakMemoryBytes: UInt64,
                thermalStateAtStart: Int, maximumThermalState: Int, platform: String? = nil,
                deviceIdentifier: String? = nil, appBuild: String? = nil, sourceRevision: String? = nil,
                nativeRevision: String? = nil, memoryBudgetBytes: UInt64? = nil, timeToFirstTokenSeconds: Double? = nil) {
        self.modelSHA256 = modelSHA256; self.contextTokens = contextTokens
        self.evaluatedPromptTokens = evaluatedPromptTokens; self.reusedPromptTokens = reusedPromptTokens
        self.decodeForwardPasses = decodeForwardPasses; self.prefillSeconds = prefillSeconds
        self.decodeSeconds = decodeSeconds; self.elapsedSeconds = elapsedSeconds
        self.prefillIOBytes = prefillIOBytes; self.decodeIOBytes = decodeIOBytes; self.peakMemoryBytes = peakMemoryBytes
        self.thermalStateAtStart = thermalStateAtStart; self.maximumThermalState = maximumThermalState
        self.platform = platform; self.deviceIdentifier = deviceIdentifier; self.appBuild = appBuild
        self.sourceRevision = sourceRevision; self.nativeRevision = nativeRevision; self.memoryBudgetBytes = memoryBudgetBytes
        self.timeToFirstTokenSeconds = timeToFirstTokenSeconds
    }
    var json: [String: Any] {
        var value: [String: Any] = ["model_sha256": modelSHA256, "context_capacity_tokens": contextTokens, "full_model_numerics_verified": false,
         "evaluated_prompt_tokens": evaluatedPromptTokens, "reused_prompt_tokens": reusedPromptTokens,
         "decode_forward_passes": decodeForwardPasses, "prefill_seconds": prefillSeconds,
         "decode_seconds": decodeSeconds, "elapsed_seconds": elapsedSeconds,
         "prefill_io_bytes": prefillIOBytes, "decode_io_bytes": decodeIOBytes, "peak_memory_bytes": peakMemoryBytes,
         "thermal_state_at_start": thermalStateAtStart, "maximum_thermal_state": maximumThermalState]
        for (key, entry) in [("platform", platform), ("device_identifier", deviceIdentifier), ("app_build", appBuild),
                             ("source_revision", sourceRevision), ("native_revision", nativeRevision)] {
            if let entry { value[key] = entry }
        }
        if let memoryBudgetBytes { value["memory_budget_bytes"] = memoryBudgetBytes }
        if let timeToFirstTokenSeconds { value["time_to_first_token_seconds"] = timeToFirstTokenSeconds }
        return value
    }
}

public struct LocalCompletionUsage: Sendable {
    public let promptTokens, completionTokens, reasoningTokens, forcedTokens: Int
    public let finishReason: String
    public let metrics: LocalInferenceMetrics?
    public init(promptTokens: Int, completionTokens: Int, reasoningTokens: Int, forcedTokens: Int, finishReason: String,
                metrics: LocalInferenceMetrics? = nil) {
        self.promptTokens = promptTokens; self.completionTokens = completionTokens
        self.reasoningTokens = reasoningTokens; self.forcedTokens = forcedTokens; self.finishReason = finishReason; self.metrics = metrics
    }
    var json: [String: Any] {
        var value: [String: Any] = ["prompt_tokens": promptTokens, "completion_tokens": completionTokens,
        "total_tokens": promptTokens + completionTokens,
        "completion_tokens_details": ["reasoning_tokens": reasoningTokens], "forced_thinking_end_tokens": forcedTokens]
        if let metrics { value["flash_local"] = metrics.json }
        return value
    }
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
