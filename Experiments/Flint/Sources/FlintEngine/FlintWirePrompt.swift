import Foundation
import MinirunKit

/// Structured publisher-template input; no hand-written role-token prompt.
public enum FlintWirePrompt {
    public static func encode(_ request: LocalChatRequest) throws -> Data {
        guard request.requestedModelID == LocalModelCapabilities(family: .flint).modelID else {
            throw LocalAPIError(status: 404, message: "This executor only serves the admitted Flint model", parameter: "model")
        }
        var messages: [[String: Any]] = request.messages.map { message in
            var value: [String: Any] = ["role": message.role, "content": message.content]
            if let reasoning = message.reasoning { value["reasoning_content"] = reasoning }
            if !message.toolCalls.isEmpty { value["tool_calls"] = message.toolCalls.map(\.json) }
            if let id = message.toolCallID { value["tool_call_id"] = id }
            return value
        }
        // Tool-choice constraints are instructions, not fabricated tool calls.
        // The completed output must still pass the declared schema and choice.
        if !request.toolInstructions.isEmpty {
            let position = messages.first?["role"] as? String == "system" ? 1 : 0
            messages.insert(["role": "developer", "content": request.toolInstructions], at: position)
        }
        let value: [String: Any] = ["messages": messages,
            "tools": request.toolChoice.enabled ? request.tools.map(\.json) : [], "add_generation_prompt": true]
        return try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
    }
}
