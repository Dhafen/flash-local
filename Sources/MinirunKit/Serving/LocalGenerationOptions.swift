import Foundation

/// Total session window, including template, tools, prompt and generated tokens.
/// Selecting a larger window requests a memory admission, not a quality claim.
public enum LocalContextProfile: Int, Codable, CaseIterable, Identifiable, Sendable {
    case standard = 8192, test16K = 16384, test32K = 32768, test65K = 65536
    public var id: Int { rawValue }
    public var title: String {
        self == .standard ? "8K" : "\(rawValue / 1024)K · test"
    }
}

public struct LocalModelCapabilities: Sendable {
    public enum Family: Sendable { case ornith, flint }
    public let context: LocalContextProfile
    public let family: Family
    public init(context: LocalContextProfile = .standard, family: Family = .ornith) {
        self.context = context; self.family = family
    }
    public var modelID: String { family == .ornith ? LocalChatRequest.modelID : "recursive-flint-40b-a2.5b-int4" }
    public var defaultOptions: LocalGenerationOptions {
        var value = LocalGenerationOptions()
        if family == .flint {
            value.temperature = 1; value.topP = 0.95
            // Rec publishes reasoning output, not Ornith's forced-budget or
            // enable_thinking protocol. Do not advertise an invented control.
            value.enableThinking = true; value.thinkingBudget = nil
        }
        return value
    }
    var json: [String: Any] {
        ["context_capacity_tokens": context.rawValue, "max_output_tokens": 4096,
         "reasoning_presets": family == .ornith ? ["off", "small", "medium", "ultra"] : [],
         "reasoning_output": true, "thinking_budget_control": family == .ornith,
         "tool_calls": true, "tool_quality_verified": false, "context_quality_verified": false]
    }
    func parseTools(_ text: String, request: LocalChatRequest) throws -> (content: String, calls: [LocalToolCall]) {
        if family == .flint {
            return try LocalFlintToolOutput.parse(text, tools: request.tools, choice: request.toolChoice, parallel: request.parallelToolCalls)
        }
        return try LocalToolOutput.parse(text, tools: request.tools, choice: request.toolChoice, parallel: request.parallelToolCalls)
    }
}

/// Generation controls shared by the local UI and the authenticated LAN API.
/// Routing, weights and native graph mathematics are unaffected.
public struct LocalGenerationOptions: Codable, Equatable, Sendable {
    public var maximumNewTokens: Int = 2048
    public var enableThinking: Bool = true
    public var thinkingBudget: Int? = 512
    public var temperature: Double = 0
    public var topP: Double = 1
    public var seed: UInt64 = 42
    public init() {}
    public var thinkingEnabled: Bool { enableThinking && thinkingBudget != 0 }
    public func validate() throws {
        guard (1...4096).contains(maximumNewTokens) else {
            throw LocalAPIError.invalid("max_tokens", "Välj 1–4096 token totalt för tänkande och svar.")
        }
        if let budget = thinkingBudget {
            guard (0...4096).contains(budget), budget == 0 || !enableThinking || budget + 8 < maximumNewTokens else {
                throw LocalAPIError.invalid("thinking_budget", "Tänkbudgeten måste lämna minst 9 token till avslut och svar.")
            }
        }
        guard temperature.isFinite, (0...2).contains(temperature), topP.isFinite, topP > 0, topP <= 1 else {
            throw LocalAPIError.invalid("temperature", "Temperatur måste vara 0–2 och top_p större än 0 och högst 1.")
        }
        guard seed <= UInt64(Int32.max) else { throw LocalAPIError.invalid("seed", "Slumpfrö måste vara 0–2147483647.") }
    }
}


/// Fixed, bounded reasoning presets. The output cap includes thinking and answer;
/// context remains 8192 tokens and the prompt budget is context minus output.
public enum OrnithThinkingPreset: String, CaseIterable, Identifiable, Sendable {
    case off, small, medium, ultra
    public var id: String { rawValue }
    public var title: String {
        switch self {
        case .off: "Off"
        case .small: "Small"
        case .medium: "Medium"
        case .ultra: "Ultra"
        }
    }
    public var options: LocalGenerationOptions {
        var value = LocalGenerationOptions()
        switch self {
        case .off:
            value.enableThinking = false; value.thinkingBudget = 0; value.maximumNewTokens = 2048
        case .small:
            value.enableThinking = true; value.thinkingBudget = 256; value.maximumNewTokens = 1536
        case .medium:
            value.enableThinking = true; value.thinkingBudget = 512; value.maximumNewTokens = 2048
        case .ultra:
            value.enableThinking = true; value.thinkingBudget = 2048; value.maximumNewTokens = 4096
        }
        return value
    }
    /// Reasoning controls must not reset independently selected sampling.
    public func applying(to current: LocalGenerationOptions) -> LocalGenerationOptions {
        var value = options
        value.temperature = current.temperature
        value.topP = current.topP
        value.seed = current.seed
        return value
    }
    public static func restoring(_ saved: LocalGenerationOptions) throws -> LocalGenerationOptions {
        try saved.validate()
        return (matching(saved) ?? (saved.enableThinking ? .medium : .off)).applying(to: saved)
    }
    public static func matching(_ value: LocalGenerationOptions) -> Self? {
        allCases.first { preset in
            let expected = preset.options
            return value.enableThinking == expected.enableThinking
                && value.thinkingBudget == expected.thinkingBudget
                && value.maximumNewTokens == expected.maximumNewTokens
        }
    }
}

/// Editing accepts either keyboard decimal separator, without treating an empty
/// or incomplete draft as zero. No grouping, exponent or silent range clamping.
public enum LocalSamplingInput {
    public static func decimal(_ text: String, minimum: Double, maximum: Double,
                               excludingMinimum: Bool = false) -> Double? {
        let value = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty, value.utf8.allSatisfy({ (48...57).contains($0) || $0 == 44 || $0 == 46 }),
              value.filter({ $0 == "," || $0 == "." }).count <= 1,
              value.last != ",", value.last != ".",
              let number = Double(value.replacingOccurrences(of: ",", with: ".")), number.isFinite,
              number >= minimum, number <= maximum, !excludingMinimum || number > minimum else { return nil }
        return number
    }
}

/// Deterministic greedy default; seeded temperature/nucleus sampling is opt-in.
/// The random state belongs to one request, never to the native inference graph.
public struct LocalTokenSampler {
    private var state: UInt64
    public init(seed: UInt64) { state = seed }
    private mutating func uniform() -> Double {
        state &+= 0x9e3779b97f4a7c15
        var value = state
        value = (value ^ (value >> 30)) &* 0xbf58476d1ce4e5b9
        value = (value ^ (value >> 27)) &* 0x94d049bb133111eb
        value ^= value >> 31
        return Double(value >> 11) * (1.0 / 9007199254740992.0)
    }
    public mutating func select(_ logits: [Float], options: LocalGenerationOptions) throws -> Int {
        try options.validate()
        guard !logits.isEmpty, logits.count <= 300000, logits.allSatisfy(\.isFinite) else {
            throw LocalAPIError(status: 500, message: "Invalid model logits")
        }
        var best = 0
        for index in logits.indices.dropFirst() where logits[index] > logits[best] { best = index }
        if options.temperature == 0 { return best }
        let maximum = Double(logits[best])
        let weights = logits.map { exp((Double($0) - maximum) / options.temperature) }
        var indices = Array(logits.indices)
        var total = weights.reduce(0, +)
        if options.topP < 1 {
            indices.sort { weights[$0] == weights[$1] ? $0 < $1 : weights[$0] > weights[$1] }
            let threshold = total * options.topP
            var retained = 0.0, count = 0
            repeat { retained += weights[indices[count]]; count += 1 } while retained < threshold && count < indices.count
            indices.removeSubrange(count...)
            total = retained
        }
        let draw = uniform() * total
        var cumulative = 0.0
        for index in indices {
            cumulative += weights[index]
            if draw < cumulative { return index }
        }
        // Floating-point summation at the boundary still selects a retained ID.
        return indices.last ?? best
    }
}
