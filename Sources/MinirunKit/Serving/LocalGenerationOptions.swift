import Foundation

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
    public static func matching(_ value: LocalGenerationOptions) -> Self? {
        allCases.first { preset in
            let expected = preset.options
            return value.enableThinking == expected.enableThinking
                && value.thinkingBudget == expected.thinkingBudget
                && value.maximumNewTokens == expected.maximumNewTokens
        }
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
