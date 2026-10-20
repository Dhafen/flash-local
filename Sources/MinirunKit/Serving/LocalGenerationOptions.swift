import Foundation

/// Generation controls shared by the local UI and the authenticated LAN API.
/// Routing, weights and native graph mathematics are unaffected.
public struct LocalGenerationOptions: Codable, Equatable, Sendable {
    public var maximumNewTokens: Int = 512
    public var enableThinking: Bool = true
    public var thinkingBudget: Int? = nil
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
