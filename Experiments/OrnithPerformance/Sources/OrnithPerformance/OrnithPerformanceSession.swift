import Foundation
import SwiftletCore

protocol PerformanceForwarding: Sendable {
    var contextCapacity: Int { get async }
    var position: Int { get async }
    func reset() async
    func forward(_ tokens: [Int], phase: PerformanceForwardMetrics.Phase,
                 cancellation: PerformanceCancellation) async throws -> PerformanceForwardResult
}
extension OrnithPerformanceEngine: PerformanceForwarding {
    public var contextCapacity: Int { memoryPlan.contextTokens }
}
protocol OrnithChatCoding: Sendable {
    var endTokenIDs: Set<Int> { get }
    func promptTokens(messages: [OrnithChatMessage], thinking: Bool) throws -> [Int]
    func decode(_ tokens: [Int]) -> String
}
extension OrnithChatCodec: OrnithChatCoding {}

public struct OrnithGenerationReport: Encodable, Sendable {
    public enum FinishReason: String, Codable, Sendable { case stop, length, cancelled }
    public let schemaVersion = 1
    public let runtimeRevision = OrnithPerformanceEngine.swiftletRuntimeRevision
    public let fullModelNumericsVerified = false
    public let sampling = "greedy"
    public let contextTokens: Int
    public let promptTokens: Int
    public let evaluatedPromptTokens: Int
    public let reusedPromptTokens: Int
    public let generatedTokens: Int
    public let elapsedSeconds: Double
    public let finishReason: FinishReason
    public let forwards: [PerformanceForwardMetrics]
    public var decodeForwardTokensPerSecond: Double? {
        let decode = forwards.filter { $0.phase == .decode && $0.completed }
        let seconds = decode.reduce(0) { $0 + $1.wallSeconds }
        return seconds > 0 ? Double(decode.count) / seconds : nil
    }
    public var prefillSeconds: Double { forwards.filter { $0.phase == .prefill }.reduce(0) { $0 + $1.wallSeconds } }
    public var maximumObservedFootprintBytes: UInt64 {
        forwards.map { $0.sampledPeak.maximumObservedFootprintBytes }.max() ?? 0
    }
    public var expertReturnedReadBytes: UInt64 { forwards.reduce(0) { $0 + $1.expertReturnedReadBytes } }

    private enum CodingKeys: String, CodingKey {
        case schemaVersion, runtimeRevision, fullModelNumericsVerified, sampling, contextTokens
        case promptTokens, evaluatedPromptTokens, reusedPromptTokens, generatedTokens, elapsedSeconds
        case finishReason, forwards, decodeForwardTokensPerSecond, prefillSeconds
        case maximumObservedFootprintBytes, expertReturnedReadBytes, forwardMetricsIncludeInterruptedWork
    }
    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(schemaVersion, forKey: .schemaVersion)
        try c.encode(runtimeRevision, forKey: .runtimeRevision)
        try c.encode(fullModelNumericsVerified, forKey: .fullModelNumericsVerified)
        try c.encode(sampling, forKey: .sampling)
        try c.encode(contextTokens, forKey: .contextTokens)
        try c.encode(promptTokens, forKey: .promptTokens)
        try c.encode(evaluatedPromptTokens, forKey: .evaluatedPromptTokens)
        try c.encode(reusedPromptTokens, forKey: .reusedPromptTokens)
        try c.encode(generatedTokens, forKey: .generatedTokens)
        try c.encode(elapsedSeconds, forKey: .elapsedSeconds)
        try c.encode(finishReason, forKey: .finishReason)
        try c.encode(forwards, forKey: .forwards)
        try c.encodeIfPresent(decodeForwardTokensPerSecond, forKey: .decodeForwardTokensPerSecond)
        try c.encode(prefillSeconds, forKey: .prefillSeconds)
        try c.encode(maximumObservedFootprintBytes, forKey: .maximumObservedFootprintBytes)
        try c.encode(expertReturnedReadBytes, forKey: .expertReturnedReadBytes)
        // Failed/interrupted engine steps are available separately on the
        // engine. These per-generation sums contain successful forwards only.
        try c.encode(false, forKey: .forwardMetricsIncludeInterruptedWork)
    }
}

public struct OrnithGenerationResult: Sendable {
    public let text: String
    public let report: OrnithGenerationReport
}

/// Chat ownership and exact-token prefix reuse over the bounded Metal runner.
/// One active request; cancellation is a separate thread-safe handle, so Stop
/// does not wait behind a long actor-isolated prompt step.
public actor OrnithPerformanceSession {
    private let engine: any PerformanceForwarding
    private let codec: any OrnithChatCoding
    private var cachedTokens: [Int] = []
    private var active: PerformanceCancellation?
    private var resetting = false

    public init(engine: OrnithPerformanceEngine, codec: OrnithChatCodec) {
        self.engine = engine
        self.codec = codec
    }
    init(testingEngine: any PerformanceForwarding, codec: any OrnithChatCoding) {
        engine = testingEngine
        self.codec = codec
    }

    public func reset() async {
        resetting = true
        defer { resetting = false }
        active?.cancel()
        await engine.reset()
        cachedTokens = []
    }

    public func generate(messages: [OrnithChatMessage], maximumNewTokens: Int = 512,
                         thinking: Bool = false, cancellation: PerformanceCancellation = .init(),
                         onDelta: @Sendable (String) async -> Void = { _ in }) async throws -> OrnithGenerationResult {
        guard active == nil, !resetting else { throw OrnithPerformanceError.io("a generation is already active") }
        guard maximumNewTokens > 0, maximumNewTokens <= 4096 else {
            throw OrnithPerformanceError.invalidSource("invalid output token limit")
        }
        active = cancellation
        defer { active = nil }
        let started = ProcessInfo.processInfo.systemUptime
        let prompt = try codec.promptTokens(messages: messages, thinking: thinking)
        let capacity = await engine.contextCapacity
        let admitted = try ContextWindow(maximumTokens: capacity).admittedMaxNew(
            processedTokens: 0, incomingTokens: prompt.count, requestedMaxNew: maximumNewTokens)
        var reused = 0
        if !cachedTokens.isEmpty, prompt.count > cachedTokens.count,
           prompt.starts(with: cachedTokens), await engine.position == cachedTokens.count {
            reused = cachedTokens.count
        } else {
            await engine.reset()
            cachedTokens = []
        }
        var forwards: [PerformanceForwardMetrics] = []
        var generated: [Int] = []
        var printed = ""
        var finish = OrnithGenerationReport.FinishReason.length
        do {
            try checkCancellation(cancellation)
            let suffix = Array(prompt.dropFirst(reused))
            var forward = try await engine.forward(suffix, phase: .prefill, cancellation: cancellation)
            cachedTokens.append(contentsOf: suffix)
            forwards.append(forward.metrics)
            for index in 0..<admitted {
                try checkCancellation(cancellation)
                let token = try Self.greedy(forward.logits)
                if codec.endTokenIDs.contains(token) { finish = .stop; break }
                generated.append(token)
                let decoded = codec.decode(generated)
                if let update = StreamingText.delta(printed: printed, decoded: decoded) {
                    printed = update.printed
                    await onDelta(update.delta)
                }
                if index + 1 < admitted {
                    forward = try await engine.forward([token], phase: .decode, cancellation: cancellation)
                    cachedTokens.append(token)
                    forwards.append(forward.metrics)
                }
            }
            try checkCancellation(cancellation)
        } catch {
            await engine.reset()
            cachedTokens = []
            if cancellation.isCancelled || Task.isCancelled { finish = .cancelled }
            else { throw error }
        }
        let decoded = codec.decode(generated)
        if let final = StreamingText.finalDelta(printed: printed, decoded: decoded) { await onDelta(final) }
        let report = OrnithGenerationReport(contextTokens: capacity, promptTokens: prompt.count,
            evaluatedPromptTokens: prompt.count - reused, reusedPromptTokens: reused, generatedTokens: generated.count,
            elapsedSeconds: max(0, ProcessInfo.processInfo.systemUptime - started), finishReason: finish, forwards: forwards)
        return .init(text: decoded, report: report)
    }

    private func checkCancellation(_ control: PerformanceCancellation) throws {
        if control.isCancelled || Task.isCancelled { throw OrnithPerformanceError.cancelled }
    }

    static func greedy(_ logits: [Float]) throws -> Int {
        guard !logits.isEmpty, logits.allSatisfy(\.isFinite) else {
            throw OrnithPerformanceError.integrity("invalid output logits")
        }
        var selected = 0
        for index in 1..<logits.count where logits[index] > logits[selected] { selected = index }
        return selected
    }
}
