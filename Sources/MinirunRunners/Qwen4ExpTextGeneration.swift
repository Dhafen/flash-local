import Foundation
import GGUFCore
import MLX
import MinirunKit
import ModelAdapters
import StorageCore

/// Deterministic initial sampling policy. Equal scores choose the lowest ID;
/// invalid logits fail instead of disguising numerical failures as output.
enum Qwen4ExpGreedyToken {
    static func select(_ logits: [Float]) throws -> Int {
        guard !logits.isEmpty, logits.count <= 300000, logits.allSatisfy(\.isFinite) else {
            throw GGUFError.invalid("empty, oversized or nonfinite generation logits")
        }
        var best = 0
        for i in logits.indices.dropFirst() where logits[i] > logits[best] { best = i }
        return best
    }
}

/// Carries an incomplete UTF-8 scalar over token boundaries. Complete invalid
/// subsequences use Swift's normal replacement policy; at most 3 bytes wait.
struct Qwen4ExpTokenText {
    private var pending: [UInt8] = []
    var pendingByteCount: Int { pending.count }
    mutating func append(_ data: Data) -> String {
        var bytes = pending; bytes.append(contentsOf: data); pending.removeAll(keepingCapacity: true)
        guard !bytes.isEmpty else { return "" }
        var lead = bytes.count - 1
        while lead >= 0, bytes[lead] & 0xc0 == 0x80 { lead -= 1 }
        var keep = bytes.count
        if lead >= 0 {
            let first = bytes[lead]
            let needed = (0xc2...0xdf).contains(first) ? 2 : (0xe0...0xef).contains(first) ? 3
                : (0xf0...0xf4).contains(first) ? 4 : 1
            if needed > bytes.count - lead {
                var valid = bytes[(lead + 1)...].allSatisfy { $0 & 0xc0 == 0x80 }
                if lead + 1 < bytes.count {
                    let second = bytes[lead + 1]
                    if first == 0xe0 && second < 0xa0 || first == 0xed && second >= 0xa0
                        || first == 0xf0 && second < 0x90 || first == 0xf4 && second >= 0x90 { valid = false }
                }
                if valid { keep = lead; pending = Array(bytes[lead...]) }
            }
        }
        return String(decoding: bytes[..<keep], as: UTF8.self)
    }
    mutating func finish() -> String {
        defer { pending.removeAll(keepingCapacity: true) }
        return String(decoding: pending, as: UTF8.self)
    }
}

/// Synchronous model execution, confined to one off-main inference executor.
/// Every output ID comes from the streamed decoder's actual logits. There are
/// no network calls, mock responses, hidden context truncation or model-sized
/// arrays. The app owns the worker/cancellation lifecycle around this object.
public final class Qwen4ExpTextGeneration {
    public enum StopReason: String, Codable, Sendable { case endOfGeneration, tokenLimit, cancelled }
    public struct Progress: Sendable {
        public let promptTokensCompleted: Int
        public let promptTokensTotal: Int
        public let generatedTokens: Int
        public let elapsedSeconds: Double
        public let ioBytesRead: UInt64
        public let footprintBytes: UInt64
    }
    public enum Event: Sendable {
        case prefill(Progress)
        case token(id: Int, textDelta: String, progress: Progress)
        case trailingText(String)
    }
    public struct Summary: Codable, Sendable {
        public let modelSHA256: String
        public let promptTokenIDs: [Int]
        public let generatedTokenIDs: [Int]
        public let stopReason: StopReason
        public let prefillSeconds: Double
        public let decodeForwardSeconds: Double
        public let decodeForwardPasses: Int
        public let elapsedSeconds: Double
        /// Returned reader bytes, including any read-ahead. Physical flash I/O
        /// amplification and filesystem/controller caching are not measured here.
        public let ioBytesRead: UInt64
        public let projections: GGUFProjectionStatistics
        public let rawCacheHits: Int
        public let rawCacheMisses: Int
        public let rawCacheEvictions: Int
        public let rawCacheCapacityBytes: Int
        /// Lifetime of this engine's weight source, which can span reset runs.
        public let rawCacheLifetimePeakBytes: Int
        public let maximumObservedFootprintBytes: UInt64
        /// Explicitly process-lifetime, since a previous run may own this peak.
        public let processLifetimePeakFootprintBytes: UInt64
        public var residentWeightCache: GGUFResidentWeightCacheStatistics? = nil
        public var promptOutputProjectionsSkipped: Int? = nil
        public var prefillIOBytesRead: UInt64? = nil
        public var decodeIOBytesRead: UInt64? = nil
        public var promptTokensCompleted: Int? = nil
        /// A stopped partial forward is excluded from completed-forward speed.
        public var interruptedDecodeForwardSeconds: Double? = nil
        public var decodeTokensPerSecond: Double? {
            decodeForwardPasses > 0 && decodeForwardSeconds > 0 ? Double(decodeForwardPasses) / decodeForwardSeconds : nil
        }
    }
    private let artifact: VerifiedGGUFArtifact
    private let tokenizer: CanonicalQwen35Tokenizer
    private let decoder: Qwen4ExpDecoder
    private let contextTokens: Int
    private var running = false

    public init(artifact: VerifiedGGUFArtifact, contextTokens: Int = 1024,
                budgetBytes: UInt64 = 4_000_000_000, rawCacheBytes: Int = 64 << 20,
                residentCacheBytes: Int = 0) throws {
        guard (16...65536).contains(contextTokens), budgetBytes <= 5_500_000_000,
              let footprint = ProcessFootprint.current(), footprint.footprintBytes < budgetBytes else {
            throw GGUFError.limit("context, total process budget or physical footprint availability")
        }
        self.artifact = artifact; self.contextTokens = contextTokens
        tokenizer = try CanonicalQwen35Tokenizer(artifact: artifact)
        decoder = try Qwen4ExpDecoder(artifact: artifact, contextTokens: contextTokens,
            budgetBytes: budgetBytes, rawCacheBytes: rawCacheBytes, residentCacheBytes: residentCacheBytes)
        guard decoder.layout.vocabularySize == tokenizer.vocabularySize else {
            throw GGUFError.invalid("tokenizer and model vocabulary sizes disagree")
        }
    }

    public func generate(messages: [Qwen35ChatMessage], maximumNewTokens: Int = 128,
                         cancellationRequested: () -> Bool = { false },
                         captureCancelledSummary: Bool = false,
                         onEvent: (Event) throws -> Void) throws -> Summary {
        guard !running, (1...4096).contains(maximumNewTokens), maximumNewTokens < contextTokens else {
            throw GGUFError.limit("generation already active or output/context budget invalid")
        }
        running = true; decoder.reset()
        defer { decoder.onFootprintObservation = nil; decoder.reset(); running = false }
        let started = ProcessInfo.processInfo.systemUptime
        let baselineIO = decoder.readStatistics.bytesRead
        let baselineProjections = decoder.projectionStatistics
        let baselineCache = decoder.rawCacheStatistics
        var peak: UInt64 = 0, lifetimePeak: UInt64 = 0
        decoder.onFootprintObservation = {
            peak = max(peak, $0.footprintBytes); lifetimePeak = max(lifetimePeak, $0.peakFootprintBytes)
        }
        func check() throws {
            if cancellationRequested() { throw CancellationError() }
            try artifact.assertUnchanged()
            guard let current = ProcessFootprint.current() else { throw GGUFError.limit("physical footprint unavailable") }
            peak = max(peak, current.footprintBytes); lifetimePeak = max(lifetimePeak, current.peakFootprintBytes)
        }
        func ioBytes() throws -> UInt64 {
            let stats = decoder.readStatistics
            guard !stats.byteCountOverflowed, stats.bytesRead >= baselineIO else { throw GGUFError.invalid("I/O counter overflow") }
            return stats.bytesRead - baselineIO
        }
        try check()
        let prompt = try tokenizer.chat(messages, maximumTokens: contextTokens - maximumNewTokens)
        var logits: [Float] = [], generated: [Int] = [], text = Qwen4ExpTokenText(), outputBytes = 0
        func progress(_ completed: Int) throws -> Progress {
            Progress(promptTokensCompleted: completed, promptTokensTotal: prompt.tokenIDs.count,
                generatedTokens: generated.count, elapsedSeconds: ProcessInfo.processInfo.systemUptime - started,
                ioBytesRead: try ioBytes(), footprintBytes: ProcessFootprint.current()?.footprintBytes ?? peak)
        }
        var completedPrompt = 0, prefillSeconds = 0.0, prefillIO: UInt64?
        var reason = StopReason.tokenLimit, decodeSeconds = 0.0, decodePasses = 0
        var interruptedDecodeSeconds = 0.0, pendingForwardStart: Double?
        do {
            for (i, id) in prompt.tokenIDs.enumerated() {
                try check()
                try autoreleasepool {
                    if i + 1 == prompt.tokenIDs.count {
                        logits = try decoder.logits(for: UInt64(id), cancellationRequested: cancellationRequested).asArray(Float.self)
                    } else { try decoder.consume(UInt64(id), cancellationRequested: cancellationRequested) }
                }
                completedPrompt = i + 1
                try onEvent(.prefill(progress(i + 1)))
            }
            prefillSeconds = ProcessInfo.processInfo.systemUptime - started
            prefillIO = try ioBytes()
            for step in 0..<maximumNewTokens {
                try check()
                let id = try Qwen4ExpGreedyToken.select(logits)
                if tokenizer.isEndOfGeneration(id) { reason = .endOfGeneration; break }
                let bytes = try tokenizer.bytes(for: id, includeControl: true)
                outputBytes += bytes.count
                guard outputBytes <= 4 << 20 else { throw GGUFError.limit("generated text exceeds 4 MiB") }
                generated.append(id)
                try onEvent(.token(id: id, textDelta: text.append(bytes), progress: progress(prompt.tokenIDs.count)))
                if step + 1 < maximumNewTokens {
                    let begin = ProcessInfo.processInfo.systemUptime
                    pendingForwardStart = begin
                    logits = try autoreleasepool { try decoder.logits(for: UInt64(id), cancellationRequested: cancellationRequested).asArray(Float.self) }
                    decodeSeconds += ProcessInfo.processInfo.systemUptime - begin; decodePasses += 1
                    pendingForwardStart = nil
                }
            }
            try check()
        } catch {
            guard captureCancelledSummary, error is CancellationError, cancellationRequested() else { throw error }
            reason = .cancelled
            if prefillIO == nil { prefillSeconds = ProcessInfo.processInfo.systemUptime - started }
            if let pendingForwardStart {
                interruptedDecodeSeconds = ProcessInfo.processInfo.systemUptime - pendingForwardStart
            }
        }
        let trailing = text.finish()
        if !trailing.isEmpty { try onEvent(.trailingText(trailing)) }
        let cache = decoder.rawCacheStatistics
        guard cache.faults.isEmpty, cache.isBudgetRespected,
              cache.cacheHitCount >= baselineCache.cacheHitCount,
              cache.cacheMissCount >= baselineCache.cacheMissCount,
              cache.cacheEvictionCount >= baselineCache.cacheEvictionCount else {
            throw GGUFError.invalid("raw cache accounting or lease fault")
        }
        let totalIO = try ioBytes(), promptIO = prefillIO ?? totalIO
        return Summary(modelSHA256: artifact.manifest.sha256, promptTokenIDs: prompt.tokenIDs,
            generatedTokenIDs: generated, stopReason: reason, prefillSeconds: prefillSeconds,
            decodeForwardSeconds: decodeSeconds, decodeForwardPasses: decodePasses,
            elapsedSeconds: ProcessInfo.processInfo.systemUptime - started, ioBytesRead: totalIO,
            projections: try decoder.projectionStatistics.subtracting(baselineProjections),
            rawCacheHits: cache.cacheHitCount - baselineCache.cacheHitCount,
            rawCacheMisses: cache.cacheMissCount - baselineCache.cacheMissCount,
            rawCacheEvictions: cache.cacheEvictionCount - baselineCache.cacheEvictionCount,
            rawCacheCapacityBytes: cache.budgetBytes, rawCacheLifetimePeakBytes: cache.peakInUseBytes,
            maximumObservedFootprintBytes: peak, processLifetimePeakFootprintBytes: lifetimePeak,
            residentWeightCache: decoder.residentCacheStatistics,
            promptOutputProjectionsSkipped: min(completedPrompt, max(0, prompt.tokenIDs.count - 1)),
            prefillIOBytesRead: promptIO, decodeIOBytesRead: totalIO - promptIO,
            promptTokensCompleted: completedPrompt, interruptedDecodeForwardSeconds: interruptedDecodeSeconds)
    }
}
