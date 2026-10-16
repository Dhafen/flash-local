import Foundation
import GGUFCore
import MLX
import MinirunKit
import ModelAdapters
import StorageCore

/// A short, local diagnostic. CPU and MLX models live in separate function
/// scopes; only bounded copied reference logits survive into the MLX phase.
/// This never downloads weights or infers success for untested positions.
public enum Qwen4ExpLocalParity {
    public enum Phase: String, Sendable { case cpuReference, streamedMLX }
    /// Hardware class only; no hostname, account name or unique device ID.
    public struct Environment: Codable, Sendable {
        public let platform: String
        public let deviceIdentifier: String?
        public let operatingSystem: String
        public let physicalMemoryBytes: UInt64

        init() {
            #if targetEnvironment(simulator)
            platform = "iOSSimulator"
            #elseif os(iOS)
            platform = "iOSDevice"
            #elseif os(macOS)
            platform = "macOS"
            #else
            platform = "unsupported"
            #endif
            #if os(iOS)
            deviceIdentifier = Sysctl.string("hw.machine")
            #else
            deviceIdentifier = Sysctl.string("hw.model")
            #endif
            operatingSystem = ProcessInfo.processInfo.operatingSystemVersionString
            physicalMemoryBytes = ProcessInfo.processInfo.physicalMemory
        }
    }
    public struct Position: Codable, Sendable {
        public let position: Int
        public let inputTokenID: Int
        public let maximumAbsoluteError: Double
        public let relativeL2Error: Double
        public let referenceTopTokenID: Int
        public let streamedTopTokenID: Int
        public let referenceWinnerMargin: Double?
        public let withinNumericalTolerance: Bool
    }
    public struct Report: Codable, Sendable {
        public let schemaVersion: Int
        public let timestamp: String
        public let environment: Environment
        public let modelSHA256: String
        public let modelFileBytes: UInt64
        public let modelProvenance: GGUFModelProvenance?
        public let sourceRevision: String
        public let referenceRevision: String
        public let tokenIDs: [Int]
        public let positions: [Position]
        public let absoluteTolerance: Double
        public let relativeL2Tolerance: Double
        public let passedAtTestedPositions: Bool
        public let budgetBytes: UInt64
        public let cpuSeconds: Double
        public let streamedSeconds: Double
        public let cpuMaximumObservedFootprintBytes: UInt64
        public let streamedMaximumObservedFootprintBytes: UInt64
        public let readerBytes: UInt64
        public let projections: GGUFProjectionStatistics
    }
    static let absoluteTolerance = 0.01
    static let relativeTolerance = 1e-3
    private struct CPUPass {
        let rows: [[Float]]
        let peak: UInt64
        let seconds: Double
    }

    /// Canonical template/tokenizer, with explicit refusal above 32 prompt
    /// tokens. Nothing is silently truncated to make a reference test pass.
    public static func run(artifact: VerifiedGGUFArtifact, messages: [Qwen35ChatMessage],
                           budgetBytes: UInt64 = 4_000_000_000, rawCacheBytes: Int = 64 << 20,
                           cancellationRequested: @escaping @Sendable () -> Bool = { false },
                           onProgress: (Phase, Int, Int) -> Void = { _, _, _ in }) throws -> Report {
        let tokens = try autoreleasepool {
            try CanonicalQwen35Tokenizer(artifact: artifact).chat(messages, maximumTokens: 32).tokenIDs
        }
        return try run(artifact: artifact, tokenIDs: tokens, budgetBytes: budgetBytes,
            rawCacheBytes: rawCacheBytes, cancellationRequested: cancellationRequested, onProgress: onProgress)
    }

    public static func run(artifact: VerifiedGGUFArtifact, tokenIDs: [Int],
                           budgetBytes: UInt64 = 4_000_000_000, rawCacheBytes: Int = 64 << 20,
                           cancellationRequested: @escaping @Sendable () -> Bool = { false },
                           onProgress: (Phase, Int, Int) -> Void = { _, _, _ in }) throws -> Report {
        let vocabularySize = try GGUFStreamedDecoder.vocabularySize(index: artifact.index)
        guard (1...32).contains(tokenIDs.count), tokenIDs.allSatisfy({ (0..<vocabularySize).contains($0) }),
              UInt64(vocabularySize) * UInt64(tokenIDs.count) * 4 <= 64 << 20,
              budgetBytes <= 5_500_000_000 else { throw GGUFError.limit("short local parity input/output bound") }
        let reference = try cpuPass(artifact: artifact, tokens: tokenIDs, budget: budgetBytes,
            cancelled: cancellationRequested, progress: onProgress)
        // cpuPass has returned and destroyed its model/context before this
        // function constructs any streamed decoder, states or expert cache.
        return try streamedPass(artifact: artifact, tokens: tokenIDs, reference: reference,
            budget: budgetBytes, rawCacheBytes: rawCacheBytes, cancelled: cancellationRequested, progress: onProgress)
    }

    private static func cpuPass(artifact: VerifiedGGUFArtifact, tokens: [Int], budget: UInt64,
                                cancelled: @escaping @Sendable () -> Bool,
                                progress: (Phase, Int, Int) -> Void) throws -> CPUPass {
        try autoreleasepool {
            let started = ProcessInfo.processInfo.systemUptime
            progress(.cpuReference, 0, tokens.count)
            let cpu = try Qwen4ExpCPUReference(artifact: artifact, contextTokens: 128,
                budgetBytes: budget, cancellationRequested: cancelled)
            var rows: [[Float]] = []; rows.reserveCapacity(tokens.count)
            for (i, token) in tokens.enumerated() {
                if cancelled() { throw CancellationError() }
                rows.append(try cpu.logits(for: token))
                progress(.cpuReference, i + 1, tokens.count)
            }
            return CPUPass(rows: rows, peak: cpu.maximumObservedFootprintBytes,
                seconds: ProcessInfo.processInfo.systemUptime - started)
        }
    }

    private static func streamedPass(artifact: VerifiedGGUFArtifact, tokens: [Int], reference: CPUPass,
                                     budget: UInt64, rawCacheBytes: Int, cancelled: @escaping @Sendable () -> Bool,
                                     progress: (Phase, Int, Int) -> Void) throws -> Report {
        try autoreleasepool {
            if cancelled() { throw CancellationError() }
            let started = ProcessInfo.processInfo.systemUptime
            progress(.streamedMLX, 0, tokens.count)
            let decoder = try GGUFStreamedDecoder(artifact: artifact, contextTokens: 128,
                budgetBytes: budget, rawCacheBytes: rawCacheBytes)
            var peak: UInt64 = 0, positions: [Position] = []
            decoder.onFootprintObservation = { peak = max(peak, $0.footprintBytes) }
            defer { decoder.onFootprintObservation = nil; decoder.reset() }
            for (i, token) in tokens.enumerated() {
                if cancelled() { throw CancellationError() }
                let result = try autoreleasepool {
                    let actual = try decoder.logits(for: UInt64(token), cancellationRequested: cancelled).asArray(Float.self)
                    return try compare(position: i, tokenID: token, reference: reference.rows[i], actual: actual)
                }
                positions.append(result); progress(.streamedMLX, i + 1, tokens.count)
            }
            let io = decoder.readStatistics
            let cache = decoder.rawCacheStatistics
            guard !io.byteCountOverflowed, cache.isBudgetRespected, cache.faults.isEmpty else {
                throw GGUFError.invalid("parity reader overflow, cache budget or lease fault")
            }
            return Report(schemaVersion: 1, timestamp: ISO8601DateFormatter().string(from: Date()),
                environment: Environment(),
                modelSHA256: artifact.manifest.sha256, modelFileBytes: artifact.manifest.fileByteCount,
                modelProvenance: artifact.manifest.provenance,
                sourceRevision: BuildInfo.gitRevision, referenceRevision: Qwen4ExpCPUReference.sourceRevision,
                tokenIDs: tokens, positions: positions, absoluteTolerance: absoluteTolerance,
                relativeL2Tolerance: relativeTolerance,
                passedAtTestedPositions: positions.allSatisfy {
                    $0.withinNumericalTolerance && $0.referenceTopTokenID == $0.streamedTopTokenID
                }, budgetBytes: budget, cpuSeconds: reference.seconds,
                streamedSeconds: ProcessInfo.processInfo.systemUptime - started,
                cpuMaximumObservedFootprintBytes: reference.peak, streamedMaximumObservedFootprintBytes: peak,
                readerBytes: io.bytesRead, projections: decoder.projectionStatistics)
        }
    }

    static func compare(position: Int, tokenID: Int, reference: [Float], actual: [Float]) throws -> Position {
        guard !reference.isEmpty, reference.count == actual.count, reference.count <= 300000,
              reference.allSatisfy(\.isFinite), actual.allSatisfy(\.isFinite) else {
            throw GGUFError.invalid("parity logits shape or finite values")
        }
        let targetTop = try Qwen4ExpGreedyToken.select(reference), actualTop = try Qwen4ExpGreedyToken.select(actual)
        var maximum = 0.0, errorSquared = 0.0, targetSquared = 0.0
        var second = -Double.infinity
        for i in reference.indices {
            let target = Double(reference[i]), delta = Double(actual[i]) - target
            maximum = max(maximum, abs(delta)); errorSquared += delta * delta; targetSquared += target * target
            if i != targetTop { second = max(second, target) }
        }
        let relative = sqrt(errorSquared / max(targetSquared, 1e-30))
        return Position(position: position, inputTokenID: tokenID, maximumAbsoluteError: maximum,
            relativeL2Error: relative, referenceTopTokenID: targetTop, streamedTopTokenID: actualTop,
            referenceWinnerMargin: second.isFinite ? Double(reference[targetTop]) - second : nil,
            withinNumericalTolerance: maximum <= absoluteTolerance && relative <= relativeTolerance)
    }
}
