#if canImport(OrnithLES)
import Foundation
import OrnithLES
import MinirunKit
import ModelAdapters

/// On-device ABBA comparison of weight residency. Never changes quantization,
/// the verified graph, routing or the active chat/API owner automatically.
public enum OrnithCacheModeBenchmark {
    public struct Run: Codable, Sendable {
        public let mode: OrnithLESChatGeneration.CacheMode
        public let generation: OrnithLESChatGeneration.Summary
    }

    public struct Report: Codable, Sendable {
        public let promptID = "ornith-cache-mode-abba-v1"
        public let osFileCacheControlled = false
        public let fullModelNumericsVerified = false
        public let runs: [Run]

        public var comparable: Bool {
            guard runs.map(\.mode) == [.osPageCache, .boundedExpertCache, .boundedExpertCache, .osPageCache],
                  let first = runs.first else { return false }
            return runs.allSatisfy { run in
                let result = run.generation
                return result.backend == run.mode.rawValue &&
                    result.promptTokenIDs == first.generation.promptTokenIDs &&
                    result.generatedTokenIDs.count >= 129 &&
                    result.generatedTokenIDs == first.generation.generatedTokenIDs &&
                    result.decodeForwardPasses >= 128 &&
                    result.reusedPromptTokens == 0 &&
                    result.evaluatedPromptTokens == result.promptTokenIDs.count &&
                    result.decodeForwardSeconds > 0 && result.prefillSeconds > 0 &&
                    !result.lowPowerModeAtStart &&
                    result.thermalStateAtStart <= 1 &&
                    result.maximumThermalState == result.thermalStateAtStart &&
                    result.maximumObservedFootprintBytes <= result.memoryPlan.processCeilingBytes &&
                    result.sessionMaximumObservedFootprintBytes <= result.memoryPlan.processCeilingBytes &&
                    (run.mode == .osPageCache ? result.memoryPlan.cacheSlotsPerLayer == 0 :
                        result.memoryPlan.cacheSlotsPerLayer >= 8)
            } && Set(runs.map { $0.generation.thermalStateAtStart }).count == 1
        }

        private func rate(for mode: OrnithLESChatGeneration.CacheMode, prefill: Bool) -> Double? {
            guard comparable else { return nil }
            let selected = runs.filter { $0.mode == mode }.map(\.generation)
            if prefill {
                let count = selected.reduce(0) { $0 + $1.evaluatedPromptTokens }
                let seconds = selected.reduce(0.0) { $0 + $1.prefillSeconds }
                return seconds > 0 ? Double(count) / seconds : nil
            }
            let count = selected.reduce(0) { $0 + $1.decodeForwardPasses }
            let seconds = selected.reduce(0.0) { $0 + $1.decodeForwardSeconds }
            return seconds > 0 ? Double(count) / seconds : nil
        }

        public var pageCacheDecodeTokensPerSecond: Double? { rate(for: .osPageCache, prefill: false) }
        public var expertCacheDecodeTokensPerSecond: Double? { rate(for: .boundedExpertCache, prefill: false) }

        public var preferredCacheMode: OrnithLESChatGeneration.CacheMode? {
            guard comparable, let bd = pageCacheDecodeTokensPerSecond,
                  let cd = expertCacheDecodeTokensPerSecond,
                  let bp = rate(for: .osPageCache, prefill: true),
                  let cp = rate(for: .boundedExpertCache, prefill: true) else { return nil }
            return Self.improves(baselineDecode: bd, candidateDecode: cd,
                baselinePrefill: bp, candidatePrefill: cp) ? .boundedExpertCache : .osPageCache
        }

        /// A ten-percent decode gain must not cost more than 15% of prefill.
        /// This is a gate for offering the option, not proof of 15 tokens/s.
        public static func improves(baselineDecode: Double, candidateDecode: Double,
                                    baselinePrefill: Double, candidatePrefill: Double) -> Bool {
            [baselineDecode, candidateDecode, baselinePrefill, candidatePrefill].allSatisfy { $0.isFinite && $0 > 0 } &&
                candidateDecode >= baselineDecode * 1.10 &&
                candidatePrefill >= baselinePrefill * 0.85
        }
    }

    public static func execute(artifact: VerifiedGGUFArtifact, prepared: URL,
                               baseline: LESNumericalCheck, control: LESControl,
                               contextProfile: LocalContextProfile,
                               progress: @Sendable (String) -> Void,
                               onAdmission: (LESMemoryAdmission) -> Void) throws -> Report {
        guard baseline.nativeBenchmarkAdmitted, baseline.modelSHA256 == artifact.manifest.sha256,
              baseline.nativeRevision == LESContext.revision else {
            throw GGUFError.invalid("Expertcache kräver en godkänd kontroll för aktuell modell och motor")
        }
        let modes: [OrnithLESChatGeneration.CacheMode] =
            [.osPageCache, .boundedExpertCache, .boundedExpertCache, .osPageCache]
        let question = Qwen35ChatMessage(role: .user, content: LESBenchmark.prompt)
        var runs: [Run] = []
        for (index, mode) in modes.enumerated() {
            if control.shouldAbort() { throw CancellationError() }
            let generation = try autoreleasepool {
                let engine = try OrnithLESChatGeneration(artifact: artifact, prepared: prepared,
                    check: baseline, control: control, contextProfile: contextProfile,
                    cacheMode: mode, onAdmission: onAdmission)
                defer { engine.close() }
                var options = LocalGenerationOptions()
                options.maximumNewTokens = 129
                options.enableThinking = true
                return try engine.generate(messages: [question], options: options) { event in
                    switch event {
                    case .prefill(let value):
                        progress("Expertcache ABBA \(index + 1)/4 · \(mode.rawValue) · prompt \(value.promptTokensCompleted)/\(value.promptTokensTotal)")
                    case .token(_, _, let value):
                        progress("Expertcache ABBA \(index + 1)/4 · \(mode.rawValue) · \(value.generatedTokens)/129 token")
                    case .trailingText: break
                    }
                }
            }
            runs.append(.init(mode: mode, generation: generation))
        }
        return .init(runs: runs)
    }
}
#endif
