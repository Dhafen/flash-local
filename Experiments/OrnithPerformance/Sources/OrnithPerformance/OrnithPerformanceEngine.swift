import Foundation
import Metal
import SwiftletCore

public struct PerformanceForwardMetrics: Codable, Sendable {
    public enum Phase: String, Codable, Sendable { case prefill, decode }
    public let phase: Phase
    public let inputTokens: Int
    public let completed: Bool
    public let wallSeconds: Double
    public let metalGPUSeconds: Double?
    public let gpuTimedCommandBuffers: Int
    public let gpuUntimedCommandBuffers: Int
    public let commandBuffers: Int
    public let dispatches: Int
    public let encodeSeconds: Double
    public let blockingWaitSeconds: Double
    public let expertFetchWallSeconds: Double
    public let expertReadCalls: Int
    public let expertReturnedReadBytes: UInt64
    public let expertSummedReadSeconds: Double
    public let expertCacheHits: Int
    public let expertCacheMisses: Int
    public let expertCacheEvictions: Int
    public let expertCacheAllocatedBytes: UInt64
    public let expertCacheBudgetBytes: UInt64
    public let cacheReplacementCount: Int
    public let deviceMetalAllocatedBytes: UInt64
    public let endDevice: PerformanceDeviceSample?
    public let sampledPeak: PerformanceSampledPeak
    // No token IDs, text, prompts or logits are included in performance JSON.
}

public struct PerformanceForwardResult: Sendable {
    public let logits: [Float]
    public let metrics: PerformanceForwardMetrics
}

/// Single-owner, Metal-only QPack execution. No full-checkpoint mmap path or
/// CPU fallback can be selected through this adapter. REAM/GGUF stay outside
/// the experiment, with no dependency on this package.
public actor OrnithPerformanceEngine {
    public static let swiftletRuntimeRevision = "1388b08883d3d469addb5fdb88916ac7c5a19cd6"
    public let memoryPlan: PerformanceMemoryPlan
    public let sourceIdentity: String
    private let model: QwenMetalModel
    private let device: MTLDevice
    private var state = QwenCPUModel.DecodeState()
    private var replacements = 0
    public private(set) var lastForwardMetrics: PerformanceForwardMetrics?

    public init(install: VerifiedQPackInstall, plan: QPackInstallPlan, contextTokens: Int = 8192,
                limits: PerformanceMemoryPlan.Limits = .init(), prefillChunkTokens: Int = 32) throws {
        try plan.source.geometry.requirePublishedOrnith()
        guard install.sourceIdentity == (try plan.source.manifest.identity()),
              (1...32).contains(prefillChunkTokens), let device = MTLCreateSystemDefaultDevice(),
              let sample = PerformanceDeviceSample.current() else {
            throw OrnithPerformanceError.invalidSource("no verified model, memory reading or Metal device")
        }
        let memory = try PerformanceMemoryPlan.make(install: plan, contextTokens: contextTokens,
            baselineFootprintBytes: sample.footprintBytes, osAvailableBytes: sample.availableProcessBytes,
            metalRecommendedWorkingSetBytes: device.recommendedMaxWorkingSetSize,
            retainCPUKVMirror: false, limits: limits)
        self.device = device
        memoryPlan = memory
        sourceIdentity = install.sourceIdentity
        model = try QwenMetalModel(modelDir: install.directory,
            cacheBudgetGB: Double(memory.expertCacheBytes) / 1_073_741_824, retainCPUKVMirror: false)
        guard model.expertCache != nil, !model.retainsCPUKVMirror else {
            throw OrnithPerformanceError.invalidSource("bounded QPack execution is required")
        }
        model.prefillMode = .layerMajor(chunkTokens: prefillChunkTokens)
    }

    public var position: Int { state.position }
    public func reset() { state = QwenCPUModel.DecodeState() }

    public func forward(_ tokens: [Int], phase: PerformanceForwardMetrics.Phase,
                        cancellation: PerformanceCancellation = .init()) throws -> PerformanceForwardResult {
        try ContextWindow(maximumTokens: memoryPlan.contextTokens).validateStep(
            processedTokens: state.position, incomingTokens: tokens.count)
        guard tokens.allSatisfy({ (0..<model.config.vocabSize).contains($0) }),
              phase != .decode || tokens.count == 1 else {
            throw OrnithPerformanceError.invalidSource("invalid model input")
        }
        try adaptCacheBeforeStep()
        let before = try cacheCounters()
        let sampler = PerformanceSampler()
        do {
            let logits = try model.step(tokens, state: state, shouldCancel: {
                cancellation.isCancelled || Task.isCancelled
            })
            let metrics = try metrics(phase: phase, inputTokens: tokens.count, before: before,
                                      sampled: sampler.finish())
            lastForwardMetrics = metrics
            guard metrics.completed, logits.count == model.config.vocabSize,
                  logits.allSatisfy(\.isFinite) else {
                state = QwenCPUModel.DecodeState()
                throw OrnithPerformanceError.integrity("forward did not complete with finite logits")
            }
            return .init(logits: logits, metrics: metrics)
        } catch {
            lastForwardMetrics = try? metrics(phase: phase, inputTokens: tokens.count,
                                              before: before, sampled: sampler.finish())
            // A cancelled/failed step may have mutated only some layers.
            // Never let that partial state contaminate the next conversation.
            state = QwenCPUModel.DecodeState()
            throw error
        }
    }

    private struct CacheCounters {
        let calls: Int
        let bytes: UInt64
        let seconds: Double
        let evictions: Int
    }
    private func cacheCounters() throws -> CacheCounters {
        guard let cache = model.expertCache else { throw OrnithPerformanceError.invalidSource("expert cache missing") }
        let io = cache.ioStatistics
        return .init(calls: io.readCalls, bytes: io.returnedBytes, seconds: io.summedReadSeconds, evictions: cache.evictions)
    }

    private func metrics(phase: PerformanceForwardMetrics.Phase, inputTokens: Int,
                         before: CacheCounters, sampled: PerformanceSampledPeak) throws -> PerformanceForwardMetrics {
        guard let cache = model.expertCache else { throw OrnithPerformanceError.invalidSource("expert cache missing") }
        let step = model.lastStepMetrics
        let io = cache.ioStatistics
        return .init(phase: phase, inputTokens: inputTokens, completed: step.completedWithoutThrow,
            wallSeconds: step.stepWallSeconds,
            metalGPUSeconds: step.gpuUntimedCommandBuffers == 0 && step.gpuTimedCommandBuffers > 0
                ? step.gpuExecutionSeconds : nil,
            gpuTimedCommandBuffers: step.gpuTimedCommandBuffers, gpuUntimedCommandBuffers: step.gpuUntimedCommandBuffers,
            commandBuffers: step.commandBuffersCommitted, dispatches: step.computeDispatchesEncoded,
            encodeSeconds: step.encodeSeconds, blockingWaitSeconds: step.blockingWaitSeconds,
            expertFetchWallSeconds: step.cpuGap.expertFetchSeconds, expertReadCalls: io.readCalls - before.calls,
            expertReturnedReadBytes: io.returnedBytes - before.bytes,
            expertSummedReadSeconds: io.summedReadSeconds - before.seconds,
            expertCacheHits: step.cpuGap.expertFetchHits, expertCacheMisses: step.cpuGap.expertFetchMisses,
            expertCacheEvictions: cache.evictions - before.evictions, expertCacheAllocatedBytes: UInt64(cache.allocatedBytes),
            expertCacheBudgetBytes: UInt64(cache.budgetBytes), cacheReplacementCount: replacements,
            deviceMetalAllocatedBytes: UInt64(device.currentAllocatedSize), endDevice: .current(), sampledPeak: sampled)
    }

    private func adaptCacheBeforeStep() throws {
        guard let cache = model.expertCache, let sample = PerformanceDeviceSample.current() else {
            throw OrnithPerformanceError.invalidSource("cannot refresh memory allowance")
        }
        // The initial plan already reserves the full requested KV context.
        // Add an unknown-growth reserve to the live check without subtracting
        // dense weights a second time from the measured process footprint.
        let allowed = memoryPlan.safeCacheLimit(currentFootprintBytes: sample.footprintBytes,
            currentCacheAllocatedBytes: UInt64(cache.allocatedBytes), osAvailableBytes: sample.availableProcessBytes,
            futureStateGrowthBytes: memoryPlan.kvGrowthTransientBytes)
        guard allowed >= memoryPlan.minimumExpertCacheBytes else {
            throw OrnithPerformanceError.insufficientMemory(required: memoryPlan.minimumExpertCacheBytes, available: allowed)
        }
        if allowed + 16_384 < UInt64(cache.budgetBytes) {
            // Replacement happens only with no command buffer in flight.
            // The old cache is released before the next token allocates slots.
            model.shrinkCache(toGB: Double(allowed) / 1_073_741_824)
            replacements += 1
            guard let resized = model.expertCache, UInt64(resized.budgetBytes) <= allowed else {
                throw OrnithPerformanceError.io("expert cache could not shrink to the live allowance")
            }
        }
    }
}
