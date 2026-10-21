import Darwin
import Foundation
import FlashOrnithLES
import GGUFCore
import MinirunKit
import ModelAdapters
import StorageCore

public final class LESControl: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false
    private var observedPeak: UInt64 = 0
    public init() {}
    public func cancel() { lock.lock(); cancelled = true; lock.unlock() }
    public var isCancellationRequested: Bool { lock.lock(); defer { lock.unlock() }; return cancelled }
    public var maximumObservedFootprintBytes: UInt64 { lock.lock(); defer { lock.unlock() }; return observedPeak }
    public func shouldAbort() -> Bool {
        let sample = ProcessFootprint.current()
        lock.lock(); defer { lock.unlock() }
        guard let sample else { return true }
        observedPeak = max(observedPeak, sample.footprintBytes)
        return cancelled || sample.footprintBytes >= 5_500_000_000 - (128 << 20) ||
            sample.availableBytes.map({ $0 < 128 << 20 }) == true
    }
}

public struct LESMetrics: Codable, Sendable, Equatable {
    public let ioBytes, ioCalls, expertReads, selectedExpertUnion: UInt64
    public let predictions, predictionHits, prefetchBytes: UInt64
    public let expertManagementNanoseconds, installNanoseconds: UInt64
    public let metalDispatches, metalCommandBuffers, metalNanoseconds: UInt64
    public let routingDigest, routingDecisions: UInt64
    public let canonicalRoutingDigest: UInt64?
    public let canonicalExpertSetDigest: UInt64?
    public let demandCacheHits, demandCacheMisses: UInt64
    public let metalTimestampedCommandBuffers: UInt64
    public let configuredPhysicalBatchTokens, maximumPhysicalBatchTokens: Int?
    public let skippedIntermediateOutputBatches: UInt64?
    init(_ value: flo_metrics) {
        ioBytes = value.io_bytes; ioCalls = value.io_calls; expertReads = value.misses
        selectedExpertUnion = value.selected_experts; predictions = value.predicted
        predictionHits = value.prediction_hits; prefetchBytes = value.prefetch_bytes
        expertManagementNanoseconds = value.process_ns; installNanoseconds = value.install_ns
        metalDispatches = value.dispatches; metalCommandBuffers = value.command_buffers; metalNanoseconds = value.gpu_ns
        routingDigest = value.routing_digest; routingDecisions = value.routing_decisions
        canonicalRoutingDigest = value.canonical_routing_digest
        canonicalExpertSetDigest = value.canonical_expert_set_digest
        demandCacheHits = value.demand_hits; demandCacheMisses = value.demand_misses
        metalTimestampedCommandBuffers = value.timestamped_command_buffers
        configuredPhysicalBatchTokens = Int(value.configured_physical_batch_tokens)
        maximumPhysicalBatchTokens = Int(value.maximum_physical_batch_tokens)
        skippedIntermediateOutputBatches = value.skipped_intermediate_output_batches
    }
}

public struct LESProjectionProbe: Codable, Sendable, Equatable {
    public let samples, columns, weightType, cpuInputType: UInt64
    public let backendMaximumAbsoluteError, cpuMaximumAbsoluteError: Double
    public let backendRelativeL2Error, cpuRelativeL2Error: Double
    init?(_ value: flo_metrics) {
        guard value.probe_samples > 0 else { return nil }
        samples = value.probe_samples; columns = value.probe_columns
        weightType = value.probe_weight_type; cpuInputType = value.probe_cpu_input_type
        backendMaximumAbsoluteError = value.probe_backend_max; cpuMaximumAbsoluteError = value.probe_cpu_max
        backendRelativeL2Error = sqrt(value.probe_backend_squared / max(value.probe_reference_squared, 1e-30))
        cpuRelativeL2Error = sqrt(value.probe_cpu_squared / max(value.probe_reference_squared, 1e-30))
    }
}

/// Ownership counters are separate from process footprint: a mapped file span
/// is not resident RAM, and device allocation includes other Metal backends.
public struct LESRuntimeMemory: Codable, Sendable, Equatable {
    public let liveContexts, liveMetalBuffers, ownedMetalBytes, mappedMetalBytes: UInt64
    public let deviceAllocatedBytes: UInt64?
    init(_ value: flo_metrics) {
        liveContexts = value.live_contexts; liveMetalBuffers = value.live_metal_buffers
        ownedMetalBytes = value.owned_metal_bytes; mappedMetalBytes = value.mapped_metal_bytes
        deviceAllocatedBytes = value.metal_device_observed ? value.device_allocated_bytes : nil
    }
}

/// Confine to one serial inference executor. Its C runtime enforces a single
/// owner and is symbol-isolated from the existing tokenizer/CPU reference.
public final class LESContext {
    private let artifact: VerifiedGGUFArtifact
    private let control: LESControl
    private let checkpoints: GGUFCheckpointCollector?
    private var handle: OpaquePointer?
    public let vocabularySize: Int
    public let plan: LESMemoryPlan
    public init(artifact: VerifiedGGUFArtifact, prepared: URL, plan: LESMemoryPlan,
                control: LESControl, traceRouting: Bool = false, bypassPageCache: Bool = false,
                syntheticCPUControl: Bool = false, checkpoints: GGUFCheckpointCollector? = nil) throws {
        try artifact.assertUnchanged()
        guard prepared.isFileURL, !control.shouldAbort() else { throw CancellationError() }
        self.artifact = artifact; self.control = control; self.plan = plan; self.checkpoints = checkpoints
        var options = flo_options()
        options.context_tokens = Int32(plan.contextTokens); options.prefill_slots = Int32(plan.prefillSlots)
        options.cache_slots_per_layer = Int32(plan.cacheSlotsPerLayer); options.lookahead_depth = Int32(plan.lookaheadDepth)
        if syntheticCPUControl {
            guard artifact.manifest.fileByteCount <= 32 << 20,
                  artifact.index.metadata["general.name"]?.text == "synthetic-graph-test" else {
                throw GGUFError.unsupported("CPU control requires generated tiny fixture")
            }
        }
        options.io_threads = 4; options.cpu_threads = 4; options.metal = !syntheticCPUControl
        options.bypass_page_cache = bypassPageCache; options.trace_routing = traceRouting
        options.memory_ceiling_bytes = plan.processCeilingBytes
        if let checkpoints {
            options.checkpoint = { stage, layer, values, count, user in
                guard let user else { return false }
                return Unmanaged<GGUFCheckpointCollector>.fromOpaque(user).takeUnretainedValue()
                    .capture(stage: stage, layer: layer, values: values, count: count)
            }
            options.checkpoint_user = Unmanaged.passUnretained(checkpoints).toOpaque()
        }
        let descriptor = try artifact.duplicateForReading(); defer { Darwin.close(descriptor) }
        guard let handle = flo_open(descriptor, prepared.path, options, { user in
            guard let user else { return true }
            return Unmanaged<LESControl>.fromOpaque(user).takeUnretainedValue().shouldAbort()
        }, Unmanaged.passUnretained(control).toOpaque()) else { throw Self.failure() }
        do { try artifact.assertUnchanged() } catch { flo_close(handle); throw error }
        self.handle = handle; vocabularySize = Int(flo_vocabulary_size(handle))
    }
    deinit { close() }
    /// End native ownership at a phase boundary, regardless of ARC lifetime
    /// extension. Idempotent, on the same serial executor as inference.
    public func close() {
        guard let handle else { return }
        self.handle = nil
        flo_close(handle)
    }
    public static var runtimeMemory: LESRuntimeMemory { LESRuntimeMemory(flo_snapshot(nil)) }
    public func logits(tokens: [Int]) throws -> [Float] {
        try artifact.assertUnchanged()
        guard let handle else { throw GGUFError.invalid("LES context is closed") }
        guard !tokens.isEmpty, tokens.count <= 2048, tokens.allSatisfy({ (0..<vocabularySize).contains($0) }),
              !control.shouldAbort() else { throw CancellationError() }
        let ids = tokens.map(Int32.init)
        var output = [Float](repeating: 0, count: vocabularySize)
        let status = ids.withUnsafeBufferPointer { input in
            output.withUnsafeMutableBufferPointer { flo_decode(handle, input.baseAddress, Int32(input.count), $0.baseAddress, Int32($0.count)) }
        }
        guard status == 0 else { throw Self.failure() }
        try artifact.assertUnchanged(); return output
    }
    public func reset() { flo_reset(handle) }
    public func capturePrefix() -> UInt64 { flo_capture_prefix(handle) }
    public func restorePrefix() throws {
        guard let handle, flo_restore_prefix(handle) == 0 else { throw Self.failure() }
    }
    public var metrics: LESMetrics { LESMetrics(flo_snapshot(handle)) }
    public var projectionProbe: LESProjectionProbe? { LESProjectionProbe(flo_snapshot(handle)) }
    public static var revision: String { String(cString: flo_revision()) }
    static func failure() -> Error { GGUFError.invalid(String(cString: flo_last_error())) }
}
