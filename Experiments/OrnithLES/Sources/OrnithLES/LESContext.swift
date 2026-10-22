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
    private var nativeDiagnosticsEnabled = true
    private let journal: LESAccelerationJournal?
    public init(journal: LESAccelerationJournal? = nil) { self.journal = journal }
    fileprivate var hasDiagnosticJournal: Bool { journal != nil }
    public func setNativeDiagnosticsEnabled(_ enabled: Bool) {
        lock.lock(); nativeDiagnosticsEnabled = enabled; lock.unlock()
    }
    public func recordDiagnosticPhase(_ phase: String) throws { try journal?.record(phase) }
    func recordNativePhase(_ phase: Int32) -> Bool {
        lock.lock(); let enabled = nativeDiagnosticsEnabled; lock.unlock()
        return !enabled || journal?.recordNativePhase(phase) != false
    }
    public func cancel() { lock.lock(); cancelled = true; lock.unlock() }
    public var isCancellationRequested: Bool { lock.lock(); defer { lock.unlock() }; return cancelled }
    public var maximumObservedFootprintBytes: UInt64 { lock.lock(); defer { lock.unlock() }; return observedPeak }
    /// Only between isolated benchmark runs, after the previous owner closes.
    public func beginBenchmarkFootprintWindow() { lock.lock(); observedPeak = 0; lock.unlock() }
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
    public let targetTokensEvaluated, dflashBlocks, dflashProposals, dflashAccepted, dflashReplays: UInt64?
    public let dflashFeaturePeakBytes: UInt64?
    public let flashAttentionActive: Bool?
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
        targetTokensEvaluated = value.target_tokens_evaluated; dflashBlocks = value.dflash_blocks
        dflashProposals = value.dflash_proposals; dflashAccepted = value.dflash_accepted
        dflashReplays = value.dflash_replays; dflashFeaturePeakBytes = value.dflash_feature_peak_bytes
        flashAttentionActive = value.flash_attention_active
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
    private var draftArtifact: VerifiedGGUFArtifact?
    private var handle: OpaquePointer?
    public let vocabularySize: Int
    public let plan: LESMemoryPlan
    public init(artifact: VerifiedGGUFArtifact, prepared: URL, plan: LESMemoryPlan,
                control: LESControl, traceRouting: Bool = false, bypassPageCache: Bool = false,
                syntheticCPUControl: Bool = false, checkpoints: GGUFCheckpointCollector? = nil,
                flashAttention: Bool = false, experimentalDFlash: Bool = false) throws {
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
        options.additional_reserve_bytes = plan.experimentReserveBytes ?? 0
        options.flash_attention = flashAttention; options.experimental_dflash = experimentalDFlash
        if control.hasDiagnosticJournal {
            options.diagnostic = { phase, user in
                guard let user else { return false }
                return Unmanaged<LESControl>.fromOpaque(user).takeUnretainedValue().recordNativePhase(phase)
            }
            options.diagnostic_user = Unmanaged.passUnretained(control).toOpaque()
        }
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
        draftArtifact = nil
    }
    public static var runtimeMemory: LESRuntimeMemory { LESRuntimeMemory(flo_snapshot(nil)) }
    public func logits(tokens: [Int]) throws -> [Float] {
        try artifact.assertUnchanged()
        try draftArtifact?.assertUnchanged()
        guard let handle else { throw GGUFError.invalid("LES context is closed") }
        guard !tokens.isEmpty, tokens.count <= 2048, tokens.allSatisfy({ (0..<vocabularySize).contains($0) }),
              !control.shouldAbort() else { throw CancellationError() }
        let ids = tokens.map(Int32.init)
        var output = [Float](repeating: 0, count: vocabularySize)
        let status = ids.withUnsafeBufferPointer { input in
            output.withUnsafeMutableBufferPointer { flo_decode(handle, input.baseAddress, Int32(input.count), $0.baseAddress, Int32($0.count)) }
        }
        guard status == 0 else { throw Self.failure() }
        try artifact.assertUnchanged(); try draftArtifact?.assertUnchanged(); return output
    }
    public func reset() { flo_reset(handle) }
    public func capturePrefix() -> UInt64 { flo_capture_prefix(handle) }
    public func restorePrefix() throws {
        guard let handle, flo_restore_prefix(handle) == 0 else { throw Self.failure() }
    }
    /// The isolated benchmark owns the sole recurrent checkpoint. Never attach
    /// this to a live chat/API context that also keeps conversation snapshots.
    public func attachDraft(_ draft: VerifiedGGUFArtifact) throws {
        try draft.assertUnchanged()
        guard let handle else { throw GGUFError.invalid("LES context is closed") }
        try control.recordDiagnosticPhase("draft.headroom")
        // The first admission ran before allocating the target. Recheck actual
        // headroom now: a model loader may allocate its whole GPU buffer before
        // its first progress/abort callback can stop it.
        try Self.requireDraftHeadroom(ProcessFootprint.current(), reserve: plan.experimentReserveBytes ?? 0,
            ceiling: plan.processCeilingBytes)
        let fd = try draft.duplicateForReading(); defer { Darwin.close(fd) }
        guard flo_attach_dflash(handle, fd) == 0 else { throw Self.failure() }
        try draft.assertUnchanged()
        draftArtifact = draft
    }
    static func requireDraftHeadroom(_ memory: ProcessFootprint?, reserve: UInt64, ceiling: UInt64) throws {
        let margin: UInt64 = 128 << 20
        guard let memory, reserve > 0, reserve <= 1_500_000_000,
              memory.footprintBytes < ceiling, reserve <= ceiling - memory.footprintBytes,
              margin <= ceiling - memory.footprintBytes - reserve,
              memory.availableBytes.map({ $0 >= reserve + margin }) != false else {
            throw GGUFError.limit("DFlash saknar aktuellt minnesutrymme för draft, KV-cache och säkerhetsmarginal. Testet stoppades före draftladdningen; vanlig chatt kan användas.")
        }
    }
    public func dflashStep(logits: inout [Float], maximumTokens: Int) throws -> [Int] {
        try artifact.assertUnchanged(); try draftArtifact?.assertUnchanged()
        guard let handle, logits.count == vocabularySize, (1...4).contains(maximumTokens) else {
            throw GGUFError.invalid("DFlash step dimensions")
        }
        var tokens = [Int32](repeating: 0, count: maximumTokens)
        let count = logits.withUnsafeMutableBufferPointer { rows in
            tokens.withUnsafeMutableBufferPointer { output in
                flo_dflash_step(handle, rows.baseAddress, Int32(rows.count), output.baseAddress, Int32(maximumTokens))
            }
        }
        guard count > 0, count <= maximumTokens else { throw Self.failure() }
        try artifact.assertUnchanged(); try draftArtifact?.assertUnchanged()
        return tokens.prefix(Int(count)).map(Int.init)
    }
    public var metrics: LESMetrics { LESMetrics(flo_snapshot(handle)) }
    public var projectionProbe: LESProjectionProbe? { LESProjectionProbe(flo_snapshot(handle)) }
    public static var revision: String { String(cString: flo_revision()) }
    static func failure() -> Error { GGUFError.invalid(String(cString: flo_last_error())) }
}
