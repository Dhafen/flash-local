import Darwin
import Foundation
import FlashFlint
import GGUFCore
import MinirunKit
import StorageCore

public enum FlintFailure: Error, LocalizedError, Sendable {
    case invalid(String)
    public var errorDescription: String? { switch self { case .invalid(let message): return message } }
}

public struct FlintConfiguration: Equatable, Sendable {
    public enum Backend: Equatable, Sendable { case metal, cpuReference }
    public var backend: Backend = .metal
    public var contextTokens = 8192
    public var evaluationBatch = 128
    public var physicalBatch = 4
    public var cpuThreads = 4
    public var memoryCeilingBytes: UInt64 = 5_500_000_000
    public init() {}
    public func validate() throws {
        guard (32...65536).contains(contextTokens), (1...2048).contains(evaluationBatch),
              (1...16).contains(physicalBatch), physicalBatch <= evaluationBatch,
              (1...8).contains(cpuThreads), memoryCeilingBytes > 0, memoryCeilingBytes <= 5_500_000_000 else {
            throw FlintFailure.invalid("Invalid bounded Flint configuration")
        }
    }
}

public enum FlintArtifact {
    public static let manifest = try! GGUFModelManifest(fileName: "flint-int4-q4_1.gguf",
        fileByteCount: 25_992_966_976,
        sha256: "43c07baccdba42eb7528a9630ee15699b1942c297291dc1b39a061f508ee1a4b")
    public static let sourceURL = URL(string: "https://huggingface.co/RecursiveSI/flint-int4-gguf/resolve/9c47151036bc2c34331142dda04488bceab2bc9f/flint-int4-q4_1.gguf")!
}

public struct FlintProgress: Sendable {
    public let phase: Int
    /// Nil means a real phase with no measurable percentage.
    public let fraction: Double?
    public let observedFootprintBytes: UInt64?
}

public struct FlintMetrics: Sendable {
    /// Counts explicit expert reads, not mmap faults or physical NAND traffic.
    public let expertIOBytes, expertIOCalls, cachedTokens, evaluatedTokens, stateOnlyTokens: UInt64
    public let estimatedProcessBytes, expertBankBytes, estimatedKVBytes: UInt64
    public let forwardSeconds: Double
    public let flashAttentionActive: Bool
    init(_ value: flf_metrics) {
        expertIOBytes = value.io_bytes; expertIOCalls = value.io_calls; cachedTokens = value.cached_tokens
        evaluatedTokens = value.evaluated_tokens; stateOnlyTokens = value.state_only_tokens
        estimatedProcessBytes = value.estimated_process_bytes; expertBankBytes = value.expert_bank_bytes
        estimatedKVBytes = value.estimated_kv_bytes; forwardSeconds = value.forward_seconds
        flashAttentionActive = value.flash_attention_active
    }
}

// Safety invariant: the only cross-executor mutable fields are protected by
// this lock. onProgress is installed before the synchronous native load and
// cleared only after its loader workers return; cancellation never accesses it.
// No native pointer crosses this bridge. Replace the lock with checked atomic
// storage when the project's minimum Swift toolchain supports that migration.
private final class FlintCancellation: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false
    private var ceiling: UInt64 = 0
    var onProgress: (@Sendable (FlintProgress) -> Void)?
    func begin(ceiling: UInt64) { lock.lock(); cancelled = false; self.ceiling = ceiling; lock.unlock() }
    func cancel() { lock.lock(); cancelled = true; lock.unlock() }
    func shouldAbort() -> Bool {
        lock.lock(); let stopped = cancelled, limit = ceiling; lock.unlock()
        guard let memory = ProcessFootprint.current() else { return true }
        return stopped || memory.footprintBytes >= limit - min(limit, 128 << 20) ||
            memory.availableBytes.map({ $0 < 128 << 20 }) == true
    }
}

private final class FlintNativeOwner {
    let pointer: OpaquePointer
    let directory: URL
    // The C callback borrows this object until flf_close completes. Retain it
    // here as well as on the actor, independent of stored-property teardown.
    let control: FlintCancellation
    init(pointer: OpaquePointer, directory: URL, control: FlintCancellation) {
        self.pointer = pointer; self.directory = directory; self.control = control
    }
    deinit { flf_close(pointer); _ = Darwin.rmdir(directory.path) }
}

/// Experimental inference executor, independent of SwiftUI/view lifetime.
/// This is a control/adapter API, not an enabled production chat backend.
/// The app must close its Ornith owner before load (native omni knobs are still
/// process-wide). Do not advertise model quality from successful file loading.
public actor FlintSession {
    private nonisolated let cancellation = FlintCancellation()
    private var native: FlintNativeOwner?
    private var artifact: VerifiedGGUFArtifact?
    private var configuration = FlintConfiguration()
    private var activeCeiling: UInt64 = 5_500_000_000
    private var stateTokens: [Int32] = []
    private var lastLogits: [Float]?

    public init() {}
    public nonisolated func cancel() { cancellation.cancel() }
    private func failure() -> FlintFailure { .invalid(String(cString: flf_last_error())) }
    private func owner() throws -> FlintNativeOwner {
        guard let native else { throw FlintFailure.invalid("Load Flint first") }
        return native
    }

    public func load(artifact: VerifiedGGUFArtifact, privateRoot: URL,
                     configuration: FlintConfiguration = .init(),
                     progress: @escaping @Sendable (FlintProgress) -> Void = { _ in }) throws {
        try configuration.validate()
        try artifact.assertUnchanged()
        guard native == nil else {
            if self.artifact?.manifest == artifact.manifest, self.configuration == configuration {
                try self.artifact?.assertUnchanged()
                return
            }
            throw FlintFailure.invalid("Unload the existing Flint session before changing its model/configuration")
        }
        guard artifact.manifest == FlintArtifact.manifest,
              artifact.index.metadata["general.architecture"]?.text == "flint", privateRoot.isFileURL,
              let memory = ProcessFootprint.current() else {
            throw FlintFailure.invalid("Expected the verified official Flint INT4 GGUF and a private session directory")
        }
        let limit = min(configuration.memoryCeilingBytes, memory.availableBytes.map { memory.footprintBytes + $0 } ?? configuration.memoryCeilingBytes)
        cancellation.begin(ceiling: limit)
        cancellation.onProgress = progress
        defer { cancellation.onProgress = nil }
        let fm = FileManager.default
        try fm.createDirectory(at: privateRoot, withIntermediateDirectories: true)
        let root = privateRoot.resolvingSymlinksInPath().standardizedFileURL
        let directory = root.appendingPathComponent("flint-session-" + UUID().uuidString, isDirectory: true)
        try fm.createDirectory(at: directory, withIntermediateDirectories: false)
        var directoryAdopted = false
        defer { if !directoryAdopted { _ = Darwin.rmdir(directory.path) } }
        let fd = try artifact.duplicateForReading(); defer { Darwin.close(fd) }
        var options = flf_options()
        options.context_tokens = Int32(configuration.contextTokens)
        options.evaluation_batch = Int32(configuration.evaluationBatch)
        options.physical_batch = Int32(configuration.physicalBatch)
        options.cpu_threads = Int32(configuration.cpuThreads)
        options.process_ceiling_bytes = limit
        options.app_reserve_bytes = max(memory.footprintBytes, 400_000_000)
        options.metal = configuration.backend == .metal
        options.f32_kv_control = configuration.backend == .cpuReference
        let control = Unmanaged.passUnretained(cancellation).toOpaque()
        let pointer = flf_open(fd, directory.path, options, { opaque in
            guard let opaque else { return true }
            return Unmanaged<FlintCancellation>.fromOpaque(opaque).takeUnretainedValue().shouldAbort()
        }, { phase, fraction, opaque in
            guard let opaque else { return }
            let control = Unmanaged<FlintCancellation>.fromOpaque(opaque).takeUnretainedValue()
            control.onProgress?(.init(phase: Int(phase), fraction: fraction >= 0 ? Double(fraction) : nil,
                observedFootprintBytes: ProcessFootprint.current()?.footprintBytes))
        }, control)
        guard let pointer else { _ = Darwin.rmdir(directory.path); throw failure() }
        let loaded = FlintNativeOwner(pointer: pointer, directory: directory, control: cancellation)
        try artifact.assertUnchanged()
        self.native = loaded; self.artifact = artifact; self.configuration = configuration
        activeCeiling = limit; directoryAdopted = true
        stateTokens.removeAll(); lastLogits = nil
    }

    public func unload() { cancellation.cancel(); native = nil; artifact = nil; stateTokens.removeAll(); lastLogits = nil }
    public func resetConversation() throws {
        let owner = try self.owner()
        guard flf_reset(owner.pointer) == 0 else { throw failure() }
        cancellation.begin(ceiling: activeCeiling)
        stateTokens.removeAll(); lastLogits = nil
    }
    public func metrics() throws -> FlintMetrics { .init(flf_snapshot(try owner().pointer)) }

    public func renderPrompt(requestJSON: Data) throws -> String {
        guard !requestJSON.isEmpty, requestJSON.count <= 2 << 20 else { throw FlintFailure.invalid("Flint request exceeds 2 MiB") }
        return try requestJSON.withUnsafeBytes { raw in
            let input = raw.bindMemory(to: CChar.self).baseAddress!
            let size = flf_render_prompt(input, Int32(raw.count), nil, 0)
            guard size < 0 else { throw failure() }
            var output = [CChar](repeating: 0, count: Int(-size))
            guard flf_render_prompt(input, Int32(raw.count), &output, Int32(output.count)) == -size else { throw failure() }
            return String(decoding: output.map { UInt8(bitPattern: $0) }, as: UTF8.self)
        }
    }
    public func tokenize(_ prompt: String) throws -> [Int32] {
        let owner = try self.owner(), utf8 = Array(prompt.utf8CString)
        guard utf8.count <= (2 << 20) + 1 else { throw FlintFailure.invalid("Flint prompt exceeds 2 MiB") }
        return try utf8.withUnsafeBufferPointer { buffer in
            let size = flf_tokenize(owner.pointer, buffer.baseAddress!, Int32(buffer.count - 1), nil, 0)
            if size == 0, prompt.isEmpty { return [] }
            guard size < 0 else { throw failure() }
            var output = [Int32](repeating: 0, count: Int(-size))
            guard flf_tokenize(owner.pointer, buffer.baseAddress!, Int32(buffer.count - 1), &output, Int32(output.count)) == -size else { throw failure() }
            return output
        }
    }
    /// Append/evaluate exact IDs, keeping one bounded KV session. No rollback.
    public func evaluate(tokens: [Int32], outputLogits: Bool) throws -> [Float]? {
        let owner = try self.owner()
        guard !tokens.isEmpty, tokens.count <= configuration.evaluationBatch else { throw FlintFailure.invalid("Invalid Flint evaluation batch") }
        try artifact?.assertUnchanged()
        if Task.isCancelled { cancellation.cancel() }
        var output = outputLogits ? [Float](repeating: 0, count: Int(flf_vocab_size(owner.pointer))) : []
        let status = tokens.withUnsafeBufferPointer { ids in
            output.withUnsafeMutableBufferPointer { logits in
                flf_evaluate(owner.pointer, ids.baseAddress, Int32(ids.count), outputLogits ? logits.baseAddress : nil, Int32(logits.count))
            }
        }
        guard status == 0 else {
            let error = failure()
            _ = flf_reset(owner.pointer)
            stateTokens.removeAll(); lastLogits = nil
            throw error
        }
        try artifact?.assertUnchanged()
        stateTokens.append(contentsOf: tokens); lastLogits = outputLogits ? output : nil
        return lastLogits
    }
    /// Reuses only a complete, unchanged cached prefix; shortened/changed
    /// templates reset instead of attempting an invalid SWA-cache rewind.
    public func preparePrompt(tokens: [Int32], progress: @Sendable (Int, Int) -> Void = { _, _ in }) throws -> (logits: [Float], reusedTokens: Int) {
        guard !tokens.isEmpty, tokens.count < configuration.contextTokens else { throw FlintFailure.invalid("Flint prompt leaves no output capacity") }
        cancellation.begin(ceiling: activeCeiling)
        if try metrics().cachedTokens != UInt64(stateTokens.count) { try resetConversation() }
        if FlintPrefixPolicy.requiresReset(cached: stateTokens, prompt: tokens) ||
            (tokens == stateTokens && lastLogits == nil) { try resetConversation() }
        let reused = stateTokens.count
        var first = reused
        while first < tokens.count {
            let end = min(first + configuration.evaluationBatch, tokens.count)
            _ = try evaluate(tokens: Array(tokens[first..<end]), outputLogits: end == tokens.count)
            progress(end, tokens.count); first = end
        }
        guard let lastLogits else { throw FlintFailure.invalid("Flint prefill produced no logits") }
        return (lastLogits, reused)
    }
}
