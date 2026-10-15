import Darwin
import Foundation
import GGUFCore
import LlamaTokenizerRuntime
import MinirunKit
import StorageCore

/// Diagnostic reference, never the bounded production runner. Uses unchanged
/// pinned llama.cpp CPU kernels, lazy PLE and mmap without mlock or repacking.
/// mmap may still issue kernel read-ahead: this is not a flash-cache guarantee.
/// Run serially, release it before constructing the MLX decoder, and admit only
/// short reference sequences. A failure leaves full-model parity unproven.
public final class Qwen4ExpCPUReference {
    private final class Governor: @unchecked Sendable {
        enum Stop { case cancelled, memory, unavailable }
        let budget: UInt64
        let cancelled: @Sendable () -> Bool
        let lock = NSLock()
        var reason: Stop?
        var peak: UInt64 = 0
        init(budget: UInt64, cancelled: @escaping @Sendable () -> Bool) {
            self.budget = budget; self.cancelled = cancelled
        }
        func shouldAbort() -> Bool {
            let cancellation = cancelled()
            let footprint = ProcessFootprint.current()
            lock.lock(); defer { lock.unlock() }
            if cancellation { reason = .cancelled }
            guard let footprint else { reason = reason ?? .unavailable; return true }
            peak = max(peak, footprint.footprintBytes)
            let headroom: UInt64 = 512 << 20
            if footprint.footprintBytes >= budget || budget - footprint.footprintBytes < headroom
                || footprint.availableBytes.map({ $0 < headroom }) == true { reason = reason ?? .memory }
            return reason != nil
        }
        func failure(_ fallback: String) -> Error {
            lock.lock(); defer { lock.unlock() }
            switch reason {
            case .cancelled: return CancellationError()
            case .memory: return GGUFError.limit("local CPU reference exceeded memory headroom; parity remains unverified")
            case .unavailable: return GGUFError.limit("local CPU reference footprint unavailable")
            case nil: return GGUFError.invalid(fallback)
            }
        }
        var maximumObservedFootprint: UInt64 { lock.lock(); defer { lock.unlock() }; return peak }
    }
    private let artifact: VerifiedGGUFArtifact
    private let governor: Governor
    private let handle: OpaquePointer
    public let vocabularySize: Int

    public init(artifact: VerifiedGGUFArtifact, contextTokens: Int = 128, budgetBytes: UInt64 = 4_000_000_000,
                cancellationRequested: @escaping @Sendable () -> Bool = { false }) throws {
        guard (16...128).contains(contextTokens), budgetBytes <= 5_500_000_000 else {
            throw GGUFError.limit("short CPU reference context or memory budget")
        }
        _ = try Qwen4ExpDecoder.Layout(index: artifact.index)
        let governor = Governor(budget: budgetBytes, cancelled: cancellationRequested)
        self.artifact = artifact; self.governor = governor
        try artifact.assertUnchanged()
        let descriptor = try artifact.duplicateForReading()
        defer { _ = Darwin.close(descriptor) }
        guard let handle = flr_create(descriptor, Int32(contextTokens), { user in
            guard let user else { return true }
            return Unmanaged<Governor>.fromOpaque(user).takeUnretainedValue().shouldAbort()
        }, Unmanaged.passUnretained(governor).toOpaque()) else {
            throw governor.failure(String(cString: flr_last_error()))
        }
        do { try artifact.assertUnchanged() }
        catch { flr_free(handle); throw error }
        self.handle = handle; vocabularySize = Int(flr_vocabulary_size(handle))
    }

    deinit { flr_free(handle) }
    public var maximumObservedFootprintBytes: UInt64 { governor.maximumObservedFootprint }

    public func logits(for token: Int) throws -> [Float] {
        guard token >= 0, token < vocabularySize else { throw GGUFError.invalid("CPU reference token") }
        try artifact.assertUnchanged()
        var output = [Float](repeating: 0, count: vocabularySize)
        let count = output.withUnsafeMutableBufferPointer {
            flr_step(handle, Int32(token), $0.baseAddress, Int32($0.count))
        }
        guard Int(count) == vocabularySize else { throw governor.failure(String(cString: flr_last_error())) }
        try artifact.assertUnchanged()
        return output
    }
}
