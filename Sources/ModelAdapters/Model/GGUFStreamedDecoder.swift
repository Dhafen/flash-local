import Foundation
import GGUFCore
import MLX
import MLXBridge
import MinirunKit
import StorageCore

/// Closed dispatch over explicitly validated architectures. A model name or
/// its parameter count never selects a different graph implicitly.
public final class GGUFStreamedDecoder {
    private enum Backend { case ream(Qwen4ExpDecoder), qwen35(Qwen35MoEDecoder) }
    private let backend: Backend
    public enum Checkpoint {
        case ream(Qwen4ExpDecoder.Checkpoint)
        case qwen35(Qwen35MoEDecoder.Checkpoint)
        public var payloadBytes: UInt64 {
            switch self { case .ream(let c): c.payloadBytes; case .qwen35(let c): c.payloadBytes }
        }
    }

    public static func vocabularySize(index: GGUFIndex) throws -> Int {
        switch index.metadata["general.architecture"]?.text {
        case "qwen4exp": return try Qwen4ExpDecoder.Layout(index: index).vocabularySize
        case "qwen35moe": return try Qwen35MoEConfig(index: index).vocabularySize
        default: throw GGUFError.unsupported("unregistered streamed GGUF architecture")
        }
    }
    public static func memoryPlan(index: GGUFIndex, contextTokens: Int, budgetBytes: UInt64,
                                  workingReserveBytes: UInt64 = 1_000_000_000) throws -> GGUFDecoderMemoryPlan {
        switch index.metadata["general.architecture"]?.text {
        case "qwen4exp":
            return try GGUFDecoderMemoryPlan(config: Qwen4ExpDecoder.Layout(index: index).config,
                contextTokens: contextTokens, budgetBytes: budgetBytes, reservedWorkingBytes: workingReserveBytes)
        case "qwen35moe":
            return try Qwen35MoEConfig(index: index).memoryPlan(contextTokens: contextTokens,
                budgetBytes: budgetBytes, reservedWorkingBytes: workingReserveBytes)
        default: throw GGUFError.unsupported("unregistered streamed GGUF architecture")
        }
    }
    public init(artifact: VerifiedGGUFArtifact, contextTokens: Int, budgetBytes: UInt64,
                rawCacheBytes: Int = 64 << 20, residentCacheBytes: Int = 0) throws {
        switch artifact.index.metadata["general.architecture"]?.text {
        case "qwen4exp":
            backend = .ream(try Qwen4ExpDecoder(artifact: artifact, contextTokens: contextTokens,
                budgetBytes: budgetBytes, rawCacheBytes: rawCacheBytes, residentCacheBytes: residentCacheBytes))
        case "qwen35moe":
            backend = .qwen35(try Qwen35MoEDecoder(artifact: artifact, contextTokens: contextTokens,
                budgetBytes: budgetBytes, rawCacheBytes: rawCacheBytes, residentCacheBytes: residentCacheBytes))
        default: throw GGUFError.unsupported("unregistered streamed GGUF architecture")
        }
    }
    public var vocabularySize: Int {
        switch backend { case .ream(let d): d.layout.vocabularySize; case .qwen35(let d): d.layout.vocabularySize }
    }
    public var onFootprintObservation: ((ProcessFootprint) -> Void)? {
        get { switch backend { case .ream(let d): d.onFootprintObservation; case .qwen35(let d): d.onFootprintObservation } }
        set { switch backend { case .ream(let d): d.onFootprintObservation = newValue; case .qwen35(let d): d.onFootprintObservation = newValue } }
    }
    public var readStatistics: TileReaderStatistics {
        switch backend { case .ream(let d): d.readStatistics; case .qwen35(let d): d.readStatistics }
    }
    public var rawCacheStatistics: BufferPoolStatistics {
        switch backend { case .ream(let d): d.rawCacheStatistics; case .qwen35(let d): d.rawCacheStatistics }
    }
    public var projectionStatistics: GGUFProjectionStatistics {
        switch backend { case .ream(let d): d.projectionStatistics; case .qwen35(let d): d.projectionStatistics }
    }
    public var residentCacheStatistics: GGUFResidentWeightCacheStatistics {
        switch backend { case .ream(let d): d.residentCacheStatistics; case .qwen35(let d): d.residentCacheStatistics }
    }
    public func reset() { switch backend { case .ream(let d): d.reset(); case .qwen35(let d): d.reset() } }
    public func checkpoint() throws -> Checkpoint? {
        switch backend {
        case .ream(let d): return try d.checkpoint().map(Checkpoint.ream)
        case .qwen35(let d): return try d.checkpoint().map(Checkpoint.qwen35)
        }
    }
    public func restore(_ checkpoint: Checkpoint) throws {
        switch (backend, checkpoint) {
        case (.ream(let d), .ream(let c)): try d.restore(c)
        case (.qwen35(let d), .qwen35(let c)): try d.restore(c)
        default: throw GGUFError.invalid("checkpoint belongs to a different architecture")
        }
    }
    public func consume(_ token: UInt64, cancellationRequested: () -> Bool = { false }) throws {
        switch backend {
        case .ream(let d): try d.consume(token, cancellationRequested: cancellationRequested)
        case .qwen35(let d): try d.consume(token, cancellationRequested: cancellationRequested)
        }
    }
    public func logits(for token: UInt64, cancellationRequested: () -> Bool = { false }) throws -> MLXArray {
        switch backend {
        case .ream(let d): return try d.logits(for: token, cancellationRequested: cancellationRequested)
        case .qwen35(let d): return try d.logits(for: token, cancellationRequested: cancellationRequested)
        }
    }
}
