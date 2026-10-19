import Foundation
import GGUFCore
import ModelAdapters
import StorageCore

public struct LESMemoryPlan: Codable, Sendable, Equatable {
    public let contextTokens: Int
    public let processCeilingBytes: UInt64
    public let baselineFootprintBytes: UInt64
    public let baselineAvailableBytes: UInt64?
    public let denseBytes: UInt64
    public let kvBytes: UInt64
    public let ssmBytes: UInt64
    public let runtimeReserveBytes: UInt64
    public let prefillSlots: Int
    public let cacheSlotsPerLayer: Int
    public let lookaheadDepth: Int
    public let requestedLookaheadDepth: Int
    public let expertBytesPerSlot: UInt64
    public let prefillAllocationBytes: UInt64
    public let expertAllocationBytes: UInt64
    public let admittedProcessBytes: UInt64

    public init(index: GGUFIndex, contextTokens: Int, footprint: ProcessFootprint,
                lookaheadDepth: Int = 4, pageCacheOnly: Bool = false,
                preferredExpertBytes: UInt64 = 2_000_000_000, physicalBatchTokens: Int = 4) throws {
        let config = try Qwen35MoEConfig(index: index)
        guard (32...65536).contains(contextTokens), contextTokens <= config.contextLimit,
              (0...16).contains(lookaheadDepth), config.expertsPerToken <= 8,
              [1, 4, 8, 16].contains(physicalBatchTokens) else {
            throw GGUFError.unsupported("LES context or scheduling geometry")
        }
        let ceiling: UInt64 = 5_500_000_000
        let physicalLimit = footprint.availableBytes.map {
            footprint.footprintBytes.addingReportingOverflow($0).overflow ? ceiling : footprint.footprintBytes + $0
        } ?? ceiling
        processCeilingBytes = min(ceiling, physicalLimit)
        baselineFootprintBytes = footprint.footprintBytes
        baselineAvailableBytes = footprint.availableBytes
        self.contextTokens = contextTokens
        requestedLookaheadDepth = pageCacheOnly ? 0 : min(lookaheadDepth, config.experts)
        // A microbatch may choose eight different experts for every token.
        // Budget the complete union before increasing graph dimensions.
        prefillSlots = physicalBatchTokens * 8
        var groups: [String: (bytes: UInt64, layers: UInt64)] = [:]
        var dense = UInt64(index.headerByteCount + 65536)
        for tensor in index.tensors.values {
            let parts = tensor.name.split(separator: ".")
            let routed = parts.count == 4 && parts[0] == "blk" &&
                ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_up_exps"].contains(String(parts[2]))
            if routed {
                guard tensor.dimensions.count == 3, tensor.dimensions[2] == UInt64(config.experts),
                      tensor.byteCount % UInt64(config.experts) == 0 else { throw GGUFError.invalid("LES expert axis") }
                let bytes = tensor.byteCount / UInt64(config.experts)
                let key = "\(parts[2]):\(bytes)"
                let old = groups[key]; groups[key] = (bytes, (old?.layers ?? 0) + 1)
            } else { dense += tensor.byteCount + 32 }
        }
        guard !groups.isEmpty else { throw GGUFError.invalid("LES routed weights missing") }
        denseBytes = dense
        kvBytes = UInt64(config.denseAttentionLayers * config.keyValueHeads *
            (config.keyLength + config.valueLength) * contextTokens * 2)
        ssmBytes = UInt64(config.recurrentLayers * config.recurrentHeads *
            config.recurrentStateSize * config.recurrentStateSize * 4) + (4 << 20)
        // Includes the native runtime's 1 GiB reservation plus a stop margin.
        runtimeReserveBytes = (1 << 30) + (128 << 20)
        let fixed = baselineFootprintBytes + denseBytes + kvBytes + ssmBytes + runtimeReserveBytes
        let staging = groups.values.reduce(UInt64(0)) { $0 + $1.bytes * UInt64(prefillSlots) }
        let perSlot = groups.values.reduce(UInt64(0)) { $0 + $1.bytes * $1.layers }
        prefillAllocationBytes = staging; expertBytesPerSlot = perSlot
        let detail = "baseline=\(footprint.footprintBytes) available=\(footprint.availableBytes.map { String($0) } ?? "unknown") ceiling=\(processCeilingBytes) dense=\(denseBytes) KV=\(kvBytes) SSM=\(ssmBytes) reserve=\(runtimeReserveBytes) staging=\(staging) bytesPerSlot=\(perSlot)"
        guard fixed < processCeilingBytes, staging < processCeilingBytes - fixed, perSlot > 0 else {
            throw GGUFError.limit("LES fixed weights/KV/SSM/runtime exceed available process memory; \(detail)")
        }
        let available = processCeilingBytes - fixed - staging
        let totalWidth = Int(min(256, min(available, preferredExpertBytes) / perSlot))
        // Prediction is optional I/O work. Real top-k routing always retains
        // its full slot allowance; a narrow window reduces prediction only.
        self.lookaheadDepth = pageCacheOnly ? 0 : min(requestedLookaheadDepth, max(0, totalWidth - config.expertsPerToken))
        cacheSlotsPerLayer = pageCacheOnly ? 0 : totalWidth - self.lookaheadDepth
        guard pageCacheOnly || cacheSlotsPerLayer >= config.expertsPerToken else {
            throw GGUFError.limit("LES expert window cannot hold all \(config.expertsPerToken) selected experts; window=\(totalWidth) requestedPrediction=\(requestedLookaheadDepth); \(detail)")
        }
        expertAllocationBytes = staging + (pageCacheOnly ? 0 : perSlot * UInt64(totalWidth))
        admittedProcessBytes = fixed + expertAllocationBytes
    }
}

/// Phase-boundary evidence also survives a failed admission. It contains no
/// prompt, tokens, logits or filesystem paths. Reclaim never touches live state.
public struct LESMemoryAdmission: Codable, Sendable {
    public let phase: String
    public let beforeReclaim: ProcessFootprint?
    public let afterReclaim: ProcessFootprint?
    public let plan: LESMemoryPlan?
    public let failure: String?
    public var nativeMemory: LESRuntimeMemory? = nil
}
