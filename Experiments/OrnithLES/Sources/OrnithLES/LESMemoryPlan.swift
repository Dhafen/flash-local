import Foundation
import GGUFCore
import ModelAdapters
import StorageCore

public struct LESMemoryPlan: Codable, Sendable, Equatable {
    public let contextTokens: Int
    public let processCeilingBytes: UInt64
    public let baselineFootprintBytes: UInt64
    public let denseBytes: UInt64
    public let kvBytes: UInt64
    public let ssmBytes: UInt64
    public let runtimeReserveBytes: UInt64
    public let prefillSlots: Int
    public let cacheSlotsPerLayer: Int
    public let lookaheadDepth: Int
    public let expertAllocationBytes: UInt64
    public let admittedProcessBytes: UInt64

    public init(index: GGUFIndex, contextTokens: Int, footprint: ProcessFootprint,
                lookaheadDepth: Int = 4, pageCacheOnly: Bool = false,
                preferredExpertBytes: UInt64 = 2_000_000_000) throws {
        let config = try Qwen35MoEConfig(index: index)
        guard (32...65536).contains(contextTokens), contextTokens <= config.contextLimit,
              (0...16).contains(lookaheadDepth), config.expertsPerToken <= 8 else {
            throw GGUFError.unsupported("LES context or scheduling geometry")
        }
        let ceiling: UInt64 = 5_500_000_000
        let physicalLimit = footprint.availableBytes.map {
            footprint.footprintBytes.addingReportingOverflow($0).overflow ? ceiling : footprint.footprintBytes + $0
        } ?? ceiling
        processCeilingBytes = min(ceiling, physicalLimit)
        baselineFootprintBytes = footprint.footprintBytes
        self.contextTokens = contextTokens; self.lookaheadDepth = pageCacheOnly ? 0 : lookaheadDepth
        prefillSlots = 32
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
        let staging = groups.values.reduce(UInt64(0)) { $0 + $1.bytes * 32 }
        let perSlot = groups.values.reduce(UInt64(0)) { $0 + $1.bytes * $1.layers }
        guard fixed < processCeilingBytes, staging < processCeilingBytes - fixed, perSlot > 0 else {
            throw GGUFError.limit("LES fixed weights/KV/SSM/runtime exceed available process memory")
        }
        let available = processCeilingBytes - fixed - staging
        let totalWidth = Int(min(256, min(available, preferredExpertBytes) / perSlot))
        cacheSlotsPerLayer = pageCacheOnly ? 0 : totalWidth - self.lookaheadDepth
        guard pageCacheOnly || cacheSlotsPerLayer >= config.expertsPerToken else {
            throw GGUFError.limit("LES expert window cannot hold all selected experts plus prefetch")
        }
        expertAllocationBytes = staging + (pageCacheOnly ? 0 : perSlot * UInt64(totalWidth))
        admittedProcessBytes = fixed + expertAllocationBytes
    }
}
