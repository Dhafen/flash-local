import Foundation
import GGUFCore
import MLX
import MLXBridge
import MinirunKit
import StorageCore

/// Real text trunk for qwen35moe GGUF, with bounded selected-expert reads.
/// Shares quantized projections, FP16 KV and the GDN recurrence with REAM,
/// but uses ordinary residual/norm blocks and SiLU GDN output gating.
public final class Qwen35MoEDecoder {
    public let layout: Qwen35MoEConfig
    public let memoryPlan: GGUFDecoderMemoryPlan
    public private(set) var position = 0
    public var onFootprintObservation: ((ProcessFootprint) -> Void)?
    var footprintSnapshot: () -> ProcessFootprint? = { ProcessFootprint.current() }
    #if DEBUG
    var onParityCheckpoint: ((String, [Float]) -> Void)?
    #endif
    private let weights: GGUFWeightSource
    private var attentionCaches: [Int: Qwen4ExpTextAttention.Cache] = [:]
    private var recurrentStates: [Int: MLXArray] = [:]
    private var convolutionHistories: [Int: MLXArray] = [:]
    private var requiresReset = false
    private let checkpointIdentity = UUID()

    public final class Checkpoint {
        public let payloadBytes: UInt64
        fileprivate let owner: UUID
        fileprivate let position: Int
        fileprivate let attention: [Int: Qwen4ExpTextAttention.Cache.Checkpoint]
        fileprivate let recurrent: [Int: MLXArray]
        fileprivate let convolution: [Int: MLXArray]
        fileprivate init(owner: UUID, position: Int, payloadBytes: UInt64,
                         attention: [Int: Qwen4ExpTextAttention.Cache.Checkpoint],
                         recurrent: [Int: MLXArray], convolution: [Int: MLXArray]) {
            self.owner = owner; self.position = position; self.payloadBytes = payloadBytes
            self.attention = attention; self.recurrent = recurrent; self.convolution = convolution
        }
    }

    public init(artifact: VerifiedGGUFArtifact, contextTokens: Int, budgetBytes: UInt64,
                workingReserveBytes: UInt64 = 1_000_000_000, rawCacheBytes: Int = 64 << 20,
                residentCacheBytes: Int = 0, execution: GGUFLinearExecution = .packedMetal) throws {
        layout = try Qwen35MoEConfig(index: artifact.index)
        memoryPlan = try layout.memoryPlan(contextTokens: contextTokens, budgetBytes: budgetBytes,
            reservedWorkingBytes: workingReserveBytes)
        guard rawCacheBytes >= 8192, residentCacheBytes >= 0,
              UInt64(rawCacheBytes) <= memoryPlan.cacheBytes,
              UInt64(residentCacheBytes) <= memoryPlan.cacheBytes - UInt64(rawCacheBytes) else {
            throw GGUFError.limit("qwen35moe weight cache admission")
        }
        weights = try GGUFWeightSource(artifact: artifact, rawCacheBytes: rawCacheBytes,
            tileBytes: min(4 << 20, rawCacheBytes / 2), execution: execution,
            residentCacheBytes: residentCacheBytes, cacheSelectedExperts: true)
        for layer in 0..<layout.layers where (layer + 1) % layout.attentionInterval == 0 {
            attentionCaches[layer] = try Qwen4ExpTextAttention.Cache(keyValueHeads: layout.keyValueHeads,
                headDimension: layout.keyLength, contextLimit: contextTokens,
                maximumBytes: memoryPlan.kvBytes / UInt64(layout.denseAttentionLayers))
        }
        MLX.Memory.cacheLimit = 32 << 20
    }

    public var readStatistics: TileReaderStatistics { weights.readStatistics }
    public var rawCacheStatistics: BufferPoolStatistics { weights.poolStatistics }
    public var projectionStatistics: GGUFProjectionStatistics { weights.projectionStatistics }
    public var residentCacheStatistics: GGUFResidentWeightCacheStatistics { weights.residentCacheStatistics }

    public func reset() {
        for cache in attentionCaches.values { cache.reset() }
        recurrentStates.removeAll(); convolutionHistories.removeAll()
        position = 0; requiresReset = false
    }

    public func checkpoint(maximumBytes: UInt64 = 256 << 20) throws -> Checkpoint? {
        guard !requiresReset, memoryPlan.reservedWorkingBytes >= 512 << 20,
              maximumBytes <= memoryPlan.reservedWorkingBytes - (512 << 20) else {
            throw GGUFError.limit("qwen35moe prefix checkpoint admission")
        }
        var bytes = UInt64(attentionCaches.values.reduce(0) { $0 + $1.allocatedBytes })
        for tensor in Array(recurrentStates.values) + Array(convolutionHistories.values) { bytes += UInt64(tensor.nbytes) }
        guard bytes <= maximumBytes else { return nil }
        return Checkpoint(owner: checkpointIdentity, position: position, payloadBytes: bytes,
            attention: attentionCaches.mapValues { $0.checkpoint() }, recurrent: recurrentStates,
            convolution: convolutionHistories)
    }

    public func restore(_ checkpoint: Checkpoint) throws {
        guard checkpoint.owner == checkpointIdentity, Set(checkpoint.attention.keys) == Set(attentionCaches.keys) else {
            throw GGUFError.invalid("qwen35moe checkpoint owner/layout")
        }
        try weights.artifact.assertUnchanged()
        for (layer, value) in checkpoint.attention { try attentionCaches[layer]!.restore(value) }
        recurrentStates = checkpoint.recurrent; convolutionHistories = checkpoint.convolution
        position = checkpoint.position; requiresReset = false
    }

    public func logits(for token: UInt64, cancellationRequested: () -> Bool = { false }) throws -> MLXArray {
        try step(token, calculateLogits: true, cancellationRequested: cancellationRequested)!
    }
    public func consume(_ token: UInt64, cancellationRequested: () -> Bool = { false }) throws {
        _ = try step(token, calculateLogits: false, cancellationRequested: cancellationRequested)
    }

    private func check(_ cancelled: () -> Bool) throws {
        if cancelled() { throw CancellationError() }
        guard let footprint = footprintSnapshot() else { throw GGUFError.limit("qwen35moe footprint unavailable") }
        onFootprintObservation?(footprint)
        let headroom: UInt64 = 256 << 20
        guard footprint.footprintBytes < memoryPlan.budgetBytes,
              memoryPlan.budgetBytes - footprint.footprintBytes >= headroom,
              footprint.availableBytes.map({ $0 >= headroom }) != false else {
            throw GGUFError.limit("qwen35moe live process memory admission")
        }
    }

    private func step(_ token: UInt64, calculateLogits: Bool, cancellationRequested: () -> Bool) throws -> MLXArray? {
        guard !requiresReset, position < memoryPlan.contextTokens, token < UInt64(layout.vocabularySize) else {
            throw GGUFError.invalid("qwen35moe token/context/state; reset required after a failed step")
        }
        do {
            try check(cancellationRequested)
            let c = layout
            var hidden = try weights.row(token, tensor: "token_embd.weight").reshaped([1, c.hiddenSize])
            for layer in 0..<c.layers {
                try check(cancellationRequested)
                let p = "blk.\(layer)."
                let normalized = try K3Norm.rms(hidden, weight: weights.smallTensor(p + "attn_norm.weight"), eps: c.epsilon)
                let mixed: MLXArray
                if let cache = attentionCaches[layer] {
                    let attention = try Qwen4ExpTextAttention.step(
                        queryAndGate: weights.linear(normalized, tensor: p + "attn_q.weight"),
                        key: weights.linear(normalized, tensor: p + "attn_k.weight"),
                        value: weights.linear(normalized, tensor: p + "attn_v.weight"),
                        queryNorm: weights.smallTensor(p + "attn_q_norm.weight"),
                        keyNorm: weights.smallTensor(p + "attn_k_norm.weight"), epsilon: c.epsilon,
                        rotaryDimensions: c.rotaryDimensions, ropeBase: c.ropeBase, cache: cache)
                    mixed = try weights.linear(attention, tensor: p + "attn_output.weight")
                } else { mixed = try recurrent(normalized, layer: layer) }
                hidden = hidden + mixed
                let ffn = try K3Norm.rms(hidden, weight: weights.smallTensor(p + "post_attention_norm.weight"), eps: c.epsilon)
                hidden = try hidden + moe(ffn, layer: layer, cancelled: cancellationRequested)
                hidden.eval()
                #if DEBUG
                if hidden.size <= 8192 { onParityCheckpoint?("l_out-\(layer)", hidden.asArray(Float.self)) }
                #endif
            }
            let logits: MLXArray?
            if calculateLogits {
                let normalized = try K3Norm.rms(hidden, weight: weights.smallTensor("output_norm.weight"), eps: c.epsilon)
                #if DEBUG
                if normalized.size <= 8192 { onParityCheckpoint?("result_norm", normalized.asArray(Float.self)) }
                #endif
                let output = try weights.linear(normalized, tensor: c.outputTensorName)
                output.eval()
                guard output.asArray(Float.self).allSatisfy({ $0.isFinite }) else { throw GGUFError.invalid("qwen35moe nonfinite logits") }
                logits = output
            } else { logits = nil }
            try check(cancellationRequested)
            position += 1
            return logits
        } catch { requiresReset = true; throw error }
    }

    private func recurrent(_ input: MLXArray, layer: Int) throws -> MLXArray {
        let c = layout, p = "blk.\(layer).", dim = c.recurrentStateSize
        let keyWidth = c.recurrentKeyHeads * dim, valueWidth = c.recurrentHeads * dim
        let channels = 2 * keyWidth + valueWidth
        let qkv = try weights.linear(input, tensor: p + "attn_qkv.weight").reshaped([channels])
        let history = convolutionHistories[layer] ?? MLXArray.zeros([c.convolutionWidth - 1, channels], dtype: .float32)
        let conv = try Qwen4ExpGDN.convolutionStep(input: qkv,
            kernel: weights.smallTensor(p + "ssm_conv1d.weight").T, history: history)
        let q = K3Norm.l2(conv.output[0..<keyWidth].reshaped([c.recurrentKeyHeads, dim]), eps: c.epsilon)
        let k = K3Norm.l2(conv.output[keyWidth..<(2 * keyWidth)].reshaped([c.recurrentKeyHeads, dim]), eps: c.epsilon)
        let v = conv.output[(2 * keyWidth)...].reshaped([c.recurrentHeads, dim])
        let alpha = try weights.linear(input, tensor: p + "ssm_alpha.weight").reshaped([c.recurrentHeads])
        let decay = try K3Activation.softplus(alpha + weights.smallTensor(p + "ssm_dt.bias")) * weights.smallTensor(p + "ssm_a")
        let beta = try sigmoid(weights.linear(input, tensor: p + "ssm_beta.weight").reshaped([c.recurrentHeads]))
        let initial = recurrentStates[layer] ?? MLXArray.zeros([c.recurrentHeads, dim, dim], dtype: .float32)
        let step = try Qwen4ExpGDN.step(q: q, k: k, v: v, logDecay: decay, beta: beta, state: initial)
        let gate = try weights.linear(input, tensor: p + "attn_gate.weight").reshaped([c.recurrentHeads, dim])
        // Qwen3.5 uses SiLU here; REAM's sigmoid gate must not be reused.
        let gated = try K3Norm.rms(step.output, weight: weights.smallTensor(p + "ssm_norm.weight"), eps: c.epsilon)
            * K3Activation.silu(gate)
        let output = try weights.linear(gated.reshaped([1, valueWidth]), tensor: p + "ssm_out.weight")
        recurrentStates[layer] = step.state; convolutionHistories[layer] = conv.history
        return output
    }

    private func moe(_ input: MLXArray, layer: Int, cancelled: () -> Bool) throws -> MLXArray {
        let c = layout, p = "blk.\(layer)."
        let selected = try Qwen4ExpRouting.select(logits: weights.linear(input, tensor: p + "ffn_gate_inp.weight"), topK: c.expertsPerToken)
        let experts = selected.ids[0].map { UInt64($0) }
        try check(cancelled)
        let gate = try weights.linearExperts(input, tensor: p + "ffn_gate_exps.weight", experts: experts)
        let up = try weights.linearExperts(input, tensor: p + "ffn_up_exps.weight", experts: experts)
        let down = try weights.linearExperts(K3Activation.silu(gate) * up, tensor: p + "ffn_down_exps.weight", experts: experts)
        var routed = MLXArray.zeros([1, c.hiddenSize], dtype: .float32)
        for slot in experts.indices { routed = routed + down[slot..<(slot+1)] * selected.weights[0, slot] }
        routed.eval()
        let sharedGate = try weights.linear(input, tensor: p + "ffn_gate_shexp.weight")
        let sharedUp = try weights.linear(input, tensor: p + "ffn_up_shexp.weight")
        let shared = try weights.linear(K3Activation.silu(sharedGate) * sharedUp, tensor: p + "ffn_down_shexp.weight")
        let scalar = try sigmoid(sum(input * weights.smallTensor(p + "ffn_gate_inp_shexp.weight"), axis: -1, keepDims: true))
        let output = routed + shared * scalar
        output.eval()
        return output
    }
}
