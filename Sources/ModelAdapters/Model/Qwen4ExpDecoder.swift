import Foundation
import GGUFCore
import MLX
import MinirunKit
import StorageCore

/// Single-sequence, one-token-at-a-time decoder for the published dense-QSA
/// metadata variant. No tokenizer, sampling or network operations live here.
/// Full-model correctness is gated separately against pinned llama.cpp.
public final class Qwen4ExpDecoder {
    public struct Layout: Sendable {
        public let config: Qwen4ExpConfig
        public let attentionHeads: Int
        public let rotaryDimensions: Int
        public let ropeBase: Float
        public let epsilon: Float
        public let recurrentKeyHeads: Int
        public let convolutionWidth: Int
        public let pleConvolutionWidth: Int
        public let vocabularySize: Int

        public init(index: GGUFIndex) throws {
            config = try Qwen4ExpConfig(index: index)
            guard !config.usesSparseAttention else { throw GGUFError.unsupported("QSA decoder not yet validated") }
            func integer(_ suffix: String) throws -> Int {
                guard let value = index.metadata["qwen4exp." + suffix]?.uint,
                      value > 0, value <= 65536 else { throw GGUFError.invalid("runtime metadata \(suffix)") }
                return Int(value)
            }
            func number(_ suffix: String) throws -> Float {
                guard case .number(let value)? = index.metadata["qwen4exp." + suffix],
                      value.isFinite, value > 0, value <= Double(Float.greatestFiniteMagnitude) else {
                    throw GGUFError.invalid("runtime float \(suffix)")
                }
                return Float(value)
            }
            attentionHeads = try integer("attention.head_count")
            rotaryDimensions = try integer("rope.dimension_count")
            ropeBase = try number("rope.freq_base")
            epsilon = try number("attention.layer_norm_rms_epsilon")
            recurrentKeyHeads = try integer("ssm.group_count")
            convolutionWidth = try integer("ssm.conv_kernel")
            pleConvolutionWidth = try integer("ple.conv_kernel")
            guard config.layers <= 256, config.hiddenSize <= 65536, config.hyperConnections <= 16,
                  config.pleHeadDimension <= 65536,
                  config.experts <= 65536, config.expertsPerToken <= 64,
                  attentionHeads <= 256, attentionHeads % config.keyValueHeads == 0,
                  config.keyLength == config.valueLength, config.keyLength <= 1024,
                  rotaryDimensions <= config.keyLength, rotaryDimensions % 2 == 0,
                  config.recurrentHeads <= 256, config.recurrentStateSize <= 512,
                  config.recurrentHeads % recurrentKeyHeads == 0,
                  (2...64).contains(convolutionWidth), (2...64).contains(pleConvolutionWidth) else {
                throw GGUFError.unsupported("runtime architecture dimensions")
            }
            guard let embedding = index.tensors["token_embd.weight"], embedding.dimensions.count == 2,
                  embedding.dimensions[0] == UInt64(config.hiddenSize),
                  let vocabulary = Int(exactly: embedding.dimensions[1]), vocabulary <= 1_000_000 else {
                throw GGUFError.invalid("token embedding geometry")
            }
            vocabularySize = vocabulary
            let hidden = config.hiddenSize, hc = config.hyperConnections, wide = hidden * hc
            let stateDim = config.recurrentStateSize, valueHeads = config.recurrentHeads
            let channels = (2 * recurrentKeyHeads + valueHeads) * stateDim
            guard wide * pleConvolutionWidth <= 65536,
                  config.pleHeadDimension * config.pleOffsets.count <= 2 * 1024 * 1024,
                  channels * convolutionWidth <= 65536,
                  try integer("ssm.inner_size") == valueHeads * stateDim else {
                throw GGUFError.unsupported("GDN kernel/inner geometry")
            }
            let lowRank = try integer("hyper_connection.low_rank")
            let expertWidth = try integer("expert_feed_forward_length")
            let sharedWidth = try integer("expert_shared_feed_forward_length")
            func require(_ name: String, _ dimensions: [Int]) throws {
                guard index.tensors[name]?.dimensions == dimensions.map({ UInt64($0) }) else {
                    throw GGUFError.invalid("missing/wrong tensor geometry: \(name)")
                }
            }
            func hcTensors(_ prefix: String, inject: Bool) throws {
                try require(prefix + "norm.weight", [wide])
                try require(prefix + "down.weight", [wide, lowRank])
                try require(prefix + "up.weight", [lowRank, wide])
                if inject { try require(prefix + "inject.weight", [wide, hc]) }
            }
            try require("output.weight", [hidden, vocabulary])
            try hcTensors("output_hc_", inject: false)
            for layer in 0..<config.layers {
                let p = "blk.\(layer)."
                try hcTensors(p + "hc_attn_", inject: true)
                try hcTensors(p + "hc_ffn_", inject: true)
                for projection in ["gate", "up"] {
                    try require(p + "ffn_\(projection)_exps.weight", [hidden, expertWidth, config.experts])
                    try require(p + "ffn_\(projection)_shexp.weight", [hidden, sharedWidth])
                }
                try require(p + "ffn_down_exps.weight", [expertWidth, hidden, config.experts])
                try require(p + "ffn_down_shexp.weight", [sharedWidth, hidden])
                try require(p + "ffn_gate_inp.weight", [hidden, config.experts])
                try require(p + "ffn_gate_inp_shexp.weight", [hidden])
                if (layer + 1) % config.attentionInterval == 0 {
                    try require(p + "attn_q.weight", [hidden, 2 * attentionHeads * config.keyLength])
                    try require(p + "attn_k.weight", [hidden, config.keyValueHeads * config.keyLength])
                    try require(p + "attn_v.weight", [hidden, config.keyValueHeads * config.valueLength])
                    try require(p + "attn_output.weight", [attentionHeads * config.valueLength, hidden])
                    try require(p + "attn_q_norm.weight", [config.keyLength])
                    try require(p + "attn_k_norm.weight", [config.keyLength])
                } else {
                    try require(p + "attn_qkv.weight", [hidden, channels])
                    try require(p + "attn_gate.weight", [hidden, valueHeads * stateDim])
                    for name in ["alpha", "beta"] { try require(p + "ssm_\(name).weight", [hidden, valueHeads]) }
                    try require(p + "ssm_conv1d.weight", [convolutionWidth, channels])
                    try require(p + "ssm_a", [valueHeads])
                    try require(p + "ssm_dt.bias", [valueHeads])
                    try require(p + "ssm_norm.weight", [stateDim])
                    try require(p + "ssm_out.weight", [valueHeads * stateDim, hidden])
                }
                if config.pleLayers.contains(UInt64(layer)) {
                    let input = config.pleHeadDimension * config.pleOffsets.count
                    try require(p + "ple_key.weight", [input, wide])
                    try require(p + "ple_value.weight", [input, hidden])
                    for norm in ["key", "query", "conv"] { try require(p + "ple_norm_\(norm).weight", [wide]) }
                    try require(p + "ple_conv1d.weight", [pleConvolutionWidth, wide])
                }
            }
        }
    }

    public let layout: Layout
    public let memoryPlan: Qwen4ExpMemoryPlan
    public private(set) var position = 0
    /// Existing budget checkpoints also provide run-local memory observations.
    /// Invoked on the decoder's serial executor, never on an I/O worker.
    public var onFootprintObservation: ((ProcessFootprint) -> Void)?
    private let weights: GGUFWeightSource
    private var attentionCaches: [Int: Qwen4ExpTextAttention.Cache] = [:]
    private var recurrentStates: [Int: MLXArray] = [:]
    private var convolutionHistories: [Int: MLXArray] = [:]
    private var pleHistories: [Int: MLXArray] = [:]
    private var previousTokens: [UInt64] = []
    private var requiresReset = false

    public init(artifact: VerifiedGGUFArtifact, contextTokens: Int,
                budgetBytes: UInt64, workingReserveBytes: UInt64 = 1_000_000_000,
                rawCacheBytes: Int = 64 << 20) throws {
        layout = try Layout(index: artifact.index)
        memoryPlan = try Qwen4ExpMemoryPlan(config: layout.config, contextTokens: contextTokens,
            budgetBytes: budgetBytes, reservedWorkingBytes: workingReserveBytes)
        guard rawCacheBytes >= 0, UInt64(rawCacheBytes) <= memoryPlan.cacheBytes else {
            throw GGUFError.limit("raw cache exceeds the admitted run")
        }
        weights = try GGUFWeightSource(artifact: artifact, rawCacheBytes: rawCacheBytes)
        for layer in 0..<layout.config.layers where (layer + 1) % layout.config.attentionInterval == 0 {
            attentionCaches[layer] = try Qwen4ExpTextAttention.Cache(keyValueHeads: layout.config.keyValueHeads,
                headDimension: layout.config.keyLength, contextLimit: contextTokens,
                maximumBytes: memoryPlan.kvBytes / UInt64(layout.config.denseAttentionLayers))
        }
        // This decoder is confined to the app's single inference executor.
        // MLX's allocator cache is process-wide and must have an explicit cap.
        MLX.Memory.cacheLimit = 32 << 20
    }

    public var readStatistics: TileReaderStatistics { weights.readStatistics }
    public var rawCacheStatistics: BufferPoolStatistics { weights.poolStatistics }
    public var projectionStatistics: GGUFProjectionStatistics { weights.projectionStatistics }

    public func reset() {
        for cache in attentionCaches.values { cache.reset() }
        recurrentStates.removeAll(); convolutionHistories.removeAll(); pleHistories.removeAll()
        previousTokens.removeAll(keepingCapacity: true); position = 0; requiresReset = false
    }

    public func logits(for token: UInt64, cancellationRequested: () -> Bool = { false }) throws -> MLXArray {
        guard !requiresReset, position < memoryPlan.contextTokens, token < UInt64(layout.vocabularySize) else {
            throw GGUFError.invalid("decoder token/context/state; reset required after a failed step")
        }
        do {
            let c = layout.config
            var residual = broadcast(try weights.row(token, tensor: "token_embd.weight")
                .reshaped([1, 1, c.hiddenSize]), to: [1, c.hyperConnections, c.hiddenSize])
            residual.eval()
            for layer in 0..<c.layers {
                try checkBudgetAndCancellation(cancellationRequested)
                let p = "blk.\(layer)."
                if c.pleLayers.contains(UInt64(layer)) { residual = try ple(residual, token: token, layer: layer) }
                let mixed = try hc(residual, prefix: p + "hc_attn_", injection: true)
                let tokenMix: MLXArray
                if let cache = attentionCaches[layer] {
                    tokenMix = try Qwen4ExpTextAttention.step(
                        queryAndGate: weights.linear(mixed.mixed, tensor: p + "attn_q.weight"),
                        key: weights.linear(mixed.mixed, tensor: p + "attn_k.weight"),
                        value: weights.linear(mixed.mixed, tensor: p + "attn_v.weight"),
                        queryNorm: weights.smallTensor(p + "attn_q_norm.weight"),
                        keyNorm: weights.smallTensor(p + "attn_k_norm.weight"), epsilon: layout.epsilon,
                        rotaryDimensions: layout.rotaryDimensions, ropeBase: layout.ropeBase, cache: cache)
                    residual = try Qwen4ExpHyperConnections.combine(residual: residual,
                        blockOutput: weights.linear(tokenMix, tensor: p + "attn_output.weight"), injection: mixed.injection!)
                } else {
                    tokenMix = try recurrent(mixed.mixed, layer: layer)
                    residual = try Qwen4ExpHyperConnections.combine(residual: residual,
                        blockOutput: tokenMix, injection: mixed.injection!)
                }
                let ffn = try hc(residual, prefix: p + "hc_ffn_", injection: true)
                residual = try Qwen4ExpHyperConnections.combine(residual: residual,
                    blockOutput: moe(ffn.mixed, layer: layer, cancellationRequested: cancellationRequested),
                    injection: ffn.injection!)
                residual.eval()
            }
            let final = try hc(residual, prefix: "output_hc_", injection: false)
            let logits = try weights.linear(final.mixed, tensor: "output.weight")
            logits.eval()
            try checkBudgetAndCancellation(cancellationRequested)
            guard logits.asArray(Float.self).allSatisfy({ $0.isFinite }) else {
                throw GGUFError.invalid("nonfinite model logits")
            }
            previousTokens.append(token)
            if previousTokens.count > c.pleMultipliers.count - 1 { previousTokens.removeFirst() }
            position += 1
            return logits
        } catch {
            requiresReset = true
            throw error
        }
    }

    private func checkBudgetAndCancellation(_ cancellationRequested: () -> Bool) throws {
        if cancellationRequested() { throw CancellationError() }
        if let current = ProcessFootprint.current() {
            onFootprintObservation?(current)
            let headroom: UInt64 = 256 << 20
            guard current.footprintBytes < memoryPlan.budgetBytes,
                  memoryPlan.budgetBytes - current.footprintBytes >= headroom,
                  current.availableBytes == nil || current.availableBytes! >= headroom else {
                throw GGUFError.limit("live process memory headroom")
            }
        }
    }

    private func hc(_ residual: MLXArray, prefix: String, injection: Bool) throws
        -> (mixed: MLXArray, injection: MLXArray?) {
        let c = layout.config
        let gamma = try weights.smallTensor(prefix + "norm.weight").reshaped([c.hyperConnections, c.hiddenSize])
        let scatter: ((MLXArray) throws -> MLXArray)? = injection
            ? { try self.weights.linear($0, tensor: prefix + "inject.weight") } : nil
        return try Qwen4ExpHyperConnections.mix(residual, gamma: gamma, epsilon: layout.epsilon,
            down: { try self.weights.linear($0, tensor: prefix + "down.weight") },
            up: { try self.weights.linear($0, tensor: prefix + "up.weight") }, injection: scatter)
    }

    private func recurrent(_ input: MLXArray, layer: Int) throws -> MLXArray {
        let c = layout.config, p = "blk.\(layer).", dim = c.recurrentStateSize
        let keyWidth = layout.recurrentKeyHeads * dim
        let valueWidth = c.recurrentHeads * dim, channels = 2 * keyWidth + valueWidth
        let qkv = try weights.linear(input, tensor: p + "attn_qkv.weight").reshaped([channels])
        let history = convolutionHistories[layer] ?? MLXArray.zeros([layout.convolutionWidth - 1, channels], dtype: .float32)
        let conv = try Qwen4ExpGDN.convolutionStep(input: qkv,
            kernel: weights.smallTensor(p + "ssm_conv1d.weight").T, history: history)
        let q = K3Norm.l2(conv.output[0..<keyWidth].reshaped([layout.recurrentKeyHeads, dim]), eps: layout.epsilon)
        let k = K3Norm.l2(conv.output[keyWidth..<(2 * keyWidth)].reshaped([layout.recurrentKeyHeads, dim]), eps: layout.epsilon)
        let v = conv.output[(2 * keyWidth)...].reshaped([c.recurrentHeads, dim])
        let alpha = try weights.linear(input, tensor: p + "ssm_alpha.weight").reshaped([c.recurrentHeads])
        // ssm_a already stores -exp(A_log) in GGUF; do not exponentiate again.
        let decay = try K3Activation.softplus(alpha + weights.smallTensor(p + "ssm_dt.bias"))
            * weights.smallTensor(p + "ssm_a")
        let beta = try sigmoid(weights.linear(input, tensor: p + "ssm_beta.weight").reshaped([c.recurrentHeads]))
        let initial = recurrentStates[layer] ?? MLXArray.zeros([c.recurrentHeads, dim, dim], dtype: .float32)
        let step = try Qwen4ExpGDN.step(q: q, k: k, v: v, logDecay: decay, beta: beta, state: initial)
        let gated = try Qwen4ExpGDN.gatedNorm(step.output,
            gate: weights.linear(input, tensor: p + "attn_gate.weight").reshaped([c.recurrentHeads, dim]),
            weight: weights.smallTensor(p + "ssm_norm.weight"), epsilon: layout.epsilon)
        let output = try weights.linear(gated.reshaped([1, valueWidth]), tensor: p + "ssm_out.weight")
        recurrentStates[layer] = step.state; convolutionHistories[layer] = conv.history
        return output
    }

    private func moe(_ input: MLXArray, layer: Int, cancellationRequested: () -> Bool) throws -> MLXArray {
        let c = layout.config, p = "blk.\(layer)."
        let selected = try Qwen4ExpRouting.select(logits: weights.linear(input, tensor: p + "ffn_gate_inp.weight"),
                                                   topK: c.expertsPerToken)
        var routed = MLXArray.zeros([1, c.hiddenSize], dtype: .float32)
        for (slot, expert) in selected.ids[0].enumerated() {
            try checkBudgetAndCancellation(cancellationRequested)
            let gate = try weights.linear(input, tensor: p + "ffn_gate_exps.weight", expert: UInt64(expert))
            let up = try weights.linear(input, tensor: p + "ffn_up_exps.weight", expert: UInt64(expert))
            let down = try weights.linear(K3Activation.silu(gate) * up, tensor: p + "ffn_down_exps.weight", expert: UInt64(expert))
            routed = routed + down * selected.weights[0, slot]
            routed.eval()
        }
        let sharedGate = try weights.linear(input, tensor: p + "ffn_gate_shexp.weight")
        let sharedUp = try weights.linear(input, tensor: p + "ffn_up_shexp.weight")
        let shared = try weights.linear(K3Activation.silu(sharedGate) * sharedUp, tensor: p + "ffn_down_shexp.weight")
        let scalar = try sigmoid(sum(input * weights.smallTensor(p + "ffn_gate_inp_shexp.weight"), axis: -1, keepDims: true))
        let output = routed + shared * scalar
        output.eval()
        return output
    }

    private func ple(_ residual: MLXArray, token: UInt64, layer: Int) throws -> MLXArray {
        let c = layout.config, p = "blk.\(layer).", wide = c.hiddenSize * c.hyperConnections
        let rows = try Qwen4ExpPLE.rows(current: token, predecessors: Array(previousTokens.reversed()), config: c)
        let embedded = try concatenated(rows.map { try weights.row($0, tensor: "per_layer_token_embd.weight") }, axis: 0)
            .reshaped([1, c.pleHeadDimension * c.pleOffsets.count])
        func norm(_ input: MLXArray, _ kind: String) throws -> MLXArray {
            try K3Norm.rms(input.reshaped([1, c.hyperConnections, c.hiddenSize]),
                weight: weights.smallTensor(p + "ple_norm_\(kind).weight").reshaped([c.hyperConnections, c.hiddenSize]),
                eps: layout.epsilon)
        }
        let key = try norm(weights.linear(embedded, tensor: p + "ple_key.weight"), "key")
        let query = try norm(residual, "query")
        let score = sum(key * query, axis: -1, keepDims: true) / Float(sqrt(Double(c.hiddenSize)))
        let magnitude = sqrt(maximum(abs(score), MLXArray(Float(1e-6))))
        let signed = which(score .> 0, magnitude, which(score .< 0, -magnitude, MLXArray(Float(0))))
        let value = try weights.linear(embedded, tensor: p + "ple_value.weight").reshaped([1, 1, c.hiddenSize])
        let gated = value * sigmoid(signed)
        let normalized = try norm(gated, "conv").reshaped([wide])
        let dilation = c.pleMultipliers.count
        let history = pleHistories[layer] ?? MLXArray.zeros([(layout.pleConvolutionWidth - 1) * dilation, wide], dtype: .float32)
        let conv = try Qwen4ExpGDN.convolutionStep(input: normalized,
            kernel: weights.smallTensor(p + "ple_conv1d.weight").T, history: history, dilation: dilation)
        pleHistories[layer] = conv.history
        let output = residual + (gated + conv.output.reshaped(residual.shape))
        output.eval()
        return output
    }
}
