import Foundation
import GGUFCore

/// Text-only Qwen3.5/3.6 MoE GGUF trunk, including Ornith 1.5 35B-A3B.
/// The file may contain one additional MTP block. It is validated as an extra
/// block and never counted as a main decoder layer or executed speculatively.
public struct Qwen35MoEConfig: Sendable {
    public let layers: Int
    public let mtpLayers: Int
    public let contextLimit: Int
    public let hiddenSize: Int
    public let vocabularySize: Int
    public let attentionInterval: Int
    public let attentionHeads: Int
    public let keyValueHeads: Int
    public let keyLength: Int
    public let valueLength: Int
    public let rotaryDimensions: Int
    public let ropeBase: Float
    public let epsilon: Float
    public let recurrentKeyHeads: Int
    public let recurrentHeads: Int
    public let recurrentStateSize: Int
    public let convolutionWidth: Int
    public let experts: Int
    public let expertsPerToken: Int
    public let expertWidth: Int
    public let sharedWidth: Int
    public let outputTensorName: String
    public var denseAttentionLayers: Int { layers / attentionInterval }
    public var recurrentLayers: Int { layers - denseAttentionLayers }

    public init(index: GGUFIndex) throws {
        guard index.metadata["general.architecture"]?.text == "qwen35moe" else {
            throw GGUFError.unsupported("expected qwen35moe architecture")
        }
        func integer(_ suffix: String, allowZero: Bool = false, fallback: Int? = nil) throws -> Int {
            guard let entry = index.metadata["qwen35moe." + suffix] else {
                if let fallback { return fallback }
                throw GGUFError.invalid("missing qwen35moe metadata: \(suffix)")
            }
            guard let value = entry.uint, value <= 1_048_576, value > 0 || allowZero else {
                throw GGUFError.invalid("qwen35moe integer: \(suffix)")
            }
            return Int(value)
        }
        func number(_ suffix: String) throws -> Float {
            guard case .number(let value)? = index.metadata["qwen35moe." + suffix],
                  value.isFinite, value > 0, value <= Double(Float.greatestFiniteMagnitude) else {
                throw GGUFError.invalid("qwen35moe float: \(suffix)")
            }
            return Float(value)
        }
        let blocks = try integer("block_count")
        mtpLayers = try integer("nextn_predict_layers", allowZero: true, fallback: 0)
        guard (0...1).contains(mtpLayers), blocks > mtpLayers, blocks <= 256 else {
            throw GGUFError.unsupported("qwen35moe main/MTP layer counts")
        }
        layers = blocks - mtpLayers
        contextLimit = try integer("context_length")
        hiddenSize = try integer("embedding_length")
        attentionInterval = try integer("full_attention_interval", fallback: 4)
        attentionHeads = try integer("attention.head_count")
        keyValueHeads = try integer("attention.head_count_kv")
        keyLength = try integer("attention.key_length")
        valueLength = try integer("attention.value_length")
        rotaryDimensions = try integer("rope.dimension_count")
        ropeBase = try number("rope.freq_base")
        epsilon = try number("attention.layer_norm_rms_epsilon")
        recurrentKeyHeads = try integer("ssm.group_count")
        recurrentHeads = try integer("ssm.time_step_rank")
        recurrentStateSize = try integer("ssm.state_size")
        convolutionWidth = try integer("ssm.conv_kernel")
        experts = try integer("expert_count")
        expertsPerToken = try integer("expert_used_count")
        expertWidth = try integer("expert_feed_forward_length")
        sharedWidth = try integer("expert_shared_feed_forward_length")
        guard hiddenSize <= 65536, attentionInterval <= layers, layers % attentionInterval == 0,
              (1...256).contains(attentionHeads), (1...256).contains(keyValueHeads),
              attentionHeads % keyValueHeads == 0, keyLength == valueLength, keyLength <= 1024,
              rotaryDimensions <= keyLength, rotaryDimensions % 2 == 0,
              (1...256).contains(recurrentKeyHeads), (1...256).contains(recurrentHeads),
              recurrentHeads % recurrentKeyHeads == 0, recurrentStateSize <= 512,
              (2...64).contains(convolutionWidth), experts <= 65536,
              expertsPerToken <= min(64, experts), expertWidth <= 65536, sharedWidth <= 65536 else {
            throw GGUFError.unsupported("qwen35moe bounded runtime geometry")
        }
        guard index.metadata["qwen35moe.attention.recurrent_layers"] == nil,
              index.metadata["qwen35moe.rope.scaling.type"] == nil,
              index.metadata["qwen35moe.attention.scale"] == nil else {
            throw GGUFError.unsupported("qwen35moe non-default recurrence/RoPE/attention scaling")
        }
        let rawSections = index.metadata["qwen35moe.rope.dimension_sections"]?.elements
        let sections = rawSections?.compactMap(\.uint)
        let rotation = rotaryDimensions
        guard rawSections?.count == 4, let sections, sections.count == 4, sections[3] == 0,
              sections.allSatisfy({ $0 <= UInt64(rotation) }),
              sections.reduce(0, +) * 2 == UInt64(rotation) else {
            throw GGUFError.invalid("qwen35moe text MRoPE sections")
        }
        let channels = (2 * recurrentKeyHeads + recurrentHeads) * recurrentStateSize
        guard channels * convolutionWidth <= 65536,
              try integer("ssm.inner_size") == recurrentHeads * recurrentStateSize,
              let embedding = index.tensors["token_embd.weight"], embedding.dimensions.count == 2,
              embedding.dimensions[0] == UInt64(hiddenSize),
              let vocabulary = Int(exactly: embedding.dimensions[1]), (1...300000).contains(vocabulary) else {
            throw GGUFError.invalid("qwen35moe recurrent/embedding geometry")
        }
        vocabularySize = vocabulary
        outputTensorName = index.tensors["output.weight"] == nil ? "token_embd.weight" : "output.weight"
        func require(_ name: String, _ dimensions: [Int]) throws {
            guard index.tensors[name]?.dimensions == dimensions.map({ UInt64($0) }) else {
                throw GGUFError.invalid("missing/wrong qwen35moe tensor: \(name)")
            }
        }
        try require(outputTensorName, [hiddenSize, vocabularySize])
        try require("output_norm.weight", [hiddenSize])
        for layer in 0..<blocks {
            let p = "blk.\(layer)."
            try require(p + "attn_norm.weight", [hiddenSize])
            try require(p + "post_attention_norm.weight", [hiddenSize])
            for kind in ["gate", "up"] {
                try require(p + "ffn_\(kind)_exps.weight", [hiddenSize, expertWidth, experts])
                try require(p + "ffn_\(kind)_shexp.weight", [hiddenSize, sharedWidth])
            }
            try require(p + "ffn_down_exps.weight", [expertWidth, hiddenSize, experts])
            try require(p + "ffn_down_shexp.weight", [sharedWidth, hiddenSize])
            try require(p + "ffn_gate_inp.weight", [hiddenSize, experts])
            try require(p + "ffn_gate_inp_shexp.weight", [hiddenSize])
            if layer >= layers || (layer + 1) % attentionInterval == 0 {
                try require(p + "attn_q.weight", [hiddenSize, 2 * attentionHeads * keyLength])
                try require(p + "attn_k.weight", [hiddenSize, keyValueHeads * keyLength])
                try require(p + "attn_v.weight", [hiddenSize, keyValueHeads * valueLength])
                try require(p + "attn_output.weight", [attentionHeads * valueLength, hiddenSize])
                for kind in ["q", "k"] { try require(p + "attn_\(kind)_norm.weight", [keyLength]) }
            } else {
                try require(p + "attn_qkv.weight", [hiddenSize, channels])
                try require(p + "attn_gate.weight", [hiddenSize, recurrentHeads * recurrentStateSize])
                for kind in ["alpha", "beta"] { try require(p + "ssm_\(kind).weight", [hiddenSize, recurrentHeads]) }
                try require(p + "ssm_conv1d.weight", [convolutionWidth, channels])
                try require(p + "ssm_dt.bias", [recurrentHeads])
                try require(p + "ssm_a", [recurrentHeads])
                try require(p + "ssm_norm.weight", [recurrentStateSize])
                try require(p + "ssm_out.weight", [recurrentHeads * recurrentStateSize, hiddenSize])
            }
            if layer >= layers {
                try require(p + "nextn.eh_proj.weight", [2 * hiddenSize, hiddenSize])
                for kind in ["enorm", "hnorm", "shared_head_norm"] {
                    try require(p + "nextn.\(kind).weight", [hiddenSize])
                }
            }
        }
    }

    public func memoryPlan(contextTokens: Int, budgetBytes: UInt64,
                           reservedWorkingBytes: UInt64 = 1_000_000_000) throws -> GGUFDecoderMemoryPlan {
        try GGUFDecoderMemoryPlan(contextTokens: contextTokens, maximumContextTokens: contextLimit,
            denseAttentionLayers: denseAttentionLayers, keyValueHeads: keyValueHeads,
            keyLength: keyLength, valueLength: valueLength, recurrentLayers: recurrentLayers,
            recurrentHeads: recurrentHeads, recurrentStateSize: recurrentStateSize,
            budgetBytes: budgetBytes, reservedWorkingBytes: reservedWorkingBytes)
    }
}
