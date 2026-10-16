import CryptoKit
import Foundation
import SwiftletCore

public enum OrnithPerformanceError: Error, Equatable, LocalizedError {
    case invalidSource(String)
    case unsupportedTensor(String)
    case arithmeticOverflow
    case insufficientMemory(required: UInt64, available: UInt64)
    case integrity(String)
    case io(String)
    case cancelled

    public var errorDescription: String? {
        switch self {
        case .invalidSource(let detail): return "Model configuration: \(detail)"
        case .unsupportedTensor(let name): return "Unsupported model tensor: \(name)"
        case .arithmeticOverflow: return "The model size exceeds the supported bounds."
        case .insufficientMemory(let required, let available):
            return "Not enough memory: requires \(required) bytes; allowance \(available) bytes."
        case .integrity(let detail): return "Model verification failed: \(detail)"
        case .io(let detail): return "Model storage: \(detail)"
        case .cancelled: return "Cancelled."
        }
    }
}

func digest(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

func checkedAdd(_ a: Int, _ b: Int) throws -> Int {
    let (result, overflow) = a.addingReportingOverflow(b)
    guard !overflow, result >= 0 else { throw OrnithPerformanceError.arithmeticOverflow }
    return result
}

func checkedMultiply(_ a: Int, _ b: Int) throws -> Int {
    guard a >= 0, b >= 0 else { throw OrnithPerformanceError.arithmeticOverflow }
    let (result, overflow) = a.multipliedReportingOverflow(by: b)
    guard !overflow else { throw OrnithPerformanceError.arithmeticOverflow }
    return result
}

public struct MLXSourceManifest: Codable, Sendable {
    public struct File: Codable, Sendable {
        public let name: String
        public let byteCount: Int
        public let sha256: String
    }
    public struct Shard: Codable, Sendable {
        public let name: String
        public let byteCount: Int
        public let sha256: String
        public let headerSHA256: String
        public let headerBase64: String
    }
    public let schemaVersion: Int
    public let repository: String
    public let revision: String
    public let swiftletRevision: String
    public let configBase64: String
    public let auxiliary: [File]
    public let shards: [Shard]

    public static func ornith() throws -> MLXSourceManifest {
        guard let url = Bundle.module.url(forResource: "ornith-mlx-source", withExtension: "json") else {
            throw OrnithPerformanceError.invalidSource("missing bundled source contract")
        }
        let manifest = try JSONDecoder().decode(Self.self, from: Data(contentsOf: url))
        guard manifest.repository == "ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit",
              manifest.revision == "19504d912fa8fc7622bf6b1de3db5d5d890b1f02",
              manifest.swiftletRevision == "de1a942241f65ac43290cecffc85d69ea5b52310" else {
            throw OrnithPerformanceError.invalidSource("source/engine revision changed")
        }
        return manifest
    }

    /// A resume journal is useful only for this exact source and byte layout.
    public func identity() throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return digest(try encoder.encode(self))
    }
}

public struct MLXModelGeometry: Codable, Equatable, Sendable {
    public let hidden: Int
    public let layers: Int
    public let experts: Int
    public let topK: Int
    public let intermediate: Int
    public let sharedIntermediate: Int
    public let vocabulary: Int
    public let attentionInterval: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDimension: Int
    public let linearKeyHeads: Int
    public let linearValueHeads: Int
    public let linearKeyDimension: Int
    public let linearValueDimension: Int
    public let convolutionWidth: Int
    public let maximumContext: Int

    public var attentionLayers: Int { layers / attentionInterval }
    public var recurrentLayers: Int { layers - attentionLayers }
    public func isLinear(_ layer: Int) -> Bool { (layer + 1) % attentionInterval != 0 }

    init(_ config: [String: Any]) throws {
        let text = config["text_config"] as? [String: Any] ?? config
        guard (try QwenConfig.modelType(fromConfig: config)).hasPrefix("qwen3_5") else {
            throw OrnithPerformanceError.invalidSource("experiment requires the split Qwen3.5 MoE graph")
        }
        func positive(_ key: String) throws -> Int {
            guard let n = text[key] as? Int, n > 0, n <= 1_048_576 else {
                throw OrnithPerformanceError.invalidSource("invalid architecture field: \(key)")
            }
            return n
        }
        hidden = try positive("hidden_size")
        layers = try positive("num_hidden_layers")
        experts = try positive("num_experts")
        topK = try positive("num_experts_per_tok")
        intermediate = try positive("moe_intermediate_size")
        sharedIntermediate = try positive("shared_expert_intermediate_size")
        vocabulary = try positive("vocab_size")
        attentionInterval = try positive("full_attention_interval")
        attentionHeads = try positive("num_attention_heads")
        kvHeads = try positive("num_key_value_heads")
        headDimension = try positive("head_dim")
        linearKeyHeads = try positive("linear_num_key_heads")
        linearValueHeads = try positive("linear_num_value_heads")
        linearKeyDimension = try positive("linear_key_head_dim")
        linearValueDimension = try positive("linear_value_head_dim")
        convolutionWidth = try positive("linear_conv_kernel_dim")
        maximumContext = try positive("max_position_embeddings")
        guard topK <= experts, layers <= 128, experts <= 1024,
              convolutionWidth <= 16, layers % attentionInterval == 0,
              text["norm_topk_prob"] as? Bool != false,
              text["tie_word_embeddings"] as? Bool != true else {
            throw OrnithPerformanceError.invalidSource("unsupported routing/layer/embedding geometry")
        }
    }

    public func requirePublishedOrnith() throws {
        guard hidden == 2048, layers == 40, experts == 256, topK == 8,
              intermediate == 512, sharedIntermediate == 512, vocabulary == 248_320,
              attentionInterval == 4, attentionHeads == 16, kvHeads == 2, headDimension == 256,
              linearKeyHeads == 16, linearValueHeads == 32, linearKeyDimension == 128,
              linearValueDimension == 128, convolutionWidth == 4 else {
            throw OrnithPerformanceError.invalidSource("published Ornith architecture differs")
        }
    }
}

public struct MLXStoredTensor: Sendable {
    public let name: String
    public let dtype: String
    public let shape: [Int]
    public let sourceShard: Int
    public let sourceRange: Range<Int>
    public var byteCount: Int { sourceRange.count }
}

/// Validates *every* layer's actual stored bytes. Matching a model name or the
/// first layer is insufficient for fixed-stride expert placement.
public struct MLXSourceIndex: Sendable {
    public let manifest: MLXSourceManifest
    public let geometry: MLXModelGeometry
    public let configData: Data
    public let sourceHeaders: [Data]
    public let tensors: [String: MLXStoredTensor]

    public init(manifest: MLXSourceManifest) throws {
        guard manifest.schemaVersion == 1, !manifest.shards.isEmpty,
              manifest.shards.count <= 64, manifest.auxiliary.count <= 16,
              let configData = Data(base64Encoded: manifest.configBase64),
              configData.count <= 1 << 20,
              manifest.auxiliary.contains(where: {
                  $0.name == "config.json" && $0.byteCount == configData.count && $0.sha256 == digest(configData)
              }),
              let config = try JSONSerialization.jsonObject(with: configData) as? [String: Any] else {
            throw OrnithPerformanceError.invalidSource("invalid source contract/config")
        }
        let geometry = try MLXModelGeometry(config)
        let quant = try Checkpoint.quantization(fromConfig: config)
        var all: [String: MLXStoredTensor] = [:]
        var headers: [Data] = []
        var filenames = Set<String>()
        for file in manifest.auxiliary {
            guard Self.safeFilename(file.name), filenames.insert(file.name).inserted,
                  file.byteCount > 0, file.byteCount <= 32 << 20, Self.isDigest(file.sha256) else {
                throw OrnithPerformanceError.invalidSource("invalid auxiliary file")
            }
        }
        for (shardIndex, shard) in manifest.shards.enumerated() {
            guard Self.safeFilename(shard.name), filenames.insert(shard.name).inserted,
                  shard.byteCount > 8, Self.isDigest(shard.sha256),
                  let raw = Data(base64Encoded: shard.headerBase64), raw.count <= 8 << 20,
                  digest(raw) == shard.headerSHA256,
                  let header = try JSONSerialization.jsonObject(with: raw) as? [String: Any] else {
                throw OrnithPerformanceError.invalidSource("invalid pinned shard/header")
            }
            var original = Data()
            var length = UInt64(raw.count).littleEndian
            withUnsafeBytes(of: &length) { original.append(contentsOf: $0) }
            original.append(raw)
            headers.append(original)
            var intervals: [Range<Int>] = []
            for (sourceName, entry) in header where sourceName != "__metadata__" {
                guard let info = entry as? [String: Any], let dtype = info["dtype"] as? String,
                      let elementBytes = ["U32": 4, "F32": 4, "F16": 2, "BF16": 2][dtype],
                      let shape = info["shape"] as? [Int], !shape.isEmpty,
                      shape.count <= 4, shape.allSatisfy({ $0 > 0 }),
                      let offsets = info["data_offsets"] as? [Int], offsets.count == 2,
                      offsets[0] >= 0, offsets[1] >= offsets[0] else {
                    throw OrnithPerformanceError.unsupportedTensor(sourceName)
                }
                let bytes = try shape.reduce(elementBytes) { try checkedMultiply($0, $1) }
                guard offsets[1] - offsets[0] == bytes else {
                    throw OrnithPerformanceError.unsupportedTensor(sourceName)
                }
                let start = try checkedAdd(original.count, offsets[0])
                let end = try checkedAdd(start, bytes)
                let name = Self.canonical(sourceName)
                guard end <= shard.byteCount, all[name] == nil else {
                    throw OrnithPerformanceError.invalidSource("duplicate/out-of-range tensor: \(sourceName)")
                }
                intervals.append(start..<end)
                all[name] = MLXStoredTensor(name: name, dtype: dtype, shape: shape,
                    sourceShard: shardIndex, sourceRange: start..<end)
            }
            var cursor = original.count
            for range in intervals.sorted(by: { $0.lowerBound < $1.lowerBound }) {
                guard cursor == range.lowerBound else {
                    throw OrnithPerformanceError.invalidSource("tensor gap/overlap; cannot reconstruct source")
                }
                cursor = range.upperBound
            }
            guard cursor == shard.byteCount else {
                throw OrnithPerformanceError.invalidSource("source shard length differs from tensors")
            }
        }
        var consumed = Set<String>()
        func expect(_ name: String, _ shape: [Int], _ dtype: String? = nil) throws {
            guard let t = all[name], t.shape == shape, dtype == nil || t.dtype == dtype else {
                throw OrnithPerformanceError.unsupportedTensor(name)
            }
            consumed.insert(name)
        }
        func plain(_ name: String, _ shape: [Int]) throws {
            try expect(name, shape)
            guard let t = all[name], ["F32", "F16", "BF16"].contains(t.dtype) else {
                throw OrnithPerformanceError.unsupportedTensor(name)
            }
        }
        func affine(_ module: String, _ rows: [Int], _ cols: Int, bits: Int) throws {
            guard let spec = Checkpoint.quantSpec(for: module, default: quant.default, overrides: quant.overrides),
                  spec.bits == bits, spec.groupSize == 64, cols % 64 == 0 else {
                throw OrnithPerformanceError.unsupportedTensor(module + " quantization")
            }
            try expect(module + ".weight", rows + [cols / (32 / bits)], "U32")
            try expect(module + ".scales", rows + [cols / 64])
            try expect(module + ".biases", rows + [cols / 64])
            guard all[module + ".scales"]!.dtype == all[module + ".biases"]!.dtype,
                  ["F32", "F16", "BF16"].contains(all[module + ".scales"]!.dtype) else {
                throw OrnithPerformanceError.unsupportedTensor(module + " scales/biases")
            }
        }
        let g = geometry
        try affine("model.embed_tokens", [g.vocabulary], g.hidden, bits: 4)
        try affine("lm_head", [g.vocabulary], g.hidden, bits: 4)
        try plain("model.norm.weight", [g.hidden])
        for layer in 0..<g.layers {
            let p = "model.layers.\(layer)."
            try plain(p + "input_layernorm.weight", [g.hidden])
            try plain(p + "post_attention_layernorm.weight", [g.hidden])
            try affine(p + "mlp.gate", [g.experts], g.hidden, bits: 8)
            try affine(p + "mlp.shared_expert_gate", [1], g.hidden, bits: 8)
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                let down = proj == "down_proj"
                try affine(p + "mlp.switch_mlp." + proj,
                    [g.experts, down ? g.hidden : g.intermediate], down ? g.intermediate : g.hidden, bits: 4)
                try affine(p + "mlp.shared_expert." + proj,
                    [down ? g.hidden : g.sharedIntermediate], down ? g.sharedIntermediate : g.hidden, bits: 4)
            }
            if g.isLinear(layer) {
                let key = try checkedMultiply(g.linearKeyHeads, g.linearKeyDimension)
                let value = try checkedMultiply(g.linearValueHeads, g.linearValueDimension)
                let conv = try checkedAdd(checkedMultiply(2, key), value)
                try plain(p + "linear_attn.conv1d.weight", [conv, g.convolutionWidth, 1])
                try plain(p + "linear_attn.A_log", [g.linearValueHeads])
                try plain(p + "linear_attn.dt_bias", [g.linearValueHeads])
                try plain(p + "linear_attn.norm.weight", [g.linearValueDimension])
                for (proj, out, input) in [
                    ("in_proj_qkv", conv, g.hidden), ("in_proj_z", value, g.hidden),
                    ("in_proj_b", g.linearValueHeads, g.hidden), ("in_proj_a", g.linearValueHeads, g.hidden),
                    ("out_proj", g.hidden, value),
                ] {
                    try affine(p + "linear_attn." + proj, [out], input, bits: 4)
                }
            } else {
                for (proj, out, input) in [
                    ("q_proj", 2 * g.attentionHeads * g.headDimension, g.hidden),
                    ("k_proj", g.kvHeads * g.headDimension, g.hidden),
                    ("v_proj", g.kvHeads * g.headDimension, g.hidden),
                    ("o_proj", g.hidden, g.attentionHeads * g.headDimension),
                ] {
                    try affine(p + "self_attn." + proj, [out], input, bits: 4)
                }
                try plain(p + "self_attn.q_norm.weight", [g.headDimension])
                try plain(p + "self_attn.k_norm.weight", [g.headDimension])
            }
        }
        guard consumed.count == all.count else {
            throw OrnithPerformanceError.unsupportedTensor(
                Set(all.keys).subtracting(consumed).sorted().first ?? "unknown tensor")
        }
        self.manifest = manifest
        self.geometry = geometry
        self.configData = configData
        self.sourceHeaders = headers
        self.tensors = all
    }

    static func canonical(_ name: String) -> String {
        name.hasPrefix("language_model.") ? String(name.dropFirst("language_model.".count)) : name
    }

    static func safeFilename(_ name: String) -> Bool {
        !name.isEmpty && name.utf8.count <= 128 && name != "." && name != ".."
            && name.utf8.allSatisfy { (48...57).contains($0) || (65...90).contains($0)
                || (97...122).contains($0) || $0 == 45 || $0 == 46 || $0 == 95 }
    }

    static func isDigest(_ text: String) -> Bool {
        text.utf8.count == 64 && text.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
    }
}
