import Foundation
@testable import OrnithPerformance

/// Deterministic untrained split-DeltaNet/attention checkpoint with the real
/// affine g64 layout, 8-bit routers and 4-bit experts. Under 1 MB of payload.
struct SyntheticMLX {
    let manifest: MLXSourceManifest
    let files: [String: Data]
    let tensors: [String: Data]
    let config: Data

    init() throws {
        var tensors: [String: Data] = [:]
        var infos: [String: [String: Any]] = [:]
        var seed: UInt32 = 0x1234abcd
        func next() -> UInt32 { seed = seed &* 1664525 &+ 1013904223; return seed }
        func tensor(_ name: String, _ shape: [Int], dtype: String, value: Float? = nil) {
            var data = Data()
            for _ in 0..<shape.reduce(1, *) {
                if dtype == "U32" {
                    var word = next().littleEndian
                    withUnsafeBytes(of: &word) { data.append(contentsOf: $0) }
                } else {
                    let f = value ?? Float(Int(next() % 101) - 50) * 0.002
                    var bf16 = UInt16(f.bitPattern >> 16).littleEndian
                    withUnsafeBytes(of: &bf16) { data.append(contentsOf: $0) }
                }
            }
            let full = name == "lm_head.weight" || name.hasPrefix("lm_head.") ? name : "language_model." + name
            tensors[full] = data
            infos[full] = ["dtype": dtype, "shape": shape]
        }
        func linear(_ module: String, _ rows: [Int], _ input: Int, bits: Int = 4) {
            tensor(module + ".weight", rows + [input / (32 / bits)], dtype: "U32")
            tensor(module + ".scales", rows + [input / 64], dtype: "BF16", value: bits == 8 ? 0.00048828125 : 0.0078125)
            tensor(module + ".biases", rows + [input / 64], dtype: "BF16", value: -0.0625)
        }
        var quant: [String: Any] = ["bits": 4, "group_size": 64, "mode": "affine"]
        linear("model.embed_tokens", [128], 64)
        linear("lm_head", [128], 64)
        tensor("model.norm.weight", [64], dtype: "BF16")
        for layer in 0..<4 {
            let p = "model.layers.\(layer)."
            tensor(p + "input_layernorm.weight", [64], dtype: "BF16")
            tensor(p + "post_attention_layernorm.weight", [64], dtype: "BF16")
            linear(p + "mlp.gate", [8], 64, bits: 8)
            linear(p + "mlp.shared_expert_gate", [1], 64, bits: 8)
            for name in ["mlp.gate", "mlp.shared_expert_gate"] {
                quant["language_model." + p + name] = ["bits": 8, "group_size": 64]
            }
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                linear(p + "mlp.switch_mlp." + proj, [8, 64], 64)
                linear(p + "mlp.shared_expert." + proj, [64], 64)
            }
            if layer == 3 {
                for (proj, out) in [("q_proj", 128), ("k_proj", 32), ("v_proj", 32), ("o_proj", 64)] {
                    linear(p + "self_attn." + proj, [out], 64)
                }
                tensor(p + "self_attn.q_norm.weight", [16], dtype: "BF16")
                tensor(p + "self_attn.k_norm.weight", [16], dtype: "BF16")
            } else {
                tensor(p + "linear_attn.A_log", [2], dtype: "BF16", value: -1)
                tensor(p + "linear_attn.dt_bias", [2], dtype: "BF16", value: -2)
                tensor(p + "linear_attn.conv1d.weight", [192, 4, 1], dtype: "BF16")
                tensor(p + "linear_attn.norm.weight", [32], dtype: "BF16")
                for (proj, out) in [("in_proj_qkv", 192), ("in_proj_z", 64),
                                    ("in_proj_a", 2), ("in_proj_b", 2), ("out_proj", 64)] {
                    linear(p + "linear_attn." + proj, [out], 64)
                }
            }
        }
        let text: [String: Any] = [
            "model_type": "qwen3_5_moe_text", "hidden_size": 64, "num_hidden_layers": 4,
            "num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 64,
            "shared_expert_intermediate_size": 64, "vocab_size": 128, "full_attention_interval": 4,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "linear_num_key_heads": 2, "linear_num_value_heads": 2,
            "linear_key_head_dim": 32, "linear_value_head_dim": 32, "linear_conv_kernel_dim": 4,
            "max_position_embeddings": 512, "rms_norm_eps": 0.000001, "norm_topk_prob": true,
            "tie_word_embeddings": false, "rope_parameters": ["rope_theta": 10_000_000.0, "partial_rotary_factor": 0.25],
        ]
        let config = try JSONSerialization.data(withJSONObject: ["model_type": "qwen3_5_moe",
            "text_config": text, "quantization": quant], options: [.sortedKeys])
        var header: [String: Any] = [:]
        var body = Data()
        for name in tensors.keys.sorted() {
            var info = infos[name]!
            let bytes = tensors[name]!
            info["data_offsets"] = [body.count, body.count + bytes.count]
            header[name] = info
            body.append(bytes)
        }
        let rawHeader = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        var length = UInt64(rawHeader.count).littleEndian
        var shard = Data()
        withUnsafeBytes(of: &length) { shard.append(contentsOf: $0) }
        shard.append(rawHeader)
        shard.append(body)
        let shardName = "model-00001-of-00001.safetensors"
        var files = [shardName: shard, "config.json": config,
                     "tokenizer.json": Data("{}".utf8), "tokenizer_config.json": Data("{}".utf8),
                     "chat_template.jinja": Data("{{ messages }}".utf8)]
        let auxiliary = files.keys.filter { $0 != shardName }.sorted().map {
            MLXSourceManifest.File(name: $0, byteCount: files[$0]!.count, sha256: digest(files[$0]!))
        }
        manifest = MLXSourceManifest(schemaVersion: 1, repository: "synthetic/ornith-layout",
            revision: String(repeating: "0", count: 40),
            swiftletRevision: "de1a942241f65ac43290cecffc85d69ea5b52310",
            configBase64: config.base64EncodedString(), auxiliary: auxiliary,
            shards: [.init(name: shardName, byteCount: shard.count, sha256: digest(shard),
                           headerSHA256: digest(rawHeader), headerBase64: rawHeader.base64EncodedString())])
        // Raw CPU reference opens a single model.safetensors, never a full
        // trained model. This extra fixture alias is not downloaded/installed.
        files["model.safetensors"] = shard
        self.files = files
        self.tensors = tensors
        self.config = config
    }

    func plan() throws -> QPackInstallPlan { try QPackInstallPlan(source: MLXSourceIndex(manifest: manifest)) }
}

actor FixtureByteSource: QPackByteSource {
    enum Failure: Error { case interrupted }
    let files: [String: Data]
    let failAfterRanges: Int?
    var requests: [(name: String, range: Range<Int>?)] = []
    var rangeRequests = 0
    init(_ fixture: SyntheticMLX, failAfterRanges: Int? = nil) {
        files = fixture.files
        self.failAfterRanges = failAfterRanges
    }
    func data(name: String, range: Range<Int>?, totalByteCount: Int) throws -> Data {
        if range != nil, let failAfterRanges, rangeRequests >= failAfterRanges { throw Failure.interrupted }
        requests.append((name, range))
        guard let data = files[name], data.count == totalByteCount else { throw Failure.interrupted }
        if let range { rangeRequests += 1; return data.subdata(in: range) }
        return data
    }
}
