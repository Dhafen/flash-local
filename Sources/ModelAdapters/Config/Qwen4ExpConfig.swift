import Foundation
import GGUFCore

/// Architecture parameters taken from the artifact, never guessed from a
/// product name. In particular, zero QSA ratios mean dense attention.
public struct Qwen4ExpConfig: Sendable, Equatable {
    public let layers: Int
    public let hiddenSize: Int
    public let experts: Int
    public let expertsPerToken: Int
    public let attentionInterval: Int
    public let keyValueHeads: Int
    public let keyLength: Int
    public let valueLength: Int
    public let recurrentHeads: Int
    public let recurrentStateSize: Int
    public let hyperConnections: Int
    public let contextLimit: Int
    public let compressionRatios: [UInt64]
    public let pleLayers: [UInt64]
    public let pleMultipliers: [UInt64]
    public let pleOffsets: [UInt64]
    public let pleVocabularySizes: [UInt64]
    public let pleEOS: UInt64
    public let pleHeadDimension: Int

    public init(index: GGUFIndex) throws {
        let metadata = index.metadata
        guard metadata["general.architecture"]?.text == "qwen4exp" else {
            throw GGUFError.unsupported("Qwen4Exp requires general.architecture=qwen4exp")
        }
        func positive(_ suffix: String) throws -> Int {
            guard let value = metadata["qwen4exp.\(suffix)"]?.uint,
                  let integer = Int(exactly: value), integer > 0 else {
                throw GGUFError.invalid("missing or invalid qwen4exp.\(suffix)")
            }
            return integer
        }
        func array(_ suffix: String) throws -> [UInt64] {
            guard let values = metadata["qwen4exp.\(suffix)"]?.elements else {
                throw GGUFError.invalid("missing qwen4exp.\(suffix) array")
            }
            return try values.map {
                guard let value = $0.uint else { throw GGUFError.invalid("non-integer \(suffix)") }
                return value
            }
        }
        layers = try positive("block_count")
        hiddenSize = try positive("embedding_length")
        experts = try positive("expert_count")
        expertsPerToken = try positive("expert_used_count")
        attentionInterval = try positive("full_attention_interval")
        keyValueHeads = try positive("attention.head_count_kv")
        keyLength = try positive("attention.key_length")
        valueLength = try positive("attention.value_length")
        recurrentHeads = try positive("ssm.time_step_rank")
        recurrentStateSize = try positive("ssm.state_size")
        hyperConnections = try positive("hyper_connection.count")
        contextLimit = try positive("context_length")
        compressionRatios = try array("attention.compress_ratios")
        pleLayers = try array("ple.layers")
        pleMultipliers = try array("ple.layer_multipliers")
        pleOffsets = try array("ple.head_offsets")
        pleVocabularySizes = try array("ple.head_vocab_sizes")
        pleHeadDimension = try positive("embedding_length_per_layer_input")
        guard let eos = metadata["qwen4exp.ple.eos_token_id"]?.uint else {
            throw GGUFError.invalid("missing PLE EOS")
        }
        pleEOS = eos
        let ngramSize = try positive("ple.ngram_size")
        let headsPerNgram = try positive("ple.heads_per_ngram")
        guard (2...4).contains(ngramSize), headsPerNgram <= 4096 / max(1, ngramSize - 1),
              expertsPerToken <= experts,
              hyperConnections > 1, attentionInterval <= layers,
              compressionRatios.count == layers,
              pleMultipliers.count == ngramSize,
              pleOffsets.count == (ngramSize - 1) * headsPerNgram,
              pleOffsets.count == pleVocabularySizes.count,
              Set(pleLayers).count == pleLayers.count else {
            throw GGUFError.invalid("Qwen4Exp architecture dimensions")
        }
        for (layer, ratio) in compressionRatios.enumerated() {
            guard ratio == 0 || ((layer + 1) % attentionInterval == 0 && ratio > 1) else {
                throw GGUFError.invalid("QSA ratio on non-attention layer")
            }
        }
        guard let table = index.tensors["per_layer_token_embd.weight"],
              table.dimensions.count == 2,
              table.dimensions[0] == UInt64(pleHeadDimension) else {
            throw GGUFError.invalid("PLE table geometry")
        }
        for layer in pleLayers {
            guard layer < UInt64(layers), (layer + 1) % UInt64(attentionInterval) != 0 else {
                throw GGUFError.invalid("PLE layer must be recurrent")
            }
        }
        for (offset, count) in zip(pleOffsets, pleVocabularySizes) {
            let end = offset.addingReportingOverflow(count)
            guard count > 0, !end.overflow, end.partialValue <= table.dimensions[1] else {
                throw GGUFError.invalid("PLE head range outside table")
            }
        }
    }

    public var denseAttentionLayers: Int { layers / attentionInterval }
    public var recurrentLayers: Int { layers - denseAttentionLayers }
    public var usesSparseAttention: Bool { compressionRatios.contains { $0 > 0 } }
}

/// UInt64 PLE hashing matches the reference's unsigned wraparound. EOS only
/// resets predecessors; the current token does not reset its own context.
public enum Qwen4ExpPLE {
    public static func rows(current: UInt64, predecessors: [UInt64], config: Qwen4ExpConfig) throws -> [UInt64] {
        let grams = config.pleMultipliers.count
        guard predecessors.count <= grams - 1 else { throw GGUFError.invalid("PLE history length") }
        var context = [current]
        var cut = false
        for i in 0..<(grams - 1) {
            let token = i < predecessors.count ? predecessors[i] : config.pleEOS
            cut = cut || token == config.pleEOS
            context.append(cut ? config.pleEOS : token)
        }
        let perGram = config.pleOffsets.count / (grams - 1)
        var rows: [UInt64] = []
        for n in 2...grams {
            var mixed = context[0] &* config.pleMultipliers[0]
            for j in 1..<n { mixed ^= context[j] &* config.pleMultipliers[j] }
            for h in ((n - 2) * perGram)..<((n - 1) * perGram) {
                rows.append(mixed % config.pleVocabularySizes[h] + config.pleOffsets[h])
            }
        }
        return rows
    }
}

/// Admission arithmetic, not a claim about measured process footprint. Every
/// caller must pay for app/graph/Metal reserve and sample the live footprint.
public struct Qwen4ExpMemoryPlan: Sendable, Equatable {
    public let contextTokens: Int
    public let kvBytes: UInt64
    public let recurrentMatrixBytes: UInt64
    public let reservedWorkingBytes: UInt64
    public let cacheBytes: UInt64
    public let budgetBytes: UInt64

    public init(config: Qwen4ExpConfig, contextTokens: Int, budgetBytes: UInt64,
                reservedWorkingBytes: UInt64) throws {
        guard contextTokens > 0, contextTokens <= config.contextLimit,
              budgetBytes <= 5_500_000_000, reservedWorkingBytes >= 512 * 1024 * 1024 else {
            throw GGUFError.invalid("context, budget, or working reserve")
        }
        func product(_ values: [UInt64]) throws -> UInt64 {
            try values.reduce(1) {
                let result = $0.multipliedReportingOverflow(by: $1)
                guard !result.overflow else { throw GGUFError.invalid("memory arithmetic overflow") }
                return result.partialValue
            }
        }
        kvBytes = try product([UInt64(config.denseAttentionLayers), UInt64(contextTokens),
            UInt64(config.keyValueHeads), UInt64(config.keyLength) + UInt64(config.valueLength), 2])
        recurrentMatrixBytes = try product([UInt64(config.recurrentLayers), UInt64(config.recurrentHeads),
            UInt64(config.recurrentStateSize), UInt64(config.recurrentStateSize), 4])
        var floor = kvBytes
        for bytes in [recurrentMatrixBytes, reservedWorkingBytes] {
            let result = floor.addingReportingOverflow(bytes)
            guard !result.overflow else { throw GGUFError.invalid("memory arithmetic overflow") }
            floor = result.partialValue
        }
        guard floor <= budgetBytes else { throw GGUFError.limit("context and working set exceed budget") }
        self.contextTokens = contextTokens
        self.budgetBytes = budgetBytes
        self.reservedWorkingBytes = reservedWorkingBytes
        cacheBytes = budgetBytes - floor
    }
}
