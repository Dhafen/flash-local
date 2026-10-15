import Foundation
import GGUFCore
import MLX

/// Text-only dense attention for artifacts whose QSA ratios are zero. Each
/// step handles one token; chunked prefill and QSA remain separate work.
public enum Qwen4ExpTextAttention {
    public final class Cache {
        public let keyValueHeads: Int
        public let headDimension: Int
        public let contextLimit: Int
        public let chunkTokens: Int
        public private(set) var count = 0
        private var keys: [MLXArray] = []
        private var values: [MLXArray] = []

        /// FP16 K and V have an exact maximum storage cost. App/graph/allocator
        /// reserves are additional and belong to the complete memory plan.
        public init(keyValueHeads: Int, headDimension: Int, contextLimit: Int,
                    chunkTokens: Int = 256, maximumBytes: UInt64) throws {
            guard (1...128).contains(keyValueHeads), (1...1024).contains(headDimension),
                  (1...262_144).contains(contextLimit), (1...1024).contains(chunkTokens),
                  keyValueHeads * headDimension * chunkTokens <= 1_048_576 else {
                throw GGUFError.invalid("KV cache geometry")
            }
            let required = UInt64(keyValueHeads) * UInt64(headDimension) * UInt64(contextLimit) * 4
            guard required <= maximumBytes else { throw GGUFError.limit("KV cache admission") }
            self.keyValueHeads = keyValueHeads; self.headDimension = headDimension
            self.contextLimit = contextLimit; self.chunkTokens = chunkTokens
        }

        public var allocatedBytes: Int {
            keys.reduce(0) { $0 + $1.nbytes } + values.reduce(0) { $0 + $1.nbytes }
        }

        public func reset() {
            keys.removeAll(keepingCapacity: true); values.removeAll(keepingCapacity: true); count = 0
        }

        /// Query [queryHeads, dim], key/value [kvHeads, dim]. GQA uses contiguous
        /// query groups per KV head. KV chunks are FP16; reductions are FP32.
        /// Online softmax avoids expanding the entire KV cache to all Q heads
        /// or materialising a context-by-context attention matrix.
        public func appendAndAttend(query: MLXArray, key: MLXArray, value: MLXArray) throws -> MLXArray {
            guard count < contextLimit else { throw GGUFError.limit("context exhausted; no silent truncation") }
            guard query.ndim == 2, (1...256).contains(query.shape[0]),
                  query.shape[0] % keyValueHeads == 0, query.shape[1] == headDimension,
                  key.shape == [keyValueHeads, headDimension], value.shape == key.shape else {
                throw GGUFError.invalid("attention step geometry")
            }
            let newKeys = key.asType(.float16).reshaped([keyValueHeads, 1, headDimension])
            let newValues = value.asType(.float16).reshaped([keyValueHeads, 1, headDimension])
            if count % chunkTokens == 0 {
                newKeys.eval(); newValues.eval()
                keys.append(newKeys); values.append(newValues)
            } else {
                let last = keys.count - 1
                let mergedKeys = concatenated([keys[last], newKeys], axis: 1)
                let mergedValues = concatenated([values[last], newValues], axis: 1)
                mergedKeys.eval(); mergedValues.eval()
                keys[last] = mergedKeys; values[last] = mergedValues
            }
            count += 1
            let groups = query.shape[0] / keyValueHeads
            let groupedQuery = query.asType(.float32).reshaped([keyValueHeads, groups, headDimension])
            var maximum = MLXArray.full([keyValueHeads, groups, 1], values: MLXArray(-Float.infinity), dtype: .float32)
            var denominator = MLXArray.zeros([keyValueHeads, groups, 1], dtype: .float32)
            var numerator = MLXArray.zeros([keyValueHeads, groups, headDimension], dtype: .float32)
            for chunk in keys.indices {
                let scores = matmul(groupedQuery, keys[chunk].asType(.float32).transposed(0, 2, 1))
                    / Float(sqrt(Double(headDimension)))
                let nextMaximum = MLX.maximum(maximum, scores.max(axis: -1, keepDims: true))
                let oldScale = exp(maximum - nextMaximum)
                let probabilities = exp(scores - nextMaximum)
                numerator = numerator * oldScale + matmul(probabilities, values[chunk].asType(.float32))
                denominator = denominator * oldScale + sum(probabilities, axis: -1, keepDims: true)
                maximum = nextMaximum
                numerator.eval(); denominator.eval(); maximum.eval()
            }
            let output = (numerator / denominator).reshaped(query.shape)
            output.eval()
            return output
        }
    }

    /// Consumes already projected matrices, allowing all projections to use
    /// GGUFStreamedLinear. Q and gate are interleaved *per head* in the GGUF.
    /// For text, all IMRoPE position streams coincide, reducing to ordinary
    /// split-half partial RoPE. Images/video are deliberately unsupported here.
    public static func step(
        queryAndGate: MLXArray, key: MLXArray, value: MLXArray,
        queryNorm: MLXArray, keyNorm: MLXArray, epsilon: Float,
        rotaryDimensions: Int, ropeBase: Float, cache: Cache
    ) throws -> MLXArray {
        let dim = cache.headDimension, kvHeads = cache.keyValueHeads
        guard queryAndGate.ndim == 2, queryAndGate.shape[0] == 1,
              queryAndGate.shape[1] > 0, queryAndGate.shape[1] % (2 * dim) == 0,
              key.shape == [1, kvHeads * dim], value.shape == key.shape,
              queryNorm.shape == [dim], keyNorm.shape == [dim],
              epsilon > 0, epsilon.isFinite, rotaryDimensions > 0,
              rotaryDimensions <= dim, rotaryDimensions % 2 == 0,
              ropeBase > 0, ropeBase.isFinite else {
            throw GGUFError.invalid("projected text attention geometry")
        }
        let heads = queryAndGate.shape[1] / (2 * dim)
        let projected = queryAndGate.reshaped([heads, 2, dim])
        let queries = K3Norm.rms(projected[0..., 0, 0...], weight: queryNorm, eps: epsilon)
        let keys = K3Norm.rms(key.reshaped([kvHeads, dim]), weight: keyNorm, eps: epsilon)
        let rotatedQuery = MLXFast.RoPE(queries.reshaped([1, heads, 1, dim]),
            dimensions: rotaryDimensions, traditional: false, base: ropeBase, scale: 1, offset: cache.count)
        let rotatedKey = MLXFast.RoPE(keys.reshaped([1, kvHeads, 1, dim]),
            dimensions: rotaryDimensions, traditional: false, base: ropeBase, scale: 1, offset: cache.count)
        let attended = try cache.appendAndAttend(query: rotatedQuery.reshaped([heads, dim]),
            key: rotatedKey.reshaped([kvHeads, dim]), value: value.reshaped([kvHeads, dim]))
        let output = attended * sigmoid(projected[0..., 1, 0...].asType(.float32))
        return output.reshaped([1, heads * dim])
    }
}
