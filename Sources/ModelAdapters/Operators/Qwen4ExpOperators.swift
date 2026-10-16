import Foundation
import GGUFCore
import MLX

public enum Qwen4ExpGDN {
    /// One token of GGML's G=1 delta rule. Inputs q/k are already normalised;
    /// logDecay is already -exp(A_log)*softplus(alpha+dt), beta is sigmoid.
    /// State is V-first [valueHeads, valueDim, keyDim], as in GGML Metal.
    /// Q/K head h is selected by h % keyHeads, not repeat-interleave.
    public static func step(q: MLXArray, k: MLXArray, v: MLXArray,
                            logDecay: MLXArray, beta: MLXArray, state: MLXArray) throws
        -> (output: MLXArray, state: MLXArray) {
        guard q.ndim == 2, q.shape == k.shape, v.ndim == 2,
              q.shape[0] > 0, q.shape[1] > 0, v.shape[0] > 0,
              v.shape[0] % q.shape[0] == 0, v.shape[1] == q.shape[1],
              logDecay.shape == [v.shape[0]], beta.shape == [v.shape[0]],
              state.shape == [v.shape[0], v.shape[1], q.shape[1]] else {
            throw GGUFError.invalid("GDN step geometry")
        }
        let heads = v.shape[0], dim = v.shape[1]
        let repeatCount = heads / q.shape[0]
        let queries = repeatCount == 1 ? q.asType(.float32)
            : concatenated(Array(repeating: q.asType(.float32), count: repeatCount), axis: 0)
        let keys = repeatCount == 1 ? k.asType(.float32)
            : concatenated(Array(repeating: k.asType(.float32), count: repeatCount), axis: 0)
        let decayed = state.asType(.float32) * exp(logDecay.asType(.float32)).reshaped([heads, 1, 1])
        let keyRows = keys.reshaped([heads, 1, dim])
        let predicted = sum(decayed * keyRows, axis: -1)
        let delta = (v.asType(.float32) - predicted) * beta.asType(.float32).reshaped([heads, 1])
        let next = decayed + delta.reshaped([heads, dim, 1]) * keyRows
        let output = sum(next * queries.reshaped([heads, 1, dim]), axis: -1) / Float(sqrt(Double(dim)))
        // Bound graph lifetime at a recurrent step. A long conversation must
        // not retain its entire computation history through state references.
        next.eval(); output.eval()
        return (output, next)
    }

    /// Short depthwise convolution shared by GDN and the dilated PLE path.
    /// Kernel [taps, channels], history [(taps-1)*dilation, channels], oldest first.
    public static func convolutionStep(input: MLXArray, kernel: MLXArray,
                                       history: MLXArray, dilation: Int = 1) throws
        -> (output: MLXArray, history: MLXArray) {
        guard input.ndim == 1, input.shape[0] > 0, kernel.ndim == 2,
              kernel.shape[0] >= 2, kernel.shape[0] <= 64,
              dilation > 0, dilation <= 16, kernel.shape[1] == input.shape[0],
              history.shape == [(kernel.shape[0] - 1) * dilation, input.shape[0]] else {
            throw GGUFError.invalid("causal convolution geometry")
        }
        let channels = input.shape[0]
        let padded = concatenated([history.asType(.float32), input.asType(.float32).reshaped([1, channels])], axis: 0)
        var convolved = MLXArray.zeros([channels], dtype: .float32)
        for tap in 0..<kernel.shape[0] {
            convolved = convolved + padded[tap * dilation] * kernel[tap].asType(.float32)
        }
        let output = K3Activation.silu(convolved)
        let tail = padded[1..., 0...]
        output.eval(); tail.eval()
        return (output, tail)
    }

    /// Qwen4Exp uses sigmoid output gating; Qwen3.5's SiLU is different.
    public static func gatedNorm(_ input: MLXArray, gate: MLXArray,
                                 weight: MLXArray, epsilon: Float) throws -> MLXArray {
        guard input.shape == gate.shape, input.ndim == 2,
              weight.shape == [input.shape[1]], epsilon > 0, epsilon.isFinite else {
            throw GGUFError.invalid("GDN gated RMS geometry")
        }
        return K3Norm.rms(input, weight: weight, eps: epsilon) * sigmoid(gate.asType(.float32))
    }
}

public enum Qwen4ExpHyperConnections {
    /// Projections are closures so every large matrix can be streamed and
    /// released before the next is requested. Gamma is already folded 1+w
    /// by the GGUF converter: do not add one a second time.
    public static func mix(
        _ residual: MLXArray, gamma: MLXArray, epsilon: Float,
        down: (MLXArray) throws -> MLXArray,
        up: (MLXArray) throws -> MLXArray,
        injection: ((MLXArray) throws -> MLXArray)? = nil
    ) throws -> (mixed: MLXArray, injection: MLXArray?) {
        guard residual.ndim == 3, residual.shape[0] > 0, residual.shape[1] > 1,
              residual.shape[2] > 0, gamma.shape == Array(residual.shape.dropFirst()),
              epsilon > 0, epsilon.isFinite else {
            throw GGUFError.invalid("hyper-connection input geometry")
        }
        let tokens = residual.shape[0], streams = residual.shape[1], hidden = residual.shape[2]
        let normalized = K3Norm.rms(residual, weight: gamma, eps: epsilon)
        let flat = normalized.reshaped([tokens, streams * hidden])
        let low = try down(flat)
        guard low.ndim == 2, low.shape[0] == tokens else {
            throw GGUFError.invalid("hyper-connection low-rank geometry")
        }
        let gate = try up(K3Activation.silu(low / Float(streams)))
        guard gate.shape == [tokens, streams * hidden] else {
            throw GGUFError.invalid("hyper-connection gate geometry")
        }
        // Match the unfused reference's ascending stream addition order.
        let gated = normalized * sigmoid(gate.reshaped(residual.shape))
        var mixed = gated[0..., 0, 0...]
        for stream in 1..<streams { mixed = mixed + gated[0..., stream, 0...] }
        mixed = mixed / Float(streams)
        let injected = try injection?(flat)
        guard injected == nil || injected!.shape == [tokens, streams] else {
            throw GGUFError.invalid("hyper-connection injection geometry")
        }
        mixed.eval(); injected?.eval()
        return (mixed, injected)
    }

    public static func combine(residual: MLXArray, blockOutput: MLXArray,
                               injection: MLXArray) throws -> MLXArray {
        guard residual.ndim == 3, residual.shape[1] > 1,
              blockOutput.shape == [residual.shape[0], residual.shape[2]],
              injection.shape == [residual.shape[0], residual.shape[1]] else {
            throw GGUFError.invalid("hyper-connection combination geometry")
        }
        let weights = 2 * sigmoid(injection.asType(.float32) / Float(residual.shape[1]))
        return residual.asType(.float32)
            + expandedDimensions(blockOutput.asType(.float32), axis: 1) * expandedDimensions(weights, axis: -1)
    }
}

public enum Qwen4ExpRouting {
    public struct Selection {
        public let ids: [[Int]]
        public let weights: MLXArray
    }

    public static func select(logits: MLXArray, topK: Int) throws -> Selection {
        guard logits.ndim == 2, logits.shape[0] > 0, logits.shape[0] <= 256,
              logits.shape[1] <= 65_536, topK > 0, topK <= logits.shape[1] else {
            throw GGUFError.invalid("MoE routing geometry")
        }
        let gates = softmax(logits.asType(.float32), axis: -1)
        let values = gates.asArray(Float.self)
        guard values.allSatisfy({ $0.isFinite }) else { throw GGUFError.invalid("nonfinite routing") }
        let experts = logits.shape[1]
        var ids: [[Int]] = [], selected: [Float] = []
        for token in 0..<logits.shape[0] {
            let row = Array(values[(token * experts)..<((token + 1) * experts)])
            let order = (0..<experts).sorted { a, b in row[a] == row[b] ? a < b : row[a] > row[b] }
            let choice = Array(order.prefix(topK))
            ids.append(choice); selected.append(contentsOf: choice.map { row[$0] })
        }
        var weights = MLXArray(selected, [logits.shape[0], topK])
        weights = weights / sum(weights, axis: -1, keepDims: true)
        weights.eval()
        return Selection(ids: ids, weights: weights)
    }
}
