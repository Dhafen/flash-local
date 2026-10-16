import Foundation
import GGUFCore
import MLX

/// The GGML CPU RMS operation accumulates Float32 squares into a Float64 sum,
/// then rounds its mean, reciprocal square root and two products to Float32.
/// A GPU reduction/rsqrt can differ by one ULP; a subsequent Q8 activation or
/// FP16 KV cast can amplify that at a rounding boundary. This bounded helper
/// aligns qwen35moe's small norm rows, without loading any projection weights.
enum Qwen35CPUAlignedNorm {
    static func rms(_ x: MLXArray, weight: MLXArray, eps: Float) throws -> MLXArray {
        guard let columns = x.shape.last, columns > 0, columns <= 65536,
              x.size > 0, x.size <= 65536, weight.shape == [columns],
              eps.isFinite, eps > 0 else { throw GGUFError.limit("qwen35moe bounded RMS rows") }
        let values = x.asType(.float32).asArray(Float.self)
        let gamma = weight.asType(.float32).asArray(Float.self)
        guard values.allSatisfy({ $0.isFinite }), gamma.allSatisfy({ $0.isFinite }) else {
            throw GGUFError.invalid("qwen35moe nonfinite RMS operand")
        }
        var output = Array(repeating: Float.zero, count: values.count)
        for start in stride(from: 0, to: values.count, by: columns) {
            var total = 0.0
            for column in 0..<columns {
                let value = values[start + column]
                total += Double(value * value)
            }
            let mean = Float(total / Double(columns))
            let scale: Float = 1 / sqrt(mean + eps)
            guard scale.isFinite, scale > 0 else { throw GGUFError.invalid("qwen35moe RMS scale") }
            for column in 0..<columns {
                let scaled = values[start + column] * scale
                output[start + column] = scaled * gamma[column]
            }
        }
        guard output.allSatisfy({ $0.isFinite }) else { throw GGUFError.invalid("qwen35moe nonfinite RMS result") }
        return MLXArray(output, x.shape)
    }
}

/// The three normalisations this model uses. They differ in ways that matter.
public enum K3Norm {

    /// `KimiRMSNorm.forward`, `modeling_kimi_k3_linear.py:268-272`:
    ///
    /// ```text
    /// x = hidden.float(); x = x * rsqrt(mean(x^2, -1) + eps); return weight * x
    /// ```
    ///
    /// The epsilon is added to the *mean of squares*, inside the reciprocal
    /// square root, and the weight multiplies afterwards.
    public static func rms(_ x: MLXArray, weight: MLXArray, eps: Float) -> MLXArray {
        let x32 = x.asType(.float32)
        let normed = x32 * rsqrt(mean(x32 * x32, axis: -1, keepDims: true) + eps)
        return weight * normed
    }

    /// `FusedRMSNormGated`, transcribed from fla's Triton kernel
    /// (`fla/modules/fused_norm_gate.py:82-104`): **normalise, then gate**.
    ///
    /// ```text
    /// rmsnorm(x) * weight * sigmoid(g)
    /// ```
    ///
    /// The other plausible reading, `rmsnorm(x * sigmoid(g))`, is what fla's
    /// *other* gated norm does. The A100 cross-check separated the two by
    /// 5.4e-01 against a 1.3e-07 agreement for this one (M6A addendum, item 8),
    /// so this is measured rather than read.
    public static func gatedRMS(
        _ x: MLXArray, gate: MLXArray, weight: MLXArray, eps: Float
    ) -> MLXArray {
        let x32 = x.asType(.float32)
        var normed = x32 * rsqrt(mean(x32 * x32, axis: -1, keepDims: true) + eps)
        normed = normed * weight
        return normed * sigmoid(gate.asType(.float32))
    }

    /// Row-wise L2 normalisation, `fla/modules/l2norm.py:104-109`:
    ///
    /// ```text
    /// x / sqrt(sum(x^2, -1) + eps)
    /// ```
    ///
    /// **The epsilon is inside the square root.** The other form,
    /// `x / (sqrt(sum(x^2, -1)) + eps)`, is the one a port naturally writes, and
    /// the A100 cross-check measured it at 4.541e-05 against the shipped kernel
    /// where this form sits at 1.558e-07 — a 291× separation in the right
    /// direction, but *below* the 1e-04 per-layer tolerance. No committed
    /// fixture can fail the wrong form. `KDAEpsilonPlacementTests` is the
    /// dedicated amplified test that can (M6A addendum, item 1).
    public static func l2(_ x: MLXArray, eps: Float = K3Norm.l2Epsilon) -> MLXArray {
        let x32 = x.asType(.float32)
        return x32 * rsqrt(sum(x32 * x32, axis: -1, keepDims: true) + eps)
    }

    /// `l2norm_fwd`'s default, `fla/modules/l2norm.py:149`.
    public static let l2Epsilon: Float = 1e-6
}
