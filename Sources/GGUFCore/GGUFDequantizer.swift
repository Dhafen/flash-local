import Foundation
import CGGUFQuants

public enum GGUFDequantizer {
    public static let maximumTileElements = 4 * 1024 * 1024

    /// Decode into the caller's reusable storage. The operation allocates no
    /// model-sized array and never changes the artifact's quantization.
    public static func decode(
        _ packed: UnsafeRawBufferPointer, type: GGUFQuantization,
        elements: Int, into output: UnsafeMutableBufferPointer<Float>
    ) throws {
        guard elements > 0, elements <= maximumTileElements, output.count >= elements else {
            throw GGUFError.limit("dequantization tile")
        }
        let code = flash_dequantize(type.rawValue, packed.baseAddress, packed.count,
                                    output.baseAddress, output.count, elements)
        guard code == 0 else { throw GGUFError.invalid("packed tile geometry/alignment (\(code))") }
    }
}
