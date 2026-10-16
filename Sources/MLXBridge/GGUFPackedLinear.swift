import Foundation
import GGUFCore
import MLX
import StorageCore

/// Both paths use the original GGUF bytes and the caller's canonical rounded
/// activations. The older path remains an independent numerical/benchmark
/// control; production can multiply the packed bytes without a CPU FP32 tile.
public enum GGUFLinearExecution: String, Codable, Sendable {
    case packedMetal, dequantized
}

/// A fixed admission list prevents a sequential sweep of cold expert weights
/// from evicting the dense matrices used at every position. Payload capacity is
/// explicit; allocator overhead is additionally covered by the process reserve
/// and live footprint checks in the runner. Confined to its weight executor.
public final class GGUFPackedTileCache {
    public struct Statistics: Codable, Equatable, Sendable {
        public let capacityBytes: Int
        public let residentPayloadBytes: Int
        public let entries: Int
        public let hits: UInt64
        public let reusedPayloadBytes: UInt64
    }
    private struct Entry {
        let region: RegionDescriptor
        let array: MLXArray
    }
    public let capacityBytes: Int
    private let admittedSources: Set<UInt64>
    private var entries: [RegionIdentity: Entry] = [:]
    private var bytes = 0
    private var hits: UInt64 = 0
    private var reused: UInt64 = 0

    public init(capacityBytes: Int, admittedSources: Set<UInt64>) throws {
        guard capacityBytes >= 0, capacityBytes <= 4_500_000_000,
              admittedSources.count <= 4096 else { throw GGUFError.limit("resident packed cache") }
        self.capacityBytes = capacityBytes; self.admittedSources = admittedSources
    }

    public var statistics: Statistics {
        Statistics(capacityBytes: capacityBytes, residentPayloadBytes: bytes,
            entries: entries.count, hits: hits, reusedPayloadBytes: reused)
    }

    func contains(_ region: RegionDescriptor) throws -> Bool {
        guard let entry = entries[region.identity] else { return false }
        guard entry.region == region else { throw GGUFError.invalid("resident cache identity changed range") }
        return true
    }

    func lookup(_ region: RegionDescriptor) throws -> MLXArray? {
        guard let entry = entries[region.identity] else { return nil }
        guard entry.region == region else { throw GGUFError.invalid("resident cache identity changed range") }
        let nextHits = hits.addingReportingOverflow(1)
        let nextReused = reused.addingReportingOverflow(UInt64(region.length))
        guard !nextHits.overflow, !nextReused.overflow else { throw GGUFError.limit("resident cache counters") }
        hits = nextHits.partialValue; reused = nextReused.partialValue
        return entry.array
    }

    func insert(_ array: MLXArray, region: RegionDescriptor) throws {
        guard admittedSources.contains(region.identity.sourceID) else { return }
        guard array.dtype == .uint8, array.size == region.length,
              region.length <= capacityBytes - bytes, entries.count < 16384,
              entries[region.identity] == nil else { throw GGUFError.limit("resident packed tile admission") }
        entries[region.identity] = Entry(region: region, array: array)
        bytes += region.length
    }

    public func clear() { entries.removeAll(); bytes = 0 }
}

/// Fused GGML dequantization and FP32 dot product: one SIMD group per row.
/// Quantization layouts follow the vendored llama.cpp e467e5b arithmetic in
/// CGGUFQuants (MIT; see LICENSE.llama.cpp). No affine reinterpretation,
/// requantization, model conversion or lower-precision accumulator is used.
public enum GGUFPackedLinear {
    private static let kernel = MLXFast.metalKernel(
        name: "flash_local_gguf_packed_linear_v1",
        inputNames: ["packed", "activation"], outputNames: ["out"],
        source: """
        uint lane = thread_position_in_grid.x;
        uint row = thread_position_in_grid.y;
        uint token = thread_position_in_grid.z;
        if (row >= ROWS || token >= TOKENS) return;
        device const uchar* p = packed + (STACKED ? token * ROWS * ROW_BYTES : 0) + row * ROW_BYTES;
        float acc = 0.0f;
        for (uint col = lane; col < COLS; col += 32) {
            float w;
            if (FORMAT == 0) {
                uint at = 4 * col;
                uint bits = uint(p[at]) | (uint(p[at+1]) << 8)
                    | (uint(p[at+2]) << 16) | (uint(p[at+3]) << 24);
                w = as_type<float>(bits);
            } else if (FORMAT == 1 || FORMAT == 30) {
                uint at = 2 * col;
                ushort bits = ushort(p[at]) | (ushort(p[at+1]) << 8);
                w = FORMAT == 1 ? float(as_type<half>(bits)) : as_type<float>(uint(bits) << 16);
            } else if (FORMAT == 2 || FORMAT == 6 || FORMAT == 8) {
                uint block_bytes = FORMAT == 2 ? 18 : (FORMAT == 6 ? 22 : 34);
                device const uchar* b = p + (col / 32) * block_bytes;
                float d = float(as_type<half>(ushort(uint(b[0]) | (uint(b[1]) << 8))));
                uint j = col % 32;
                if (FORMAT == 8) {
                    w = d * float(as_type<char>(b[2+j]));
                } else {
                    uint q = b[(FORMAT == 2 ? 2 : 6) + j % 16];
                    q = (q >> (j < 16 ? 0 : 4)) & 15;
                    if (FORMAT == 6) q |= ((uint(b[2+j/8]) >> (j % 8)) & 1) << 4;
                    w = d * float(int(q) - (FORMAT == 2 ? 8 : 16));
                }
            } else if (FORMAT == 12) {
                device const uchar* b = p + (col / 256) * 144;
                float d = float(as_type<half>(ushort(uint(b[0]) | (uint(b[1]) << 8))));
                float m = float(as_type<half>(ushort(uint(b[2]) | (uint(b[3]) << 8))));
                uint j = col % 256, group = j / 32;
                device const uchar* s = b + 4;
                uint scale = group < 4 ? s[group] & 63
                    : (s[group+4] & 15) | ((s[group-4] >> 6) << 4);
                uint minimum = group < 4 ? s[group+4] & 63
                    : (s[group+4] >> 4) | ((s[group] >> 6) << 4);
                uint q = (b[16 + (j/64)*32 + j%32] >> ((group%2)*4)) & 15;
                // Keep the same non-fused weight reconstruction as the CPU.
                float scaled = d * float(scale);
                w = scaled * float(q) - m * float(minimum);
            } else {
                device const uchar* b = p + (col / 256) * 210;
                uint j = col % 256, half_block = j / 128, part = (j%128) / 32, l = j%32;
                uint low = b[half_block*64 + (part%2)*32 + l];
                low = (low >> (part < 2 ? 0 : 4)) & 15;
                uint high = (b[128 + half_block*32 + l] >> (part*2)) & 3;
                int q = int(low | (high << 4)) - 32;
                int scale = int(as_type<char>(b[192 + half_block*8 + part*2 + l/16]));
                float d = float(as_type<half>(ushort(uint(b[208]) | (uint(b[209]) << 8))));
                w = (d * float(scale)) * float(q);
            }
            acc += w * activation[token * COLS + col];
        }
        float value = simd_sum(acc);
        if (lane == 0) out[token * ROWS + row] = value;
        """)

    public static func run(_ x: MLXArray, packed: MLXArray, type: GGUFQuantization,
                           rows: Int, columns: Int, stackedWeights: Bool = false) throws -> MLXArray {
        guard x.ndim == 2, x.shape[0] > 0, x.dtype == .float32, x.shape[1] == columns,
              rows > 0, columns > 0, columns <= 65536,
              columns % Int(type.blockElements) == 0,
              packed.dtype == .uint8, packed.ndim == 1 else { throw GGUFError.invalid("packed Metal linear shape") }
        let rowBytes = columns / Int(type.blockElements) * Int(type.blockBytes)
        let matrixBytes = rows.multipliedReportingOverflow(by: rowBytes)
        let count = matrixBytes.partialValue.multipliedReportingOverflow(by: stackedWeights ? x.shape[0] : 1)
        let outputCount = rows.multipliedReportingOverflow(by: x.shape[0])
        guard !matrixBytes.overflow, !count.overflow, packed.size == count.partialValue,
              packed.size <= (stackedWeights ? 16 << 20 : 4 << 20),
              !outputCount.overflow, outputCount.partialValue <= 16 << 20 else {
            throw GGUFError.limit("packed Metal tile/output")
        }
        return kernel([packed, x], template: [("FORMAT", Int(type.rawValue)),
            ("ROWS", rows), ("COLS", columns), ("TOKENS", x.shape[0]), ("ROW_BYTES", rowBytes),
            ("STACKED", stackedWeights)],
            grid: (32, rows, x.shape[0]), threadGroup: (32, 4, 1),
            outputShapes: [[x.shape[0], rows]], outputDTypes: [.float32])[0]
    }
}
