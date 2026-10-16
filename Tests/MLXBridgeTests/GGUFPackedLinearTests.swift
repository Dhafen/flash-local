import Foundation
import GGUFCore
import MLX
import StorageCore
import XCTest
@testable import MLXBridge

final class GGUFPackedLinearTests: XCTestCase {
    /// Nonuniform values exercise every packed subgroup/high-bit/scale field.
    private func packed(_ type: GGUFQuantization, elements: Int, seed: Int = 0) -> Data {
        var bytes: [UInt8] = []
        for block in 0..<(elements / Int(type.blockElements)) {
            switch type {
            case .f32:
                var bits = (Float((block + seed) % 31 - 15) / 128).bitPattern.littleEndian
                withUnsafeBytes(of: &bits) { bytes.append(contentsOf: $0) }
            case .f16, .bf16:
                let value = Float((block + seed) % 31 - 15) / 128
                let bits = type == .f16 ? Float16(value).bitPattern : UInt16(value.bitPattern >> 16)
                bytes.append(UInt8(truncatingIfNeeded: bits)); bytes.append(UInt8(truncatingIfNeeded: bits >> 8))
            case .q4_0, .q5_0, .q8_0:
                bytes += [0, 0x20] // d = 1/128
                if type == .q5_0 { bytes += [0x59, 0xa6, UInt8(truncatingIfNeeded: block + seed), 0xf1] }
                let n = type == .q8_0 ? 32 : 16
                bytes += (0..<n).map { UInt8(truncatingIfNeeded: $0 * 37 + block * 11 + seed) }
            case .q4_K:
                bytes += [0, 0x14, 0, 0x10] // small nonzero d and dmin
                bytes += (0..<12).map { UInt8(truncatingIfNeeded: $0 * 43 + block + seed) }
                bytes += (0..<128).map { UInt8(truncatingIfNeeded: $0 * 19 + block * 7 + seed) }
            case .q6_K:
                bytes += (0..<192).map { UInt8(truncatingIfNeeded: $0 * 53 + block * 7 + seed) }
                bytes += (0..<16).map { UInt8(bitPattern: Int8(($0 + block + seed) % 15 - 7)) }
                bytes += [0, 0x14]
            }
        }
        return Data(bytes)
    }

    func testEveryPublishedQuantizationMatchesCanonicalDequantizationAndIndependentDots() throws {
        let columns = 512, rows = 7, tokens = 3
        let values = (0..<(tokens * columns)).map { Float(($0 * 17) % 67 - 33) / 64 }
        let x = MLXArray(values, [tokens, columns])
        for type in GGUFQuantization.allCases {
            let bytes = packed(type, elements: rows * columns)
            var decoded = Array(repeating: Float.nan, count: rows * columns)
            let elements = decoded.count
            try decoded.withUnsafeMutableBufferPointer { output in
                try bytes.withUnsafeBytes { try GGUFDequantizer.decode($0, type: type, elements: elements, into: output) }
            }
            let input = bytes.withUnsafeBytes { MLXArray($0.bindMemory(to: UInt8.self), [bytes.count]) }
            let actual = try GGUFPackedLinear.run(x, packed: input, type: type, rows: rows, columns: columns).asArray(Float.self)
            for t in 0..<tokens {
                for row in 0..<rows {
                    var target = 0.0
                    for c in 0..<columns { target += Double(values[t*columns+c]) * Double(decoded[row*columns+c]) }
                    XCTAssertEqual(Double(actual[t*rows+row]), target, accuracy: 5e-5, "\(type) t=\(t) row=\(row)")
                }
            }
        }
    }

    func testStackedExpertsUseTheirOwnWeightsAndActivationRows() throws {
        let columns = 256, rows = 3, experts = 3
        for type in GGUFQuantization.allCases {
            let data = (0..<experts).map { packed(type, elements: columns * rows, seed: $0 * 19) }
            let arrays = data.map { d in d.withUnsafeBytes { MLXArray($0.bindMemory(to: UInt8.self), [d.count]) } }
            let x = MLXArray((0..<(columns*experts)).map { Float(($0 * 11) % 59 - 29) / 64 }, [experts, columns])
            let together = try GGUFPackedLinear.run(x, packed: concatenated(arrays), type: type,
                rows: rows, columns: columns, stackedWeights: true).asArray(Float.self)
            var separately: [Float] = []
            for e in 0..<experts {
                separately += try GGUFPackedLinear.run(x[e..<(e+1)], packed: arrays[e], type: type,
                    rows: rows, columns: columns).asArray(Float.self)
            }
            XCTAssertEqual(together, separately, "Distinct expert/activation pairing must not broadcast the wrong row")
        }
    }

    func testPackedShapeAndByteBudgetFailBeforeKernelDispatch() throws {
        let x = MLXArray.zeros([1, 256])
        XCTAssertThrowsError(try GGUFPackedLinear.run(x, packed: MLXArray([UInt8(0)]), type: .q4_K, rows: 1, columns: 256))
        XCTAssertThrowsError(try GGUFPackedLinear.run(x, packed: MLXArray.zeros([144]), type: .q4_K, rows: 1, columns: 256))
        XCTAssertThrowsError(try GGUFPackedLinear.run(x, packed: MLXArray.zeros([144], dtype: .uint8), type: .q4_K, rows: 1, columns: 255))
    }

    func testResidentAdmissionProtectsDenseBytesAndRejectsIdentityAliasing() throws {
        let cache = try GGUFPackedTileCache(capacityBytes: 8, admittedSources: [1])
        let a = RegionDescriptor(identity: .init(sourceID: 1, index: 0), offset: 10, length: 4)
        let b = RegionDescriptor(identity: .init(sourceID: 2, index: 0), offset: 20, length: 4)
        let values = MLXArray([UInt8(1), 2, 3, 4])
        try cache.insert(values, region: a); try cache.insert(values, region: b)
        XCTAssertNil(try cache.lookup(b))
        XCTAssertEqual(try cache.lookup(a)?.asArray(UInt8.self), [1, 2, 3, 4])
        XCTAssertEqual(cache.statistics.residentPayloadBytes, 4)
        XCTAssertEqual(cache.statistics.reusedPayloadBytes, 4)
        XCTAssertThrowsError(try cache.lookup(.init(identity: a.identity, offset: 11, length: 4)))
        XCTAssertThrowsError(try cache.insert(MLXArray([UInt8](repeating: 0, count: 9)),
            region: .init(identity: .init(sourceID: 1, index: 1), offset: 30, length: 9)))
        cache.clear(); XCTAssertEqual(cache.statistics.residentPayloadBytes, 0)
    }

    func testExpertLRUTouchesHitsWithoutEvictingProtectedDensePayload() throws {
        let cache = try GGUFPackedTileCache(capacityBytes: 12, admittedSources: [1],
            evictableSources: [2], evictableCapacityBytes: 8)
        func region(_ source: UInt64, _ index: UInt64) -> RegionDescriptor {
            .init(identity: .init(sourceID: source, index: index), offset: index * 4, length: 4)
        }
        let dense = region(1, 0), a = region(2, 1), b = region(2, 2), c = region(2, 3)
        let bytes = MLXArray([UInt8(1), 2, 3, 4])
        try cache.insert(bytes, region: dense); try cache.insert(bytes, region: a); try cache.insert(bytes, region: b)
        XCTAssertNotNil(try cache.lookup(a)) // a is newer than b
        try cache.insert(bytes, region: c)
        XCTAssertNil(try cache.lookup(b))
        XCTAssertEqual(try cache.lookup(dense)?.asArray(UInt8.self), [1, 2, 3, 4])
        XCTAssertNotNil(try cache.lookup(a)); XCTAssertNotNil(try cache.lookup(c))
        XCTAssertEqual(cache.statistics.residentPayloadBytes, 12)
        XCTAssertEqual(cache.statistics.evictablePayloadBytes, 8)
        XCTAssertEqual(cache.statistics.evictions, 1)
        XCTAssertThrowsError(try cache.lookup(.init(identity: a.identity, offset: 99, length: 4)))
        cache.clear()
        XCTAssertEqual(cache.statistics.residentPayloadBytes, 0)
        XCTAssertEqual(cache.statistics.evictablePayloadBytes, 0)
        try cache.insert(bytes, region: c)
        XCTAssertEqual(try cache.lookup(c)?.asArray(UInt8.self), [1, 2, 3, 4])
        XCTAssertThrowsError(try GGUFPackedTileCache(capacityBytes: 12, admittedSources: [1],
            evictableSources: [1], evictableCapacityBytes: 8))
    }

    /// Synthetic hosted-device evidence only. Timings are observations, never
    /// a speed assertion or a claim about an iPhone's flash/thermal behavior.
    func testReproduciblePackedVersusFP32ProjectionBenchmark() throws {
        let rows = 512, columns = 2560, repeats = 3
        let x = MLXArray((0..<columns).map { Float(($0 * 17) % 67 - 33) / 64 }, [1, columns])
        for type in [GGUFQuantization.q4_K, .q5_0, .q8_0] {
            let data = packed(type, elements: rows * columns)
            let input = data.withUnsafeBytes { MLXArray($0.bindMemory(to: UInt8.self), [data.count]) }
            var decoded = Array(repeating: Float(0), count: rows * columns)
            func legacy() throws -> MLXArray {
                let n = decoded.count
                try decoded.withUnsafeMutableBufferPointer { output in
                    try data.withUnsafeBytes { try GGUFDequantizer.decode($0, type: type, elements: n, into: output) }
                }
                let y = matmul(x, MLXArray(decoded, [rows, columns]).T); y.eval(); return y
            }
            func direct() throws -> MLXArray {
                let y = try GGUFPackedLinear.run(x, packed: input, type: type, rows: rows, columns: columns); y.eval(); return y
            }
            let old = try legacy().asArray(Float.self), new = try direct().asArray(Float.self)
            for (a,b) in zip(old,new) { XCTAssertEqual(a,b,accuracy: 5e-4) }
            var cpu: [Double] = [], gpu: [Double] = []
            for _ in 0..<repeats {
                var start = ProcessInfo.processInfo.systemUptime; _ = try legacy(); cpu.append(ProcessInfo.processInfo.systemUptime-start)
                start = ProcessInfo.processInfo.systemUptime; _ = try direct(); gpu.append(ProcessInfo.processInfo.systemUptime-start)
            }
            let report: [String: Any] = ["kind":"synthetic-projection", "quant":String(describing:type),
                "rows":rows,"columns":columns,"repeats":repeats,"warmup":1,"packedBytes":data.count,
                "fp32TileBytes":rows*columns*4,"cpuDequantizeCopyMatmulSeconds":cpu,
                "residentPackedMetalSeconds":gpu,"os":ProcessInfo.processInfo.operatingSystemVersionString,
                "physicalMemoryBytes":ProcessInfo.processInfo.physicalMemory,
                "sourceRevision":BuildInfo.gitRevision,"physicalPhone":false,
                "excludesFileIO":true,"excludesCanonicalActivationRounding":true]
            print("GGUF_PACKED_PROJECTION_BENCH " + String(decoding:try JSONSerialization.data(withJSONObject:report,options:[.sortedKeys]),as:UTF8.self))
        }
    }
}
