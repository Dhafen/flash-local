import Foundation
import GGUFCore
import MLX
import MinirunKit
import XCTest
@testable import ModelAdapters

final class GGUFWeightSourceTests: XCTestCase {
    private func withSource(residentCacheBytes: Int = 0, _ body: (GGUFWeightSource) throws -> Void) throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        var header = Data("GGUF".utf8)
        func put(_ value: UInt64, _ count: Int) {
            for i in 0..<count { header.append(UInt8(truncatingIfNeeded: value >> (i * 8))) }
        }
        put(3, 4); put(5, 8); put(0, 8)
        let tensors: [(String, [UInt64], UInt64, UInt64)] = [
            ("experts", [4, 5, 3], 0, 0), ("embedding", [4, 3], 256, 0), ("norm", [4], 320, 0),
            ("embedding_peer", [4, 3], 352, 0), ("half_peer", [4, 3], 416, 1)]
        for (name, dimensions, offset, type) in tensors {
            let bytes = Data(name.utf8); put(UInt64(bytes.count), 8); header.append(bytes)
            put(UInt64(dimensions.count), 4)
            for dimension in dimensions { put(dimension, 8) }
            put(type, 4); put(offset, 8)
        }
        header.append(Data(repeating: 0, count: ((header.count + 31) / 32 * 32) - header.count))
        var payload = Data(count: 440)
        func store(_ value: Float, offset: Int) {
            var bits = value.bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { payload.replaceSubrange(offset..<(offset + 4), with: $0) }
        }
        for i in 0..<60 { store(Float(i), offset: i * 4) }
        for i in 0..<12 { store(Float(i + 1), offset: 256 + i * 4) }
        for i in 0..<4 { store(Float(i + 1), offset: 320 + i * 4) }
        for i in 0..<12 {
            store(Float(i + 21), offset: 352 + i * 4)
            var half = Float16(i + 41).bitPattern.littleEndian
            withUnsafeBytes(of: &half) { payload.replaceSubrange((416 + i * 2)..<(418 + i * 2), with: $0) }
        }
        header.append(payload)
        let manifest = try GGUFModelManifest(fileName: "model.fixture", fileByteCount: UInt64(header.count),
                                             sha256: FileDigestComputer.sha256(of: header))
        try header.write(to: root.appendingPathComponent(manifest.fileName))
        let artifact = try VerifiedGGUFArtifact.verify(insideRoot: root, manifest: manifest)
        let source = try GGUFWeightSource(artifact: artifact, rawCacheBytes: 8192, tileBytes: 4096,
            noCache: false, residentCacheBytes: residentCacheBytes)
        defer { source.shutdown() }
        try body(source)
    }

    func testGroupedRoutingMatchesSeparateExpertsWithDistinctActivationsAndRanges() throws {
        try withSource { source in
            let x = MLXArray([Float(1), 2, 3, 4], [1, 4])
            let single = try [UInt64(2), 0].map { try source.linear(x, tensor: "experts", expert: $0).asArray(Float.self) }
            let together = try source.linearExperts(x, tensor: "experts", experts: [2, 0]).asArray(Float.self)
            XCTAssertEqual(together, single.flatMap { $0 })
            let distinct = MLXArray([Float(1), 2, 3, 4, 4, 3, 2, 1], [2, 4])
            let actual = try source.linearExperts(distinct, tensor: "experts", experts: [2, 0]).asArray(Float.self)
            XCTAssertEqual(actual, [420, 460, 500, 540, 580, 10, 50, 90, 130, 170])
            XCTAssertThrowsError(try source.linearExperts(x, tensor: "experts", experts: [3]))
            XCTAssertThrowsError(try source.linearExperts(x, tensor: "experts", experts: []))
        }
    }

    func testDenseResidentCacheSurvivesRoutedExpertChurnWithinItsDeclaredCapacity() throws {
        try withSource(residentCacheBytes: 48) { source in
            let x = MLXArray([Float(1), 2, 3, 4], [1, 4])
            let original = try source.linear(x, tensor: "embedding").asArray(Float.self)
            for e in [UInt64(0), 1, 2] { _ = try source.linear(x, tensor: "experts", expert: e) }
            let before = source.readStatistics.bytesRead
            XCTAssertEqual(try source.linear(x, tensor: "embedding").asArray(Float.self), original)
            XCTAssertEqual(source.readStatistics.bytesRead, before)
            XCTAssertEqual(source.residentCacheStatistics.residentPayloadBytes, 48)
            XCTAssertEqual(source.residentCacheStatistics.reusedPayloadBytes, 48)
            XCTAssertEqual(source.projectionStatistics.residentWeightBytesReused, 48)
            source.shutdown()
            XCTAssertEqual(source.residentCacheStatistics.residentPayloadBytes, 0)
        }
    }

    func testFirstBuildProjectionReportsRemainReadable() throws {
        let report = Data("{\"calls\":1,\"tiles\":1,\"logicalWeightBytes\":16,\"wallSeconds\":1,\"waitOnIOSeconds\":0.1,\"dequantizeAndCopySeconds\":0.2,\"computeSeconds\":0.3,\"activationRoundingCalls\":0,\"activationRoundingSeconds\":0}".utf8)
        let old = try JSONDecoder().decode(GGUFProjectionStatistics.self, from: report)
        XCTAssertNil(old.packedMetalCalls)
        XCTAssertNil(old.canonicalCPUKVCalls)
        XCTAssertNil(old.pairedProjectionBatches)
        XCTAssertEqual(old.calls, 1)
    }

    func testDensePairCountsLogicalWorkOncePerMatrixAndReusesTheExistingCache() throws {
        try withSource(residentCacheBytes: 96) { source in
            let x = MLXArray([Float(1), 2, 3, 4], [1, 4])
            let first = try source.linearPair(x, left: "embedding", right: "embedding_peer")
            XCTAssertEqual(first.0.asArray(Float.self), [30,70,110])
            XCTAssertEqual(first.1.asArray(Float.self), [230,270,310])
            let initial = source.projectionStatistics
            XCTAssertEqual(initial.calls, 2)
            XCTAssertEqual(initial.packedMetalCalls, 2)
            XCTAssertEqual(initial.pairedProjectionBatches, 1)
            XCTAssertEqual(initial.evaluationBatches, 1)
            let io = source.readStatistics.bytesRead
            let again = try source.linearPair(x, left: "embedding", right: "embedding_peer")
            XCTAssertEqual(again.0.asArray(Float.self).map(\.bitPattern), first.0.asArray(Float.self).map(\.bitPattern))
            XCTAssertEqual(again.1.asArray(Float.self).map(\.bitPattern), first.1.asArray(Float.self).map(\.bitPattern))
            XCTAssertEqual(source.readStatistics.bytesRead, io)
            let work = try source.projectionStatistics.subtracting(initial)
            XCTAssertEqual(work.calls, 2); XCTAssertEqual(work.pairedProjectionBatches, 1)
            XCTAssertEqual(work.logicalWeightBytes, 96)
            XCTAssertEqual(work.residentWeightBytesReused, 96)
            XCTAssertEqual(work.packedWeightCopyBytes, 0)
            let before = source.projectionStatistics
            let control = try source.linearPair(x, left: "embedding", right: "embedding_peer", enabled: false)
            XCTAssertEqual(control.0.asArray(Float.self), first.0.asArray(Float.self))
            XCTAssertEqual(control.1.asArray(Float.self), first.1.asArray(Float.self))
            let separate = try source.projectionStatistics.subtracting(before)
            XCTAssertEqual(separate.calls, 2); XCTAssertEqual(separate.pairedProjectionBatches, 0)
            XCTAssertEqual(separate.evaluationBatches, 2)
            XCTAssertEqual(try JSONDecoder().decode(GGUFProjectionStatistics.self,
                from: JSONEncoder().encode(source.projectionStatistics)), source.projectionStatistics)
            XCTAssertThrowsError(try initial.subtracting(source.projectionStatistics))
            source.shutdown()
            XCTAssertEqual(first.0.asArray(Float.self), [30,70,110])
            XCTAssertThrowsError(try source.linearPair(x, left: "embedding", right: "embedding_peer"))
        }
    }

    func testPairDifferentFormatsAndRepeatedTensorsKeepTheSeparatePath() throws {
        try withSource { source in
            let x = MLXArray([Float(1), 2, 3, 4], [1, 4])
            let left = try source.linear(x, tensor: "embedding").asArray(Float.self)
            let right = try source.linear(x, tensor: "half_peer").asArray(Float.self)
            let before = source.projectionStatistics
            let pair = try source.linearPair(x, left: "embedding", right: "half_peer")
            XCTAssertEqual(pair.0.asArray(Float.self), left)
            XCTAssertEqual(pair.1.asArray(Float.self), right)
            let work = try source.projectionStatistics.subtracting(before)
            XCTAssertEqual(work.calls, 2); XCTAssertEqual(work.pairedProjectionBatches, 0)
            XCTAssertEqual(work.activationRoundingCalls, 1)
            let repeated = try source.linearPair(x, left: "embedding", right: "embedding")
            XCTAssertEqual(repeated.0.asArray(Float.self), left)
            XCTAssertEqual(repeated.1.asArray(Float.self), left)
            let io = source.readStatistics.bytesRead
            XCTAssertThrowsError(try source.linearPair(x, left: "absent", right: "embedding"))
            XCTAssertEqual(source.readStatistics.bytesRead, io)
        }
    }

    func testSmallCanonicalKVProjectionSharesProtectedCacheAndRejectsOtherShapes() throws {
        try withSource(residentCacheBytes: 48) { source in
            let x = MLXArray([Float(1), 2, 3, 4], [1, 4])
            let first = try source.linearCanonicalKV(x, tensor: "embedding")
            XCTAssertEqual(first.asArray(Float.self), [30, 70, 110])
            XCTAssertEqual(source.readStatistics.bytesRead, 48)
            let before = source.projectionStatistics
            let again = try source.linearCanonicalKV(x, tensor: "embedding")
            XCTAssertEqual(again.asArray(Float.self), [30, 70, 110])
            XCTAssertEqual(source.readStatistics.bytesRead, 48)
            let work = try source.projectionStatistics.subtracting(before)
            XCTAssertEqual(work.canonicalCPUKVCalls, 1)
            XCTAssertEqual(work.packedMetalCalls, 0)
            XCTAssertEqual(work.residentWeightBytesReused, 48)
            XCTAssertEqual(work.packedWeightCopyBytes, 48, "Explicit CPU handoff is counted even for cached weights")
            XCTAssertThrowsError(try before.subtracting(source.projectionStatistics))
            XCTAssertThrowsError(try source.linearCanonicalKV(x, tensor: "experts"))
            XCTAssertThrowsError(try source.linearCanonicalKV(MLXArray.zeros([2, 4]), tensor: "embedding"))
            XCTAssertThrowsError(try source.linearCanonicalKV(MLXArray.zeros([1, 5]), tensor: "embedding"))
            XCTAssertEqual(source.readStatistics.bytesRead, 48)
            source.shutdown()
            XCTAssertEqual(first.asArray(Float.self), [30, 70, 110], "Output owns its reusable scratch copy")
            XCTAssertThrowsError(try source.linearCanonicalKV(x, tensor: "embedding"))
        }
    }

    func testVerifiedExpertAndEmbeddingReadsShareBoundedCache() throws {
        try withSource { source in
            let x = MLXArray([Float(1), 2, 3, 4], [1, 4])
            let first = try source.linear(x, tensor: "experts", expert: 2)
            XCTAssertEqual(first.asArray(Float.self), [420, 460, 500, 540, 580])
            XCTAssertEqual(source.readStatistics.bytesRead, 80)
            let firstWork = source.projectionStatistics
            let cached = try source.linear(x, tensor: "experts", expert: 2)
            XCTAssertEqual(cached.asArray(Float.self), first.asArray(Float.self))
            XCTAssertEqual(source.readStatistics.bytesRead, 80)
            let work = source.projectionStatistics
            XCTAssertEqual(work.calls, 2); XCTAssertEqual(work.tiles, 2)
            XCTAssertEqual(work.logicalWeightBytes, 160, "Cache hits still perform logical matrix work")
            XCTAssertGreaterThanOrEqual(source.poolStatistics.cacheHitCount, 1)
            let secondWork = try work.subtracting(firstWork)
            XCTAssertEqual(secondWork.calls, 1); XCTAssertEqual(secondWork.logicalWeightBytes, 80)
            XCTAssertThrowsError(try firstWork.subtracting(work))
            XCTAssertGreaterThanOrEqual(work.wallSeconds, work.waitOnIOSeconds)
            XCTAssertGreaterThanOrEqual(work.wallSeconds, work.dequantizeAndCopySeconds)
            XCTAssertGreaterThanOrEqual(work.wallSeconds, work.computeSeconds)
            XCTAssertEqual(try JSONDecoder().decode(GGUFProjectionStatistics.self, from: JSONEncoder().encode(work)), work)
            XCTAssertEqual(try source.row(1, tensor: "embedding").asArray(Float.self), [5, 6, 7, 8])
            XCTAssertEqual(try source.smallTensor("norm").asArray(Float.self), [1, 2, 3, 4])
            XCTAssertEqual(source.rawCacheBytes, 8192)
            XCTAssertLessThanOrEqual(source.poolStatistics.peakInUseBytes, 8192)
        }
    }

    func testMissingTensorBadExpertAndClosedSourceFailWithoutReads() throws {
        try withSource { source in
            let x = MLXArray([Float(1), 2, 3, 4], [1, 4])
            XCTAssertThrowsError(try source.linear(x, tensor: "absent"))
            XCTAssertThrowsError(try source.linear(x, tensor: "experts", expert: 3))
            XCTAssertThrowsError(try source.row(3, tensor: "embedding"))
            XCTAssertThrowsError(try source.row(0, tensor: "experts"))
            XCTAssertEqual(source.readStatistics.bytesRead, 0)
            XCTAssertEqual(source.projectionStatistics.calls, 0)
            source.shutdown()
            XCTAssertThrowsError(try source.linear(x, tensor: "embedding"))
            source.shutdown() // idempotent
        }
    }

    func testCanonicalActivationRoundingUsesWeightSpecificCPUPrecisionAndOwnsCopies() throws {
        try withSource { source in
            let input: [Float] = [1, 0.49, 0.25, -1, 0.125] + Array(repeating: 0, count: 251)
            let x = MLXArray(input, [1, 256])
            let q8Step = Float(Float16(Float(1) / 127))
            for type in [GGUFQuantization.q4_0, .q5_0, .q8_0] {
                let actual = try source.canonicalActivations(x, type: type, columns: 256).asArray(Float.self)
                for (a, q) in zip(actual.prefix(5), [Float(127), 62, 32, -127, 16]) {
                    XCTAssertEqual(a, q * q8Step, accuracy: 1e-7)
                }
            }
            for type in [GGUFQuantization.q4_K, .q6_K] {
                let actual = try source.canonicalActivations(x, type: type, columns: 256).asArray(Float.self)
                for (a, q) in zip(actual.prefix(5), [Float(127), 62, 32, -127, 16]) {
                    XCTAssertEqual(a, q / 127, accuracy: 1e-7)
                }
            }
            let small: [Float] = [1.0001, 1.004, -0.3333, 0]
            let smallX = MLXArray(small, [1, 4])
            let half = try source.canonicalActivations(smallX, type: .f16, columns: 4)
            XCTAssertEqual(half.asArray(Float.self), small.map { Float(Float16($0)) })
            let bf = try source.canonicalActivations(smallX, type: .bf16, columns: 4).asArray(Float.self)
            XCTAssertEqual(bf[0], 1)
            XCTAssertEqual(bf[1], 1.0078125)
            // Reusing scratch for another projection must not mutate the first.
            _ = try source.canonicalActivations(MLXArray.zeros([1, 4]), type: .f16, columns: 4)
            XCTAssertEqual(half.asArray(Float.self), small.map { Float(Float16($0)) })
            XCTAssertThrowsError(try source.canonicalActivations(smallX, type: .q6_K, columns: 4))
            XCTAssertThrowsError(try source.canonicalActivations(MLXArray([Float.nan], [1, 1]), type: .f16, columns: 1))
            XCTAssertThrowsError(try source.canonicalActivations(MLXArray.zeros([2, 65536]), type: .f16, columns: 65536))
            XCTAssertEqual(source.readStatistics.bytesRead, 0, "Activation rounding performs no weight I/O")
        }
    }
}
