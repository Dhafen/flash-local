import Foundation
import XCTest
@testable import OrnithPerformance

final class SourceAndMemoryTests: XCTestCase {
    func primary() throws -> QPackInstallPlan {
        try QPackInstallPlan(source: MLXSourceIndex(manifest: .ornith()))
    }

    func testActualPublishedHeadersValidateEveryLayerWithoutTensorPayloads() throws {
        let plan = try primary()
        try plan.source.geometry.requirePublishedOrnith()
        XCTAssertEqual(plan.source.tensors.count, 1757)
        XCTAssertEqual(plan.densePayloadBytes, 1_389_394_176)
        XCTAssertEqual(plan.expertPayloadBytes, 18_119_393_280)
        XCTAssertEqual(plan.layout.sections.count, 9)
        XCTAssertEqual(plan.layout.expertStride, 1_769_472)
        XCTAssertEqual(plan.layout.linearLayers.filter { $0 }.count, 30)
        XCTAssertEqual(plan.source.sourceHeaders.reduce(0) { $0 + $1.count }, 236_745)
        for layer in 0..<40 {
            XCTAssertEqual(plan.source.tensors["model.layers.\(layer).mlp.gate.weight"]?.shape, [256, 512])
            XCTAssertEqual(plan.source.tensors["model.layers.\(layer).mlp.switch_mlp.gate_proj.weight"]?.shape,
                           [256, 512, 256])
        }
        print("ORNITH_QPACK_SOURCE layers=40 experts=256 topK=8 tensors=1757 denseBytes=1389394176 expertBytes=18119393280 expertStride=1769472 routerBits=8 expertBits=4 tensorPayloadDownloaded=false")
    }

    func testAllSourceWindowsMapOnlyInsideFinalFilesAndCoverPayloadExactly() throws {
        let plan = try primary()
        var mappedBytes = 0
        for (shard, file) in plan.source.manifest.shards.enumerated() {
            var first = 0
            while first < file.byteCount {
                let end = min(first + (8 << 20), file.byteCount)
                let pieces = try plan.pieces(shard: shard, range: first..<end)
                var cursor = max(first, plan.source.sourceHeaders[shard].count)
                for piece in pieces {
                    XCTAssertEqual(piece.sourceRange.lowerBound, cursor)
                    XCTAssertGreaterThan(piece.sourceRange.count, 0)
                    XCTAssertGreaterThanOrEqual(piece.destinationOffset, 0)
                    XCTAssertLessThanOrEqual(piece.destinationOffset + piece.sourceRange.count,
                                             try XCTUnwrap(plan.fileBytes[piece.file]))
                    cursor = piece.sourceRange.upperBound
                    mappedBytes += piece.sourceRange.count
                }
                XCTAssertEqual(cursor, end)
                first = end
            }
        }
        XCTAssertEqual(mappedBytes, plan.densePayloadBytes + plan.expertPayloadBytes)
        // No raw .safetensors shard is retained alongside its QPack output.
        XCTAssertFalse(plan.fileBytes.keys.contains { $0.hasPrefix("model-0000") })
    }

    func testMixedExpertStorageInAnyLaterLayerIsRejected() throws {
        var root = try contractJSON()
        var shards = try XCTUnwrap(root["shards"] as? [[String: Any]])
        for i in shards.indices {
            let data = try XCTUnwrap(Data(base64Encoded: shards[i]["headerBase64"] as! String))
            var header = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
            let prefix = "language_model.model.layers.39.mlp.switch_mlp.gate_proj."
            guard header[prefix + "scales"] != nil else { continue }
            for part in ["scales", "biases"] {
                var tensor = header[prefix + part] as! [String: Any]
                tensor["dtype"] = "F16" // Same byte size, different affine values.
                header[prefix + part] = tensor
            }
            let updated = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
            let delta = updated.count - data.count
            shards[i]["byteCount"] = (shards[i]["byteCount"] as! Int) + delta
            shards[i]["headerBase64"] = updated.base64EncodedString()
            shards[i]["headerSHA256"] = digest(updated)
        }
        root["shards"] = shards
        let source = try MLXSourceIndex(manifest: decode(root))
        XCTAssertThrowsError(try QPackInstallPlan(source: source))
    }

    func testMissingOrOverlappingTensorAndUnsafePathsAreRejected() throws {
        var root = try contractJSON()
        var shards = root["shards"] as! [[String: Any]]
        shards[0]["name"] = "../foreign.safetensors"
        root["shards"] = shards
        XCTAssertThrowsError(try MLXSourceIndex(manifest: decode(root)))
        root = try contractJSON()
        shards = root["shards"] as! [[String: Any]]
        let original = Data(base64Encoded: shards[0]["headerBase64"] as! String)!
        var header = try JSONSerialization.jsonObject(with: original) as! [String: Any]
        header.removeValue(forKey: "language_model.model.layers.0.mlp.gate.weight")
        let updated = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        shards[0]["headerBase64"] = updated.base64EncodedString()
        shards[0]["headerSHA256"] = digest(updated)
        shards[0]["byteCount"] = (shards[0]["byteCount"] as! Int) + updated.count - original.count
        root["shards"] = shards
        XCTAssertThrowsError(try MLXSourceIndex(manifest: decode(root)))
    }

    func testMemoryBudgetAdaptsToContextAndCountsTheActualF32KV() throws {
        let install = try primary()
        var previous = UInt64.max
        for tokens in [8192, 32768, 65536] {
            let plan = try PerformanceMemoryPlan.make(install: install, contextTokens: tokens,
                baselineFootprintBytes: 100_000_000, osAvailableBytes: 6_000_000_000)
            XCTAssertLessThanOrEqual(plan.reservedTotalBytes, 5_500_000_000)
            XCTAssertLessThanOrEqual(plan.expertCacheBytes, previous)
            XCTAssertGreaterThanOrEqual(plan.expertCacheBytes, plan.minimumExpertCacheBytes)
            XCTAssertEqual(plan.gpuKVCapacityBytes, UInt64(tokens) * 40_960)
            XCTAssertEqual(plan.cpuKVMirrorReserveBytes, 0)
            previous = plan.expertCacheBytes
            print("ORNITH_MEMORY_ARITHMETIC context=\(tokens) dense=\(plan.residentDensePayloadBytes) gpuKV=\(plan.gpuKVCapacityBytes) expertCache=\(plan.expertCacheBytes) totalReserved=\(plan.reservedTotalBytes) deviceMeasurement=false")
        }
        XCTAssertThrowsError(try PerformanceMemoryPlan.make(install: install, contextTokens: 32768,
            baselineFootprintBytes: 100_000_000, osAvailableBytes: 6_000_000_000, retainCPUKVMirror: true))
    }

    func testOSAndMetalHeadroomOverrideTheRequestedCacheAndZeroDoesNotBecomeUnlimited() throws {
        let install = try primary()
        XCTAssertThrowsError(try PerformanceMemoryPlan.make(install: install, contextTokens: 8192,
            baselineFootprintBytes: 100_000_000, osAvailableBytes: 0))
        XCTAssertThrowsError(try PerformanceMemoryPlan.make(install: install, contextTokens: 8192,
            baselineFootprintBytes: 100_000_000, osAvailableBytes: nil, metalRecommendedWorkingSetBytes: 1))
        let plan = try PerformanceMemoryPlan.make(install: install, contextTokens: 8192,
            baselineFootprintBytes: 100_000_000, osAvailableBytes: 3_000_000_000)
        XCTAssertLessThanOrEqual(plan.effectiveProcessCeilingBytes, 3_100_000_000)
        XCTAssertLessThan(plan.expertCacheBytes, 1_000_000_000)
        XCTAssertEqual(plan.safeCacheLimit(currentFootprintBytes: 3_100_000_000,
            currentCacheAllocatedBytes: 200_000_000, osAvailableBytes: 0, futureStateGrowthBytes: 0), 0)
        XCTAssertThrowsError(try PerformanceMemoryPlan.make(install: install, contextTokens: .max,
            baselineFootprintBytes: 0, osAvailableBytes: nil))
    }

    func contractJSON() throws -> [String: Any] {
        let data = try JSONEncoder().encode(MLXSourceManifest.ornith())
        return try JSONSerialization.jsonObject(with: data) as! [String: Any]
    }
    func decode(_ root: [String: Any]) throws -> MLXSourceManifest {
        try JSONDecoder().decode(MLXSourceManifest.self, from: JSONSerialization.data(withJSONObject: root))
    }
}
