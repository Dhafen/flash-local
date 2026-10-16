import Darwin
import Foundation
import GGUFCore
import MLX
import StorageCore
import XCTest
@testable import MLXBridge

final class GGUFStreamedLinearTests: XCTestCase {
    private func withExpertMatrix(cache: Bool = false, _ body: (GGUFIndex, TileReader, BufferPool) throws -> Void) throws {
        var header = Data("GGUF".utf8)
        func integer(_ value: UInt64, _ bytes: Int) {
            for i in 0..<bytes { header.append(UInt8(truncatingIfNeeded: value >> (i * 8))) }
        }
        integer(3, 4); integer(1, 8); integer(0, 8)
        let name = Data("experts".utf8); integer(UInt64(name.count), 8); header.append(name)
        integer(3, 4); integer(4, 8); integer(5, 8); integer(3, 8); integer(0, 4); integer(0, 8)
        let payload = (header.count + 31) / 32 * 32
        let index = try GGUFIndex.parse(header: header, fileByteCount: UInt64(payload + 60 * 4))
        header.append(Data(repeating: 0, count: payload - header.count))
        for value in 0..<60 {
            var bits = Float(value).bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { header.append(contentsOf: $0) }
        }
        let path = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try header.write(to: path)
        defer { try? FileManager.default.removeItem(at: path) }
        let pool = try BufferPool(configuration: .init(slotCount: 2, slotBytes: 3 * 4 * 4,
                                                       retainOnRelease: cache, poisonOnRelease: true))
        let reader = try TileReader(path: path.path, pool: pool,
            configuration: .init(queueDepth: 1, readAhead: 1, noCache: false, slotAcquireTimeout: 5))
        defer { reader.shutdown() }
        try body(index, reader, pool)
    }

    func testStreamsOnlySelectedExpertWithPoisonedReleasedSlots() throws {
        try withExpertMatrix { index, reader, pool in
            let tensor = try XCTUnwrap(index.tensors["experts"])
            let scratch = try GGUFLinearScratch(elements: 12)
            let x = MLXArray([Float(1), 2, 3, 4], [1, 4])
            let result = try GGUFStreamedLinear.run(x, tensor: tensor, expert: 2,
                payloadOffset: index.payloadOffset, sourceID: 1, firstRegionID: 0,
                tileRows: 3, reader: reader, scratch: scratch)
            XCTAssertEqual(result.output.asArray(Float.self), [420, 460, 500, 540, 580])
            XCTAssertEqual(result.report.logicalWeightBytes, 80)
            XCTAssertEqual(result.report.tiles, 2)
            XCTAssertEqual(result.report.scratchBytes, 48)
            XCTAssertEqual(reader.statistics.bytesRead, 80)
            XCTAssertEqual(pool.census.inUse, 0)
        }
    }

    func testSelectedExpertLRUReuseAndChurnStayWithinTheSharedPayloadBudget() throws {
        try withExpertMatrix { index, reader, pool in
            let tensor = try XCTUnwrap(index.tensors["experts"])
            let cache = try GGUFPackedTileCache(capacityBytes: 192, admittedSources: [999],
                evictableSources: [1], evictableCapacityBytes: 160)
            let dense = RegionDescriptor(identity: .init(sourceID: 999, index: 0), offset: 0, length: 32)
            try cache.insert(MLXArray([UInt8](repeating: 7, count: 32)), region: dense)
            let x = MLXArray.ones([2, 4], dtype: .float32)
            func run(_ experts: [UInt64], cache: GGUFPackedTileCache?) throws -> (output: MLXArray, report: GGUFLinearReport) {
                try GGUFStreamedLinear.runExperts(x, tensor: tensor, experts: experts,
                    payloadOffset: index.payloadOffset, sourceID: 1, tileRows: 3,
                    reader: reader, packedCache: cache)
            }
            let first = try run([0, 1], cache: cache)
            XCTAssertEqual(first.output.asArray(Float.self), [6,22,38,54,70,86,102,118,134,150])
            XCTAssertEqual(first.report.packedWeightCopyBytes, 160)
            let second = try run([1, 2], cache: cache)
            XCTAssertEqual(second.output.asArray(Float.self), [86,102,118,134,150,166,182,198,214,230])
            XCTAssertEqual(second.report.residentWeightBytesReused, 80)
            XCTAssertEqual(second.report.packedWeightCopyBytes, 80)
            XCTAssertGreaterThan(cache.statistics.evictions ?? 0, 0)
            XCTAssertEqual(cache.statistics.evictablePayloadBytes, 160)
            XCTAssertEqual(cache.statistics.residentPayloadBytes, 192)
            XCTAssertEqual(try cache.lookup(dense)?.asArray(UInt8.self), [UInt8](repeating: 7, count: 32))
            var cold: [Double] = [], hot: [Double] = []
            for sample in -1..<3 {
                let io = reader.statistics.bytesRead, begin = MonotonicClock.now()
                let streamed = try run([1, 2], cache: nil)
                let coldSeconds = MonotonicClock.seconds(since: begin)
                XCTAssertEqual(reader.statistics.bytesRead - io, 160)
                let ioWarm = reader.statistics.bytesRead, warmBegin = MonotonicClock.now()
                let resident = try run([1, 2], cache: cache)
                let hotSeconds = MonotonicClock.seconds(since: warmBegin)
                XCTAssertEqual(resident.output.asArray(Float.self), streamed.output.asArray(Float.self))
                XCTAssertEqual(reader.statistics.bytesRead - ioWarm, 0)
                XCTAssertEqual(resident.report.packedWeightCopyBytes, 0)
                XCTAssertEqual(resident.report.residentWeightBytesReused, 160)
                XCTAssertLessThanOrEqual(cache.statistics.residentPayloadBytes, 192)
                if sample >= 0 { cold.append(coldSeconds); hot.append(hotSeconds) }
            }
            XCTAssertEqual(pool.census.inUse, 0)
            let record: [String: Any] = ["syntheticOnly":true, "physicalPhone":false,
                "sourceRevision":BuildInfo.gitRevision, "os":ProcessInfo.processInfo.operatingSystemVersionString,
                "physicalMemoryBytes":ProcessInfo.processInfo.physicalMemory,
                "shape":[3,5,4], "selectedExperts":[1,2], "rawRetention":false,
                "totalCachePayloadBytes":192, "evictablePayloadBytes":160,
                "coldReaderBytes":160, "hotReaderBytes":0, "warmup":1,
                "samples":["streamedSeconds":cold,"residentSeconds":hot]]
            print("GGUF_EXPERT_LRU_BENCH " + String(decoding: try JSONSerialization.data(withJSONObject: record, options: .sortedKeys), as: UTF8.self))
        }
    }

    func testPackedBatchReducesEvaluationBarriersWithoutChangingSelectedExpertProducts() throws {
        try withExpertMatrix { index, reader, pool in
            let tensor = try XCTUnwrap(index.tensors["experts"])
            let x = MLXArray([Float(1), 2, 3, 4], [1, 4])
            let scratch = try GGUFLinearScratch(elements: 4)
            var times: [String: [Double]] = [:]
            for evaluation in [GGUFLinearEvaluation.perTile, .boundedBatch] {
                for sample in 0..<4 {
                    let result = try GGUFStreamedLinear.run(x, tensor: tensor, expert: 2,
                        payloadOffset: index.payloadOffset, sourceID: 1, firstRegionID: 0,
                        tileRows: 1, reader: reader, scratch: scratch, evaluation: evaluation)
                    XCTAssertEqual(result.output.asArray(Float.self), [420, 460, 500, 540, 580])
                    XCTAssertEqual(result.report.evaluationBatches, evaluation == .perTile ? 5 : 1)
                    XCTAssertLessThanOrEqual(result.report.peakCopiedWeightBytes, 16 << 20)
                    if sample > 0 { times[evaluation == .perTile ? "perTileSeconds" : "boundedBatchSeconds", default: []].append(result.report.wallSeconds) }
                }
            }
            XCTAssertEqual(reader.statistics.bytesRead, 8 * 80)
            XCTAssertEqual(pool.census.inUse, 0)
            let record: [String: Any] = ["kind": "synthetic-streamed-evaluation-control", "physicalPhone": false,
                "shape": [5, 4], "expertsSelected": [2], "tileRows": 1, "includesFileIO": true,
                "warmup": 1, "samples": times, "sourceRevision": BuildInfo.gitRevision,
                "os": ProcessInfo.processInfo.operatingSystemVersionString,
                "physicalMemoryBytes": ProcessInfo.processInfo.physicalMemory]
            print("GGUF_EVALUATION_BATCH_BENCH " + String(decoding: try JSONSerialization.data(withJSONObject: record, options: .sortedKeys), as: UTF8.self))
        }
    }

    func testWholeSelectedExpertsUseFewerReadsWithinTheSamePackedStagingBound() throws {
        let rows = 128, columns = 2560, count = 10
        var header = Data("GGUF".utf8)
        func integer(_ value: UInt64, _ bytes: Int) {
            for i in 0..<bytes { header.append(UInt8(truncatingIfNeeded: value >> (i * 8))) }
        }
        integer(3, 4); integer(1, 8); integer(0, 8)
        let name = Data("selected".utf8); integer(UInt64(name.count), 8); header.append(name)
        integer(3, 4); integer(UInt64(columns), 8); integer(UInt64(rows), 8); integer(UInt64(count), 8)
        integer(0, 4); integer(0, 8)
        let payloadOffset = (header.count + 31) / 32 * 32
        let payloadBytes = rows * columns * count * 4
        let index = try GGUFIndex.parse(header: header, fileByteCount: UInt64(payloadOffset + payloadBytes))
        header.append(Data(repeating: 0, count: payloadOffset - header.count))
        var payload = Data(count: payloadBytes)
        payload.withUnsafeMutableBytes { raw in
            let pointer = raw.bindMemory(to: Float.self).baseAddress!
            for expert in 0..<count {
                pointer.advanced(by: expert * rows * columns).update(repeating: Float(expert + 1), count: rows * columns)
            }
        }
        header.append(payload)
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try header.write(to: file); defer { try? FileManager.default.removeItem(at: file) }
        let pool = try BufferPool(configuration: .init(slotCount: 4, slotBytes: 4 << 20, poisonOnRelease: true))
        let reader = try TileReader(path: file.path, pool: pool,
            configuration: .init(queueDepth: 3, readAhead: 3, slotAcquireTimeout: 5))
        defer { reader.shutdown() }
        let tensor = try XCTUnwrap(index.tensors["selected"])
        let experts: [UInt64] = [9, 1, 7, 0, 8, 2, 6, 3, 5, 4]
        let x = MLXArray.ones([count, columns], dtype: .float32)
        let expected = experts.flatMap { Array(repeating: Float(columns * Int($0 + 1)), count: rows) }
        var samples: [String: [Double]] = [:]
        for tileRows in [32, 128] {
            for sample in 0..<4 {
                let before = reader.statistics
                let value = try GGUFStreamedLinear.runExperts(x, tensor: tensor, experts: experts,
                    payloadOffset: index.payloadOffset, sourceID: UInt64(tileRows), tileRows: tileRows, reader: reader)
                XCTAssertEqual(value.output.asArray(Float.self), expected)
                XCTAssertEqual(reader.statistics.readCount - before.readCount, tileRows == 32 ? 40 : 10)
                XCTAssertEqual(reader.statistics.bytesRead - before.bytesRead, UInt64(payloadBytes))
                XCTAssertLessThanOrEqual(value.report.peakCopiedWeightBytes, 32 << 20)
                XCTAssertEqual(value.report.evaluationBatches, 1)
                if sample > 0 { samples[tileRows == 32 ? "fragmentedSeconds" : "wholeExpertSeconds", default: []].append(value.report.wallSeconds) }
            }
        }
        XCTAssertEqual(pool.census.inUse, 0)
        let record: [String: Any] = ["kind": "synthetic-selected-expert-read-geometry", "physicalPhone": false,
            "shape": [count, rows, columns], "packedBytes": payloadBytes, "fragmentedReadCount": 40,
            "wholeExpertReadCount": 10, "maximumCopiedPayloadBytes": 32 << 20,
            "includesFileIO": true, "noCache": false, "warmup": 1, "samples": samples,
            "sourceRevision": BuildInfo.gitRevision, "os": ProcessInfo.processInfo.operatingSystemVersionString,
            "physicalMemoryBytes": ProcessInfo.processInfo.physicalMemory]
        print("GGUF_EXPERT_READ_GEOMETRY_BENCH " + String(decoding: try JSONSerialization.data(withJSONObject: record, options: .sortedKeys), as: UTF8.self))
    }

    func testDifferentTileSizesMatchIndependentDotProducts() throws {
        for rows in [1, 2, 3] {
            try withExpertMatrix { index, reader, _ in
                let tensor = try XCTUnwrap(index.tensors["experts"])
                let x = MLXArray([Float(1), 2, 3, 4, 4, 3, 2, 1], [2, 4])
                let result = try GGUFStreamedLinear.run(x, tensor: tensor, expert: 0,
                    payloadOffset: index.payloadOffset, sourceID: 1, firstRegionID: 0,
                    tileRows: rows, reader: reader, scratch: GGUFLinearScratch(elements: 12))
                XCTAssertEqual(result.output.asArray(Float.self), [20, 60, 100, 140, 180, 10, 50, 90, 130, 170])
                let report = try JSONEncoder().encode(result.report)
                print("GGUF_MATRIX_PROBE tileRows=\(rows) " + String(decoding: report, as: UTF8.self))
            }
        }
    }

    func testRejectsInvalidShapesAndBudgetsBeforeReading() throws {
        try withExpertMatrix { index, reader, _ in
            let tensor = try XCTUnwrap(index.tensors["experts"])
            let scratch = try GGUFLinearScratch(elements: 4)
            let x = MLXArray([Float(1), 2, 3, 4], [1, 4])
            XCTAssertThrowsError(try GGUFStreamedLinear.run(x, tensor: tensor, expert: 0,
                payloadOffset: index.payloadOffset, sourceID: 1, firstRegionID: 0,
                tileRows: 2, reader: reader, scratch: scratch, execution: .dequantized))
            XCTAssertThrowsError(try GGUFStreamedLinear.run(x, tensor: tensor, expert: 3,
                payloadOffset: index.payloadOffset, sourceID: 1, firstRegionID: 0,
                tileRows: 1, reader: reader, scratch: scratch))
            XCTAssertThrowsError(try GGUFStreamedLinear.run(x, tensor: tensor, expert: 0,
                payloadOffset: index.payloadOffset, sourceID: 1, firstRegionID: 0,
                tileRows: 1, reader: reader, scratch: scratch, maximumOutputBytes: 4))
            XCTAssertEqual(reader.statistics.bytesRead, 0)
        }
    }

    func testBoundedExpertCacheAvoidsRepeatedDiskReadsWithoutChangingOutput() throws {
        try withExpertMatrix(cache: true) { index, reader, pool in
            let tensor = try XCTUnwrap(index.tensors["experts"])
            let scratch = try GGUFLinearScratch(elements: 12)
            let x = MLXArray([Float(1), 2, 3, 4], [1, 4])
            func run(expert: UInt64, firstRegion: UInt64) throws -> MLXArray {
                try GGUFStreamedLinear.run(x, tensor: tensor, expert: expert,
                    payloadOffset: index.payloadOffset, sourceID: 1, firstRegionID: firstRegion,
                    tileRows: 3, reader: reader, scratch: scratch).output
            }
            let first = try run(expert: 2, firstRegion: 0)
            XCTAssertEqual(reader.statistics.bytesRead, 80)
            let cached = try run(expert: 2, firstRegion: 0)
            XCTAssertEqual(cached.asArray(Float.self), first.asArray(Float.self))
            XCTAssertEqual(reader.statistics.bytesRead, 80, "The second pass must hit the bounded raw cache")
            let evicting = try run(expert: 1, firstRegion: 2)
            XCTAssertEqual(evicting.asArray(Float.self), [220, 260, 300, 340, 380])
            XCTAssertEqual(reader.statistics.bytesRead, 160)
            let reloaded = try run(expert: 2, firstRegion: 0)
            XCTAssertEqual(reloaded.asArray(Float.self), first.asArray(Float.self))
            XCTAssertEqual(reader.statistics.bytesRead, 240)
            XCTAssertEqual(pool.census.inUse, 0)
            XCTAssertLessThanOrEqual(pool.statistics.peakInUseBytes, 96)
        }
    }
}
