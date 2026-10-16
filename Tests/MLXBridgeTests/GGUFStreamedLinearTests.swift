import Darwin
import Foundation
import GGUFCore
import MLX
import StorageCore
import XCTest
@testable import MLXBridge

final class GGUFStreamedLinearTests: XCTestCase {
    private func withPairMatrix(columns: Int, rows: Int, experts: Int, quantized: Bool = false,
                                materialize: Bool = true,
                                _ body: (GGUFIndex, TileReader, BufferPool) throws -> Void) throws {
        var header = Data("GGUF".utf8)
        func put(_ value: UInt64, _ count: Int) {
            for i in 0..<count { header.append(UInt8(truncatingIfNeeded: value >> (i * 8))) }
        }
        let rowBytes = quantized ? columns / 256 * 144 : columns * 4
        let matrixBytes = rows * rowBytes, expertBytes = matrixBytes * experts
        func aligned(_ bytes: Int) -> Int { (bytes + 31) / 32 * 32 }
        let upOffset = aligned(expertBytes), denseOffset = aligned(upOffset + expertBytes)
        let denseUpOffset = aligned(denseOffset + matrixBytes)
        let tensors: [(String, [Int], Int)] = [
            ("gate", [columns, rows, experts], 0), ("up", [columns, rows, experts], upOffset),
            ("dense_gate", [columns, rows], denseOffset),
            ("dense_up", [columns, rows], denseUpOffset)]
        put(3, 4); put(UInt64(tensors.count), 8); put(0, 8)
        for (name, shape, offset) in tensors {
            let bytes = Data(name.utf8); put(UInt64(bytes.count), 8); header.append(bytes)
            put(UInt64(shape.count), 4)
            for dimension in shape { put(UInt64(dimension), 8) }
            put(quantized ? 12 : 0, 4); put(UInt64(offset), 8)
        }
        let payloadOffset = (header.count + 31) / 32 * 32
        let fileBytes = payloadOffset + denseUpOffset + matrixBytes
        let index = try GGUFIndex.parse(header: header, fileByteCount: UInt64(fileBytes))
        header.append(Data(repeating: 0, count: payloadOffset - header.count))
        if materialize {
            for (name, shape, offset) in tensors {
                header.append(Data(repeating: 0, count: payloadOffset + offset - header.count))
                for expert in 0..<(shape.count == 3 ? experts : 1) {
                    for row in 0..<rows {
                        let q = 1 + (expert * 3 + row % 7 + (name.contains("up") ? 5 : 0)) % 15
                        if quantized {
                            var block = Data(repeating: 0, count: 144)
                            var d = Float16(0.0625).bitPattern.littleEndian
                            withUnsafeBytes(of: &d) { block.replaceSubrange(0..<2, with: $0) }
                            for i in 0..<4 { block[4 + i] = 1; block[12 + i] = 1 }
                            for i in 16..<144 { block[i] = UInt8(q | (q << 4)) }
                            for _ in 0..<(columns / 256) { header.append(block) }
                        } else {
                            var bits = (Float(q) / 16).bitPattern.littleEndian
                            for _ in 0..<columns { withUnsafeBytes(of: &bits) { header.append(contentsOf: $0) } }
                        }
                    }
                }
            }
            XCTAssertEqual(header.count, fileBytes)
        }
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try header.write(to: file); defer { try? FileManager.default.removeItem(at: file) }
        if !materialize {
            let handle = try FileHandle(forWritingTo: file)
            try handle.truncate(atOffset: UInt64(fileBytes)); try handle.close()
        }
        let pool = try BufferPool(configuration: .init(slotCount: 4, slotBytes: 4 << 20, poisonOnRelease: true))
        let reader = try TileReader(path: file.path, pool: pool,
            configuration: .init(queueDepth: 3, readAhead: 3, slotAcquireTimeout: 5))
        defer { reader.shutdown() }
        try body(index, reader, pool)
    }

    private func pairExpected(columns: Int, rows: Int, selected: [UInt64], right: Bool) -> [Float] {
        selected.enumerated().flatMap { slot, expert in
            (0..<rows).map { row in
                let q = 1 + (Int(expert) * 3 + row % 7 + (right ? 5 : 0)) % 15
                return Float(q * columns * (slot + 1)) / 16
            }
        }
    }

    func testPairedRowsPreserveExpertOrderProductsAndExistingCacheIdentities() throws {
        for quantized in [false, true] {
            let columns = quantized ? 256 : 4, rows = 5
            try withPairMatrix(columns: columns, rows: rows, experts: 3, quantized: quantized) { index, reader, pool in
                let left = try XCTUnwrap(index.tensors["gate"]), right = try XCTUnwrap(index.tensors["up"])
                let selected: [UInt64] = [2, 0]
                let x = MLXArray(Array(repeating: Float(1), count: columns) + Array(repeating: Float(2), count: columns), [2, columns])
                let capacity = Int(left.byteCount + right.byteCount)
                let cache = try GGUFPackedTileCache(capacityBytes: capacity, admittedSources: [11, 12])
                let a = try GGUFStreamedLinear.runExperts(x, tensor: left, experts: selected,
                    payloadOffset: index.payloadOffset, sourceID: 11, tileRows: rows, reader: reader, packedCache: cache)
                let b = try GGUFStreamedLinear.runExperts(x, tensor: right, experts: selected,
                    payloadOffset: index.payloadOffset, sourceID: 12, tileRows: rows, reader: reader, packedCache: cache)
                let before = reader.statistics.bytesRead
                let paired = try GGUFStreamedLinear.runPair(x, left: left, right: right, experts: selected,
                    payloadOffset: index.payloadOffset, sourceIDs: (11, 12), maximumRegionBytes: 4 << 20,
                    reader: reader, packedCache: cache)
                XCTAssertEqual(paired.left.asArray(Float.self), pairExpected(columns: columns, rows: rows, selected: selected, right: false))
                XCTAssertEqual(paired.right.asArray(Float.self), pairExpected(columns: columns, rows: rows, selected: selected, right: true))
                XCTAssertEqual(paired.left.asArray(Float.self).map(\.bitPattern), a.output.asArray(Float.self).map(\.bitPattern))
                XCTAssertEqual(paired.right.asArray(Float.self).map(\.bitPattern), b.output.asArray(Float.self).map(\.bitPattern))
                XCTAssertEqual(reader.statistics.bytesRead, before)
                XCTAssertEqual(paired.report.packedWeightCopyBytes, 0)
                XCTAssertEqual(paired.report.residentWeightBytesReused, paired.report.logicalWeightBytes)
                XCTAssertEqual(paired.report.logicalProjectionCount, 2)
                XCTAssertEqual(paired.report.evaluationBatches, 1)
                XCTAssertEqual(pool.census.inUse, 0)
                let denseLeft = try XCTUnwrap(index.tensors["dense_gate"])
                let denseRight = try XCTUnwrap(index.tensors["dense_up"])
                let denseCache = try GGUFPackedTileCache(capacityBytes: Int(denseLeft.byteCount + denseRight.byteCount),
                    admittedSources: [13, 14])
                let scratch = try GGUFLinearScratch(elements: columns * rows)
                let da = try GGUFStreamedLinear.run(x, tensor: denseLeft, payloadOffset: index.payloadOffset,
                    sourceID: 13, firstRegionID: 0, tileRows: rows, reader: reader, scratch: scratch, packedCache: denseCache)
                let db = try GGUFStreamedLinear.run(x, tensor: denseRight, payloadOffset: index.payloadOffset,
                    sourceID: 14, firstRegionID: 0, tileRows: rows, reader: reader, scratch: scratch, packedCache: denseCache)
                let denseIO = reader.statistics.bytesRead
                let dense = try GGUFStreamedLinear.runPair(x, left: denseLeft, right: denseRight,
                    payloadOffset: index.payloadOffset, sourceIDs: (13, 14), maximumRegionBytes: 4 << 20,
                    reader: reader, packedCache: denseCache)
                XCTAssertEqual(dense.left.asArray(Float.self), pairExpected(columns: columns, rows: rows, selected: [0,0], right: false))
                XCTAssertEqual(dense.right.asArray(Float.self), pairExpected(columns: columns, rows: rows, selected: [0,0], right: true))
                XCTAssertEqual(dense.left.asArray(Float.self).map(\.bitPattern), da.output.asArray(Float.self).map(\.bitPattern))
                XCTAssertEqual(dense.right.asArray(Float.self).map(\.bitPattern), db.output.asArray(Float.self).map(\.bitPattern))
                XCTAssertEqual(reader.statistics.bytesRead, denseIO)
                XCTAssertEqual(dense.report.evaluationBatches, 1)
                XCTAssertLessThanOrEqual(dense.report.peakCopiedWeightBytes, 8 << 20)
            }
        }
    }

    func testPairRetainsBothSelectedCacheHitsBeforeColdAdmissionAndRejectsInvalidBounds() throws {
        try withPairMatrix(columns: 4, rows: 5, experts: 3) { index, reader, pool in
            let left = try XCTUnwrap(index.tensors["gate"]), right = try XCTUnwrap(index.tensors["up"])
            let cache = try GGUFPackedTileCache(capacityBytes: 320, admittedSources: [],
                evictableSources: [11, 12], evictableCapacityBytes: 320)
            let x = MLXArray.ones([2, 4], dtype: .float32)
            func run(_ ids: [UInt64]) throws -> (left: MLXArray, right: MLXArray, report: GGUFLinearReport) {
                try GGUFStreamedLinear.runPair(x, left: left, right: right, experts: ids,
                    payloadOffset: index.payloadOffset, sourceIDs: (11, 12), maximumRegionBytes: 4096,
                    reader: reader, packedCache: cache)
            }
            _ = try run([0, 1])
            let second = try run([1, 2])
            XCTAssertEqual(second.report.residentWeightBytesReused, 160)
            XCTAssertEqual(second.report.packedWeightCopyBytes, 160)
            XCTAssertEqual(cache.statistics.residentPayloadBytes, 320)
            let before = reader.statistics.bytesRead
            let again = try run([2, 1])
            XCTAssertEqual(again.report.residentWeightBytesReused, 320)
            XCTAssertEqual(reader.statistics.bytesRead, before)
            XCTAssertEqual(again.left.asArray(Float.self), pairExpected(columns: 4, rows: 5, selected: [2, 1], right: false).enumerated().map { i, value in value / Float(i / 5 + 1) })
            XCTAssertThrowsError(try run([0, 0]))
            XCTAssertThrowsError(try run([0, 3]))
            XCTAssertThrowsError(try GGUFStreamedLinear.runPair(x, left: left, right: right, experts: [0, 1],
                payloadOffset: index.payloadOffset, sourceIDs: (11, 12), maximumRegionBytes: 79, reader: reader))
            XCTAssertThrowsError(try GGUFStreamedLinear.runPair(x, left: left, right: right, experts: [0, 1],
                payloadOffset: index.payloadOffset, sourceIDs: (11, 11), maximumRegionBytes: 4096, reader: reader))
            XCTAssertEqual(reader.statistics.bytesRead, before)
            XCTAssertEqual(pool.census.inUse, 0)
        }
        try withPairMatrix(columns: 2048, rows: 512, experts: 8, materialize: false) { index, reader, _ in
            XCTAssertThrowsError(try GGUFStreamedLinear.runPair(MLXArray.ones([8, 2048], dtype: .float32),
                left: XCTUnwrap(index.tensors["gate"]), right: XCTUnwrap(index.tensors["up"]), experts: [0,1,2,3,4,5,6,7],
                payloadOffset: index.payloadOffset, sourceIDs: (11, 12), maximumRegionBytes: 4 << 20, reader: reader)) { error in
                    guard case GGUFError.limit = error else { return XCTFail("Staging guard must reject before I/O: \(error)") }
                }
            XCTAssertEqual(reader.statistics.bytesRead, 0)
        }
    }

    func testPrimaryShapePairedProjectionBenchmarkIncludesReaderAndAssembly() throws {
        let columns = 2048, rows = 512, count = 8
        try withPairMatrix(columns: columns, rows: rows, experts: count, quantized: true) { index, reader, pool in
            let left = try XCTUnwrap(index.tensors["gate"]), right = try XCTUnwrap(index.tensors["up"])
            let selected: [UInt64] = [7, 0, 6, 1, 5, 2, 4, 3]
            let x = MLXArray.ones([count, columns], dtype: .float32)
            let expected = pairExpected(columns: columns, rows: rows, selected: selected, right: false).enumerated().map { i, v in v / Float(i / rows + 1) }
            for resident in [false, true] {
                let cache = resident ? try GGUFPackedTileCache(capacityBytes: Int(left.byteCount + right.byteCount), admittedSources: [11, 12]) : nil
                var separate: [Double] = [], paired: [Double] = []
                for sample in -1..<3 {
                    let before = reader.statistics.bytesRead, begin = MonotonicClock.now()
                    let a = try GGUFStreamedLinear.runExperts(x, tensor: left, experts: selected, payloadOffset: index.payloadOffset,
                        sourceID: 11, tileRows: rows, reader: reader, packedCache: cache)
                    let b = try GGUFStreamedLinear.runExperts(x, tensor: right, experts: selected, payloadOffset: index.payloadOffset,
                        sourceID: 12, tileRows: rows, reader: reader, packedCache: cache)
                    let aValues = a.output.asArray(Float.self), bValues = b.output.asArray(Float.self)
                    let duration = MonotonicClock.seconds(since: begin)
                    XCTAssertEqual(aValues, expected)
                    let readSeparate = reader.statistics.bytesRead - before
                    let nextRead = reader.statistics.bytesRead, next = MonotonicClock.now()
                    let result = try GGUFStreamedLinear.runPair(x, left: left, right: right, experts: selected,
                        payloadOffset: index.payloadOffset, sourceIDs: (11, 12), maximumRegionBytes: 4 << 20,
                        reader: reader, packedCache: cache)
                    let actual = result.left.asArray(Float.self), other = result.right.asArray(Float.self)
                    let pairDuration = MonotonicClock.seconds(since: next)
                    XCTAssertEqual(actual.map(\.bitPattern), aValues.map(\.bitPattern))
                    XCTAssertEqual(other.map(\.bitPattern), bValues.map(\.bitPattern))
                    XCTAssertEqual(result.report.evaluationBatches, 1)
                    XCTAssertEqual((a.report.evaluationBatches ?? 0) + (b.report.evaluationBatches ?? 0), 2)
                    XCTAssertLessThanOrEqual(result.report.peakCopiedWeightBytes, 32 << 20)
                    if sample >= 0 {
                        XCTAssertEqual(readSeparate, resident ? 0 : left.byteCount + right.byteCount)
                        XCTAssertEqual(reader.statistics.bytesRead - nextRead, resident ? 0 : left.byteCount + right.byteCount)
                        separate.append(duration); paired.append(pairDuration)
                    }
                }
                let record: [String: Any] = ["syntheticOnly":true, "physicalPhone":false, "shape":[count, rows, columns],
                    "format":"Q4_K", "resident":resident, "includesReaderAndAssembly":true, "warmup":1,
                    "readerBytesPerTimedRun":resident ? 0 : Int(left.byteCount + right.byteCount),
                    "maximumCopiedPayloadBytes":32 << 20, "separateEvaluationBatches":2, "pairedEvaluationBatches":1,
                    "sourceRevision":BuildInfo.gitRevision,"os":ProcessInfo.processInfo.operatingSystemVersionString,
                    "physicalMemoryBytes":ProcessInfo.processInfo.physicalMemory,
                    "samples":["separateSeconds":separate,"pairedSeconds":paired]]
                print("GGUF_PAIRED_PROJECTION_BENCH " + String(decoding: try JSONSerialization.data(withJSONObject: record, options: .sortedKeys), as: UTF8.self))
            }
            XCTAssertEqual(pool.census.inUse, 0)
            XCTAssertLessThanOrEqual(pool.statistics.peakInUseBytes, 16 << 20)
        }
    }

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
