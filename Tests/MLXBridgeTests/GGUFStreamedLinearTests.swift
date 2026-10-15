import Darwin
import Foundation
import GGUFCore
import MLX
import StorageCore
import XCTest
@testable import MLXBridge

final class GGUFStreamedLinearTests: XCTestCase {
    private func withExpertMatrix(_ body: (GGUFIndex, TileReader, BufferPool) throws -> Void) throws {
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
                                                       retainOnRelease: false, poisonOnRelease: true))
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

    func testDifferentTileSizesMatchIndependentDotProducts() throws {
        for rows in [1, 2, 3] {
            try withExpertMatrix { index, reader, _ in
                let tensor = try XCTUnwrap(index.tensors["experts"])
                let x = MLXArray([Float(1), 2, 3, 4, 4, 3, 2, 1], [2, 4])
                let result = try GGUFStreamedLinear.run(x, tensor: tensor, expert: 0,
                    payloadOffset: index.payloadOffset, sourceID: 1, firstRegionID: 0,
                    tileRows: rows, reader: reader, scratch: GGUFLinearScratch(elements: 12))
                XCTAssertEqual(result.output.asArray(Float.self), [20, 60, 100, 140, 180, 10, 50, 90, 130, 170])
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
                tileRows: 2, reader: reader, scratch: scratch))
            XCTAssertThrowsError(try GGUFStreamedLinear.run(x, tensor: tensor, expert: 3,
                payloadOffset: index.payloadOffset, sourceID: 1, firstRegionID: 0,
                tileRows: 1, reader: reader, scratch: scratch))
            XCTAssertThrowsError(try GGUFStreamedLinear.run(x, tensor: tensor, expert: 0,
                payloadOffset: index.payloadOffset, sourceID: 1, firstRegionID: 0,
                tileRows: 1, reader: reader, scratch: scratch, maximumOutputBytes: 4))
            XCTAssertEqual(reader.statistics.bytesRead, 0)
        }
    }
}
