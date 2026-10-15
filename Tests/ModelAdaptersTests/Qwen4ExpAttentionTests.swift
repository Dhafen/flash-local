import Foundation
import MLX
import XCTest
@testable import ModelAdapters

final class Qwen4ExpAttentionTests: XCTestCase {
    func testOnlineAttentionMatchesIndependentFullPrefixSoftmaxAcrossChunkSizes() throws {
        for chunk in [1, 3, 8] {
            let cache = try Qwen4ExpTextAttention.Cache(keyValueHeads: 2, headDimension: 2,
                contextLimit: 8, chunkTokens: chunk, maximumBytes: 128)
            var allKeys: [[Float]] = [], allValues: [[Float]] = []
            for token in 0..<8 {
                let key = (0..<4).map { Float(token * 4 + $0 + 1) / 4 }
                let value = (0..<4).map { -Float(token * 4 + $0 + 1) / 8 }
                let query: [Float] = [0.3, 0.7, -0.5, 0.25, 1, -1, 0.6, 0.1]
                allKeys.append(key); allValues.append(value)
                let output = try cache.appendAndAttend(query: MLXArray(query, [4, 2]),
                    key: MLXArray(key, [2, 2]), value: MLXArray(value, [2, 2])).asArray(Float.self)
                for head in 0..<4 {
                    let kvHead = head / 2 // contiguous GQA, unlike GDN modulo-head mapping
                    let scores = allKeys.map { row in
                        (Double(query[head * 2]) * Double(row[kvHead * 2])
                            + Double(query[head * 2 + 1]) * Double(row[kvHead * 2 + 1])) / sqrt(2)
                    }
                    let maximum = scores.max()!
                    let probabilities = scores.map { exp($0 - maximum) }
                    let denominator = probabilities.reduce(0, +)
                    for column in 0..<2 {
                        let expected = zip(probabilities, allValues).reduce(0.0) {
                            $0 + $1.0 * Double($1.1[kvHead * 2 + column])
                        } / denominator
                        XCTAssertEqual(output[head * 2 + column], Float(expected), accuracy: 1e-5)
                    }
                }
                XCTAssertEqual(cache.allocatedBytes, (token + 1) * 2 * 2 * 4)
            }
            XCTAssertThrowsError(try cache.appendAndAttend(query: .zeros([4, 2]),
                key: .zeros([2, 2]), value: .zeros([2, 2])))
            XCTAssertEqual(cache.count, 8, "No eviction or context truncation")
            cache.reset()
            XCTAssertEqual(cache.count, 0)
            XCTAssertEqual(cache.allocatedBytes, 0)
        }
    }

    func testPerHeadQGateInterleavingAndSigmoidGate() throws {
        let cache = try Qwen4ExpTextAttention.Cache(keyValueHeads: 1, headDimension: 2,
                                                   contextLimit: 2, maximumBytes: 16)
        let qGate = MLXArray([Float(1), 2, 0, 0, 3, 4, log(Float(3)), -log(Float(3))], [1, 8])
        let result = try Qwen4ExpTextAttention.step(queryAndGate: qGate,
            key: MLXArray([Float(1), 2], [1, 2]), value: MLXArray([Float(8), 12], [1, 2]),
            queryNorm: .ones([2]), keyNorm: .ones([2]), epsilon: 1e-6,
            rotaryDimensions: 2, ropeBase: 10_000_000, cache: cache)
        let values = result.asArray(Float.self)
        for (actual, expected) in zip(values, [Float(4), 6, 6, 3]) {
            XCTAssertEqual(actual, expected, accuracy: 1e-5)
        }
        XCTAssertEqual(cache.allocatedBytes, 8)
    }

    func testOnlineSoftmaxRemainsFiniteForLargeScores() throws {
        let cache = try Qwen4ExpTextAttention.Cache(keyValueHeads: 1, headDimension: 2,
            contextLimit: 2, chunkTokens: 1, maximumBytes: 16)
        _ = try cache.appendAndAttend(query: MLXArray([Float(1000), 1000], [1, 2]),
            key: MLXArray([Float(-1000), -1000], [1, 2]), value: MLXArray([Float(1), 2], [1, 2]))
        let result = try cache.appendAndAttend(query: MLXArray([Float(1000), 1000], [1, 2]),
            key: MLXArray([Float(1000), 1000], [1, 2]), value: MLXArray([Float(3), 4], [1, 2]))
        XCTAssertEqual(result.asArray(Float.self), [3, 4])
    }

    func testRejectsMemoryAdmissionAndInvalidHeadsBeforeCacheMutation() throws {
        XCTAssertThrowsError(try Qwen4ExpTextAttention.Cache(keyValueHeads: 2, headDimension: 256,
            contextLimit: 65536, maximumBytes: 100_000))
        let cache = try Qwen4ExpTextAttention.Cache(keyValueHeads: 2, headDimension: 2,
            contextLimit: 2, maximumBytes: 32)
        XCTAssertThrowsError(try cache.appendAndAttend(query: .zeros([3, 2]),
            key: .zeros([2, 2]), value: .zeros([2, 2])))
        XCTAssertEqual(cache.count, 0)
    }
}
