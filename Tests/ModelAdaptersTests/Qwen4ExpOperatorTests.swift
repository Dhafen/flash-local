import Foundation
import MLX
import XCTest
@testable import ModelAdapters

final class Qwen4ExpOperatorTests: XCTestCase {
    private func near(_ actual: MLXArray, _ expected: [Float], accuracy: Float = 1e-5,
                      file: StaticString = #filePath, line: UInt = #line) {
        let values = actual.asArray(Float.self)
        XCTAssertEqual(values.count, expected.count, file: file, line: line)
        for (a, b) in zip(values, expected) { XCTAssertEqual(a, b, accuracy: accuracy, file: file, line: line) }
    }

    func testGDNMatchesScalarReferenceWithVFirstStateAndModuloHeads() throws {
        let q = MLXArray([Float(1), 2, 3, 4], [2, 2])
        let k = MLXArray([Float(0.5), 0.25, -0.5, 1], [2, 2])
        let v = MLXArray([Float(1), 2, 3, 4, 5, 6, 7, 8], [4, 2])
        let state = MLXArray((0..<16).map { Float($0) / 10 }, [4, 2, 2])
        let result = try Qwen4ExpGDN.step(q: q, k: k, v: v,
            logDecay: MLXArray(Array(repeating: Float(log(0.5)), count: 4)),
            beta: MLXArray(Array(repeating: Float(0.25), count: 4)), state: state)
        // Host scalar implementation of the pinned GGML Metal equation;
        // expected values evaluated independently in Python float64.
        near(result.output, [0.2452776647, 0.6209281422, 2.3909048039, 3.3057242020,
                             1.7478795747, 2.1235300523, 6.0501823965, 6.9650017947])
        near(result.state, [0.1234375, 0.11171875, 0.3390625, 0.26953125,
            -0.15625, 0.9625, -0.175, 1.3, 0.9859375, 0.74296875, 1.2015625,
            0.90078125, -0.23125, 2.3125, -0.25, 2.65])
    }

    func testGDNDeltaUsesDecayedStateBeforeUpdate() throws {
        let q = MLXArray([Float(1), 0, 0, 1], [2, 2])
        let v = MLXArray([Float(2), 3, 4, 5, 6, 7, 8, 9], [4, 2])
        let first = try Qwen4ExpGDN.step(q: q, k: q, v: v,
            logDecay: .zeros([4]), beta: MLXArray(Array(repeating: Float(0.5), count: 4)),
            state: .zeros([4, 2, 2]))
        near(first.output, [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5].map { Float($0) / sqrt(2) })
        let second = try Qwen4ExpGDN.step(q: q, k: q, v: .zeros([4, 2]),
            logDecay: MLXArray(Array(repeating: Float(log(0.5)), count: 4)),
            beta: .ones([4]), state: first.state)
        near(second.output, Array(repeating: 0, count: 8))
        near(second.state, Array(repeating: 0, count: 16))
    }

    func testCausalConvolutionHistoryAndPLEDilation() throws {
        let regular = try Qwen4ExpGDN.convolutionStep(input: MLXArray([Float(4)]),
            kernel: MLXArray([Float(1), 2, 3, 4], [4, 1]),
            history: MLXArray([Float(1), 2, 3], [3, 1]))
        near(regular.output, [Float(30) / (1 + exp(Float(-30)))])
        near(regular.history, [2, 3, 4])
        let dilated = try Qwen4ExpGDN.convolutionStep(input: MLXArray([Float(10)]),
            kernel: MLXArray([Float(1), 2, 3, 4], [4, 1]),
            history: MLXArray((1...9).map { Float($0) }, [9, 1]), dilation: 3)
        near(dilated.output, [70])
        near(dilated.history, (2...10).map { Float($0) })
    }

    func testGatedNormUsesSigmoidAndAlreadyFoldedGamma() throws {
        let output = try Qwen4ExpGDN.gatedNorm(.ones([2, 2]), gate: .zeros([2, 2]),
                                               weight: MLXArray([Float(2), 2]), epsilon: 1e-6)
        near(output, Array(repeating: 1 / sqrt(1 + Float(1e-6)), count: 4))
    }

    func testHyperConnectionGroupNormMeanAndNeutralScatter() throws {
        let residual = MLXArray([Float(1), 2, 3, 4], [1, 2, 2])
        let mix = try Qwen4ExpHyperConnections.mix(residual, gamma: .ones([2, 2]), epsilon: 1e-6,
            down: { _ in .zeros([1, 1]) }, up: { _ in .zeros([1, 4]) },
            injection: { _ in .zeros([1, 2]) })
        let a = sqrt(Float(2.5) + 1e-6), b = sqrt(Float(12.5) + 1e-6)
        near(mix.mixed, [(1 / a + 3 / b) / 4, (2 / a + 4 / b) / 4])
        let output = try Qwen4ExpHyperConnections.combine(residual: residual,
            blockOutput: MLXArray([Float(2), 4], [1, 2]), injection: XCTUnwrap(mix.injection))
        near(output, [3, 6, 5, 8])
    }

    func testRoutingNormalizesSelectedSoftmaxWeightsAndBreaksTiesByID() throws {
        let selection = try Qwen4ExpRouting.select(logits:
            MLXArray([Float(0.1), 0.2, 0.3, 0.4].map { log($0) }, [1, 4]), topK: 2)
        XCTAssertEqual(selection.ids, [[3, 2]])
        near(selection.weights, [Float(4) / 7, Float(3) / 7])
        let tie = try Qwen4ExpRouting.select(logits: .zeros([1, 3]), topK: 2)
        XCTAssertEqual(tie.ids, [[0, 1]])
        near(tie.weights, [0.5, 0.5])
        XCTAssertThrowsError(try Qwen4ExpRouting.select(logits: .zeros([1, 3]), topK: 4))
        XCTAssertThrowsError(try Qwen4ExpRouting.select(logits: MLXArray([Float.nan], [1, 1]), topK: 1))
    }
}
