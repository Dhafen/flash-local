import Foundation
import MLX
import MinirunKit
import XCTest
@testable import ModelAdapters

final class Qwen4ExpDecoderParityTests: XCTestCase {
    private struct Manifest: Decodable {
        let fileName: String
        let fileByteCount: UInt64
        let sha256: String
        let tokens: [UInt64]
        let contextTokens: Int
    }
    private struct Reference: Decodable {
        let tokens: [UInt64]
        let logits: [[Float]]
    }

    func testCompleteSyntheticGraphMatchesPinnedLlamaCPPLogits() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let header = environment["GGUF_HEADER_FIXTURE"] ?? environment["TEST_RUNNER_GGUF_HEADER_FIXTURE"] else {
            throw XCTSkip("Set GGUF_HEADER_FIXTURE and run the synthetic reference harness first")
        }
        let folder = URL(fileURLWithPath: header).deletingLastPathComponent()
        let manifestURL = folder.appendingPathComponent("synthetic-qwen4exp.json")
        let referenceURL = folder.appendingPathComponent("llama-reference-logits.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path),
              FileManager.default.fileExists(atPath: referenceURL.path) else {
            throw XCTSkip("Generate tiny model and pinned CPU reference; trained weights are unnecessary")
        }
        let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: manifestURL))
        let reference = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: referenceURL))
        XCTAssertEqual(reference.tokens, manifest.tokens)
        XCTAssertEqual(reference.logits.count, manifest.tokens.count)
        let expected = try GGUFModelManifest(fileName: manifest.fileName,
            fileByteCount: manifest.fileByteCount, sha256: manifest.sha256)
        let artifact = try VerifiedGGUFArtifact.verify(insideRoot: folder, manifest: expected)
        let decoder = try Qwen4ExpDecoder(artifact: artifact, contextTokens: manifest.contextTokens,
            budgetBytes: 5_500_000_000, rawCacheBytes: 2 << 20)
        for (position, token) in manifest.tokens.enumerated() {
            let actual = try decoder.logits(for: token).asArray(Float.self)
            let target = reference.logits[position]
            XCTAssertEqual(actual.count, target.count)
            XCTAssertTrue(target.contains { abs($0) > 1e-6 }, "Zero logits are not an informative fixture")
            var maximum: Float = 0, errorSquared = 0.0, referenceSquared = 0.0
            for (a, b) in zip(actual, target) {
                maximum = max(maximum, abs(a - b))
                errorSquared += pow(Double(a) - Double(b), 2)
                referenceSquared += pow(Double(b), 2)
            }
            let relative = sqrt(errorSquared / max(referenceSquared, 1e-20))
            print("QWEN4EXP_LOGIT_PARITY position=\(position) maxAbs=\(maximum) relativeL2=\(relative)")
            XCTAssertLessThanOrEqual(maximum, 0.01)
            XCTAssertLessThanOrEqual(relative, 1e-3)
        }
        XCTAssertEqual(decoder.position, manifest.tokens.count)
        decoder.reset()
        let firstAgain = try decoder.logits(for: manifest.tokens[0]).asArray(Float.self)
        for (a, b) in zip(firstAgain, reference.logits[0]) { XCTAssertEqual(a, b, accuracy: 0.01) }
        decoder.reset()
        XCTAssertThrowsError(try decoder.logits(for: 1, cancellationRequested: { true }))
        XCTAssertThrowsError(try decoder.logits(for: 1), "Partial state requires explicit reset")
        decoder.reset()
        _ = try decoder.logits(for: 1)
        XCTAssertEqual(decoder.position, 1)
    }
}
