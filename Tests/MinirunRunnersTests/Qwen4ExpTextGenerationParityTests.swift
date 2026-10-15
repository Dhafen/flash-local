import Foundation
import MinirunKit
import ModelAdapters
import XCTest
@testable import MinirunRunners

final class Qwen4ExpTextGenerationParityTests: XCTestCase {
    private struct Manifest: Decodable {
        let fileName: String
        let fileByteCount: UInt64
        let sha256: String
        let contextTokens: Int
        let maximumNewTokens: Int
        let messages: [Qwen35ChatMessage]
    }
    private struct Reference: Decodable {
        let promptTokenIDs: [Int]
        let generatedTokenIDs: [Int]
        let stoppedAtEOG: Bool
        let decodedHex: String
        let minimumWinnerMargin: Double
    }
    func testCompleteTextSessionMatchesPinnedCPUReferenceAndRecoversAfterCancellation() throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["GGUF_HEADER_FIXTURE"] ?? env["TEST_RUNNER_GGUF_HEADER_FIXTURE"] else {
            throw XCTSkip("Set GGUF_HEADER_FIXTURE and run the synthetic text-session reference")
        }
        let folder = URL(fileURLWithPath: path).deletingLastPathComponent()
        let result = folder.appendingPathComponent("qwen4exp-session-reference.json")
        guard FileManager.default.fileExists(atPath: result.path) else { throw XCTSkip("Run scripts/run_synthetic_reference.py") }
        let manifest = try JSONDecoder().decode(Manifest.self,
            from: Data(contentsOf: folder.appendingPathComponent("synthetic-qwen4exp-chat.json")))
        let reference = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: result))
        let expected = try GGUFModelManifest(fileName: manifest.fileName,
            fileByteCount: manifest.fileByteCount, sha256: manifest.sha256)
        let artifact = try VerifiedGGUFArtifact.verify(insideRoot: folder, manifest: expected)
        let engine = try Qwen4ExpTextGeneration(artifact: artifact, contextTokens: manifest.contextTokens,
            budgetBytes: 5_500_000_000, rawCacheBytes: 2 << 20)
        var ids: [Int] = [], text = "", promptSteps: [Int] = []
        let summary = try engine.generate(messages: manifest.messages, maximumNewTokens: manifest.maximumNewTokens) { event in
            switch event {
            case .prefill(let progress): promptSteps.append(progress.promptTokensCompleted)
            case .token(let id, let delta, _): ids.append(id); text += delta
            case .trailingText(let delta): text += delta
            }
        }
        XCTAssertEqual(summary.promptTokenIDs, reference.promptTokenIDs)
        XCTAssertEqual(summary.generatedTokenIDs, reference.generatedTokenIDs)
        XCTAssertEqual(ids, reference.generatedTokenIDs)
        XCTAssertGreaterThan(ids.count, 1, "The fixture must exercise autoregressive decode beyond the first output")
        XCTAssertEqual(promptSteps, Array(1...reference.promptTokenIDs.count))
        XCTAssertEqual(summary.stopReason, reference.stoppedAtEOG ? .endOfGeneration : .tokenLimit)
        var bytes = Data(); var cursor = reference.decodedHex.startIndex
        while cursor < reference.decodedHex.endIndex {
            let next = reference.decodedHex.index(cursor, offsetBy: 2)
            bytes.append(try XCTUnwrap(UInt8(reference.decodedHex[cursor..<next], radix: 16))); cursor = next
        }
        XCTAssertEqual(text, String(decoding: bytes, as: UTF8.self))
        XCTAssertEqual(summary.modelSHA256, manifest.sha256)
        XCTAssertGreaterThan(summary.ioBytesRead, 0)
        XCTAssertGreaterThan(summary.maximumObservedFootprintBytes, 0)
        XCTAssertGreaterThan(summary.decodeForwardPasses, 0)
        XCTAssertNotNil(summary.decodeTokensPerSecond)
        XCTAssertGreaterThan(summary.projections.calls, 0)
        XCTAssertGreaterThan(summary.projections.logicalWeightBytes, 0)
        XCTAssertGreaterThan(summary.projections.computeSeconds, 0)
        XCTAssertGreaterThan(summary.rawCacheMisses, 0)
        XCTAssertLessThanOrEqual(summary.rawCacheLifetimePeakBytes, summary.rawCacheCapacityBytes)
        let report = try JSONDecoder().decode(Qwen4ExpTextGeneration.Summary.self, from: JSONEncoder().encode(summary))
        XCTAssertEqual(report.projections, summary.projections)
        XCTAssertEqual(report.generatedTokenIDs, summary.generatedTokenIDs)
        var cancelled = false
        XCTAssertThrowsError(try engine.generate(messages: manifest.messages, maximumNewTokens: manifest.maximumNewTokens,
            cancellationRequested: { cancelled }) { event in
                if case .token = event { cancelled = true }
            }) { XCTAssertTrue($0 is CancellationError) }
        let again = try engine.generate(messages: manifest.messages, maximumNewTokens: manifest.maximumNewTokens) { _ in }
        XCTAssertEqual(again.generatedTokenIDs, summary.generatedTokenIDs)
        XCTAssertEqual(again.projections.calls, summary.projections.calls,
            "Reset sessions report their own work, excluding the cancelled partial run")
        XCTAssertEqual(again.projections.logicalWeightBytes, summary.projections.logicalWeightBytes)
        XCTAssertThrowsError(try engine.generate(messages: manifest.messages, maximumNewTokens: 0) { _ in })
        print("QWEN4EXP_TEXT_SESSION_PARITY prompt=\(summary.promptTokenIDs.count) generated=\(ids.count) winnerMargin=\(reference.minimumWinnerMargin) exactTokenIDs=true")
    }
}
