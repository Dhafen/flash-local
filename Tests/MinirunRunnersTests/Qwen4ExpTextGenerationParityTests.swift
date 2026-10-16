import Foundation
import MinirunKit
import ModelAdapters
import StorageCore
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
        XCTAssertEqual(summary.promptOutputProjectionsSkipped, reference.promptTokenIDs.count - 1)
        XCTAssertEqual((summary.prefillIOBytesRead ?? 0) + (summary.decodeIOBytesRead ?? 0), summary.ioBytesRead)
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
        var stopWithReport = false
        let partial = try engine.generate(messages: manifest.messages, maximumNewTokens: manifest.maximumNewTokens,
            cancellationRequested: { stopWithReport }, captureCancelledSummary: true) { event in
                if case .token = event { stopWithReport = true }
            }
        XCTAssertEqual(partial.stopReason, .cancelled)
        XCTAssertEqual(partial.generatedTokenIDs, Array(summary.generatedTokenIDs.prefix(1)))
        XCTAssertEqual(partial.promptTokensCompleted, summary.promptTokenIDs.count)
        XCTAssertEqual(partial.decodeForwardPasses, 0)
        XCTAssertNil(partial.decodeTokensPerSecond)
        XCTAssertEqual((partial.prefillIOBytesRead ?? 0) + (partial.decodeIOBytesRead ?? 0), partial.ioBytesRead)
        var stopDuringPrompt = false
        let prefix = try engine.generate(messages: manifest.messages, maximumNewTokens: manifest.maximumNewTokens,
            cancellationRequested: { stopDuringPrompt }, captureCancelledSummary: true) { event in
                if case .prefill = event { stopDuringPrompt = true }
            }
        XCTAssertEqual(prefix.stopReason, .cancelled)
        XCTAssertEqual(prefix.promptTokensCompleted, 1)
        XCTAssertTrue(prefix.generatedTokenIDs.isEmpty)
        XCTAssertEqual(prefix.decodeIOBytesRead, 0)
        XCTAssertNil(prefix.decodeTokensPerSecond)
        let cached = try Qwen4ExpTextGeneration(artifact: artifact, contextTokens: manifest.contextTokens,
            budgetBytes: 5_500_000_000, rawCacheBytes: 2 << 20, residentCacheBytes: 1 << 20)
        var firstText = ""
        let first = try cached.generate(messages: manifest.messages, maximumNewTokens: manifest.maximumNewTokens,
            reusePromptPrefix: true) { event in
                if case .token(_, let delta, _) = event { firstText += delta }
                if case .trailingText(let delta) = event { firstText += delta }
            }
        XCTAssertEqual(first.generatedTokenIDs, reference.generatedTokenIDs)
        XCTAssertEqual(first.reusedPromptTokens, 0)
        XCTAssertGreaterThan(try XCTUnwrap(first.prefixCheckpointPayloadBytes), 0)
        let followup = manifest.messages + [.init(role: .assistant, content: firstText), .init(role: .user, content: "Next")]
        let warm = try cached.generate(messages: followup, maximumNewTokens: 2, reusePromptPrefix: true) { _ in }
        let cold = try engine.generate(messages: followup, maximumNewTokens: 2) { _ in }
        XCTAssertEqual(warm.promptTokenIDs, cold.promptTokenIDs)
        XCTAssertEqual(warm.generatedTokenIDs, cold.generatedTokenIDs)
        XCTAssertGreaterThan(try XCTUnwrap(warm.reusedPromptTokens), 0)
        XCTAssertLessThan(try XCTUnwrap(warm.evaluatedPromptTokens), try XCTUnwrap(cold.evaluatedPromptTokens))
        XCTAssertLessThan(warm.projections.logicalWeightBytes, cold.projections.logicalWeightBytes)
        XCTAssertEqual((warm.prefillProjections?.calls ?? 0) + (warm.decodeProjections?.calls ?? 0), warm.projections.calls)
        var stopWarm = false
        let stoppedWarm = try cached.generate(messages: followup, maximumNewTokens: 2, reusePromptPrefix: true,
            cancellationRequested: { stopWarm }, captureCancelledSummary: true) { event in
                if case .prefill = event { stopWarm = true }
            }
        XCTAssertEqual(stoppedWarm.stopReason, .cancelled)
        let retry = try cached.generate(messages: followup, maximumNewTokens: 2, reusePromptPrefix: true) { _ in }
        XCTAssertEqual(retry.generatedTokenIDs, cold.generatedTokenIDs)
        let changed = [Qwen35ChatMessage(role: .user, content: "Changed")]
        let diverged = try cached.generate(messages: changed, maximumNewTokens: 2, reusePromptPrefix: true) { _ in }
        let changedCold = try engine.generate(messages: changed, maximumNewTokens: 2) { _ in }
        XCTAssertEqual(diverged.reusedPromptTokens, 0)
        XCTAssertEqual(diverged.generatedTokenIDs, changedCold.generatedTokenIDs)
        try cached.resetConversation()
        let cleared = try cached.generate(messages: manifest.messages, maximumNewTokens: 2, reusePromptPrefix: true) { _ in }
        XCTAssertEqual(cleared.reusedPromptTokens, 0)
        XCTAssertEqual(cleared.generatedTokenIDs, Array(reference.generatedTokenIDs.prefix(2)))
        print("QWEN4EXP_PREFIX_REUSE syntheticOnly=true physicalPhone=false source=\(BuildInfo.gitRevision) exactTokenIDs=true reused=\(warm.reusedPromptTokens ?? 0) evaluated=\(warm.evaluatedPromptTokens ?? 0) cold=\(cold.evaluatedPromptTokens ?? 0) warmPrefillSeconds=\(warm.prefillSeconds) coldPrefillSeconds=\(cold.prefillSeconds) cancelledRetry=true divergedReset=true")
        XCTAssertThrowsError(try engine.generate(messages: manifest.messages, maximumNewTokens: 0) { _ in })
        print("QWEN4EXP_TEXT_SESSION_PARITY prompt=\(summary.promptTokenIDs.count) generated=\(ids.count) winnerMargin=\(reference.minimumWinnerMargin) exactTokenIDs=true")
    }
}
