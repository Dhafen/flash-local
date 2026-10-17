import Foundation
import XCTest
@testable import OrnithPerformance

final class PerformanceSessionTests: XCTestCase {
    func testOnlyAnExactRenderedPrefixReusesStateAndContextRejectionDoesNotDestroyIt() async throws {
        let engine = ProbeEngine()
        let session = OrnithPerformanceSession(testingEngine: engine, codec: ProbeCodec())
        let first = try await session.generate(messages: [.init(role: .user, content: "first")], maximumNewTokens: 8)
        XCTAssertEqual(first.text, "AB")
        XCTAssertEqual(first.report.reusedPromptTokens, 0)
        XCTAssertEqual(first.report.finishReason, .stop)
        do {
            _ = try await session.generate(messages: [.init(role: .user, content: "too long")], maximumNewTokens: 8)
            XCTFail("context rejection expected")
        } catch { }
        let second = try await session.generate(messages: [.init(role: .user, content: "continuation")], maximumNewTokens: 8)
        XCTAssertEqual(second.text, "C")
        XCTAssertEqual(second.report.reusedPromptTokens, 4)
        XCTAssertEqual(second.report.evaluatedPromptTokens, 1)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(second.report)) as? [String: Any])
        XCTAssertNotNil(json["decodeForwardTokensPerSecond"])
        XCTAssertNotNil(json["prefillSeconds"])
        XCTAssertEqual(json["fullModelNumericsVerified"] as? Bool, false)
        XCTAssertEqual(json["forwardMetricsIncludeInterruptedWork"] as? Bool, false)
        XCTAssertNil(json["text"])
        XCTAssertNil(json["tokenIDs"])
        let calls = await engine.calls
        XCTAssertEqual(calls, [[10, 11], [65], [66], [12], [67]])
        _ = try await session.generate(messages: [.init(role: .user, content: "different")], maximumNewTokens: 1)
        let finalCalls = await engine.calls
        XCTAssertEqual(finalCalls.last, [13, 14])
    }

    func testConcurrentGenerationIsRejectedAndCancellationClearsPartialState() async throws {
        let engine = ProbeEngine()
        let session = OrnithPerformanceSession(testingEngine: engine, codec: ProbeCodec())
        let gate = DeltaGate()
        let cancellation = PerformanceCancellation()
        let first = Task {
            try await session.generate(messages: [.init(role: .user, content: "first")], maximumNewTokens: 8,
                                       cancellation: cancellation, onDelta: { _ in await gate.hold() })
        }
        await gate.waitUntilEntered()
        do {
            _ = try await session.generate(messages: [.init(role: .user, content: "other")], maximumNewTokens: 1)
            XCTFail("concurrent request must be refused")
        } catch let error as OrnithPerformanceError {
            XCTAssertEqual(error, .io("a generation is already active"))
        }
        cancellation.cancel()
        await gate.release()
        let partial = try await first.value
        XCTAssertEqual(partial.report.finishReason, .cancelled)
        XCTAssertEqual(partial.text, "A")
        _ = try await session.generate(messages: [.init(role: .user, content: "first")], maximumNewTokens: 1)
        let calls = await engine.calls
        XCTAssertEqual(calls, [[10, 11], [10, 11]])
    }

    func testGreedyTieBreakAndNonfiniteLogitsDoNotSilentlyAlterSampling() throws {
        XCTAssertEqual(try OrnithPerformanceSession.greedy([0.5, 0.5, 0.1]), 0)
        XCTAssertThrowsError(try OrnithPerformanceSession.greedy([.nan, 0]))
        XCTAssertThrowsError(try OrnithPerformanceSession.greedy([0, .infinity]))
        XCTAssertThrowsError(try OrnithPerformanceSession.greedy([]))
    }
}

private struct ProbeCodec: OrnithChatCoding {
    let endTokenIDs: Set<Int> = [2]
    func promptTokens(messages: [OrnithChatMessage], thinking: Bool) throws -> [Int] {
        switch messages.last?.content {
        case "first": return [10, 11]
        case "continuation": return [10, 11, 65, 66, 12]
        case "too long": return Array(0..<40)
        default: return [13, 14]
        }
    }
    func decode(_ tokens: [Int]) -> String { String(String.UnicodeScalarView(tokens.compactMap { UnicodeScalar($0) })) }
}

private actor ProbeEngine: PerformanceForwarding {
    let contextCapacity = 32
    var position = 0
    var calls: [[Int]] = []
    let outputs = [65, 66, 2, 67, 2, 68, 2]
    func reset() { position = 0 }
    func forward(_ tokens: [Int], phase: PerformanceForwardMetrics.Phase,
                 cancellation: PerformanceCancellation) throws -> PerformanceForwardResult {
        if cancellation.isCancelled { throw OrnithPerformanceError.cancelled }
        let next = outputs[min(calls.count, outputs.count - 1)]
        calls.append(tokens)
        position += tokens.count
        var logits = [Float](repeating: 0, count: 128)
        logits[next] = 1
        return .init(logits: logits, metrics: .init(phase: phase, inputTokens: tokens.count, completed: true,
            wallSeconds: 0.01, metalGPUSeconds: nil, gpuTimedCommandBuffers: 0, gpuUntimedCommandBuffers: 0,
            commandBuffers: 0, dispatches: 0, encodeSeconds: 0, blockingWaitSeconds: 0, expertFetchWallSeconds: 0,
            expertReadCalls: 0, expertReturnedReadBytes: 0, expertSummedReadSeconds: 0, expertCacheHits: 0,
            expertCacheMisses: 0, expertCacheEvictions: 0, expertCacheAllocatedBytes: 0, expertCacheBudgetBytes: 0,
            cacheReplacementCount: 0, deviceMetalAllocatedBytes: 0, endDevice: nil,
            sampledPeak: .init(maximumObservedFootprintBytes: 0, minimumAvailableProcessBytes: nil,
                               samples: 0, thermalStatesObserved: []), kernelMode: .original))
    }
}

private actor DeltaGate {
    var entered = false
    var observer: CheckedContinuation<Void, Never>?
    var held: CheckedContinuation<Void, Never>?
    func hold() async {
        entered = true
        observer?.resume(); observer = nil
        await withCheckedContinuation { held = $0 }
    }
    func waitUntilEntered() async {
        if entered { return }
        await withCheckedContinuation { observer = $0 }
    }
    func release() { held?.resume(); held = nil }
}
