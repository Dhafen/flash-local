import Foundation
import XCTest
@testable import OrnithPerformance

final class PerformanceKernelCheckTests: XCTestCase {
    func testExactReplayAndFinalReset() throws {
        let driver = KernelCheckDriver()
        let report = try driver.run()
        XCTAssertTrue(report.passed)
        XCTAssertTrue(report.greedyChoicesMatch)
        XCTAssertEqual(report.comparedLogitVectors, 4)
        XCTAssertEqual(report.mismatchedLogitValues, 0)
        XCTAssertEqual(driver.calls.count, 8)
        XCTAssertEqual(driver.resets, [.original, .groupedExperts, .original])
        XCTAssertEqual(driver.mode, .original)
        for index in 0..<4 {
            XCTAssertEqual(driver.calls[index].tokens, driver.calls[index + 4].tokens)
            XCTAssertEqual(driver.calls[index].phase, driver.calls[index + 4].phase)
        }
        XCTAssertFalse(report.fullModelNumericsVerified)
    }

    func testEvenSubArgmaxDifferenceFailsClosed() throws {
        let driver = KernelCheckDriver()
        driver.change = .small
        let report = try driver.run()
        XCTAssertFalse(report.passed)
        XCTAssertTrue(report.greedyChoicesMatch)
        XCTAssertEqual(report.mismatchedLogitValues, 1)
        XCTAssertEqual(report.maximumAbsoluteLogitDifference, 0.125)
        XCTAssertEqual(driver.mode, .original)
    }

    func testSignedZeroIsNotSilentlyAcceptedAsBitwiseParity() throws {
        let driver = KernelCheckDriver()
        driver.change = .signedZero
        let report = try driver.run()
        XCTAssertFalse(report.passed)
        XCTAssertEqual(report.maximumAbsoluteLogitDifference, 0)
        XCTAssertEqual(report.mismatchedLogitValues, 1)
    }

    func testInvalidOrFailedForwardAlwaysRestoresOriginal() throws {
        for change in [KernelCheckDriver.Change.nan, .shape, .error] {
            let driver = KernelCheckDriver()
            driver.change = change
            XCTAssertThrowsError(try driver.run())
            XCTAssertEqual(driver.mode, .original)
            XCTAssertEqual(driver.resets.last, .original)
        }
        let driver = KernelCheckDriver()
        driver.cancelAtCall = 5
        XCTAssertThrowsError(try driver.run())
        XCTAssertEqual(driver.mode, .original)
        XCTAssertEqual(driver.calls.count, 5)
    }

    func testBoundedInputsRefuseWorkBeforeAForward() throws {
        for (prompt, steps) in [([], 3), ([1], 0), ([1], 4), (Array(repeating: 1, count: 33), 3)] {
            let driver = KernelCheckDriver()
            XCTAssertThrowsError(try driver.run(prompt: prompt, steps: steps))
            XCTAssertTrue(driver.calls.isEmpty)
            XCTAssertEqual(driver.mode, .original)
        }
    }

    func testExportContainsOnlyAggregateCheckEvidence() throws {
        let report = try KernelCheckDriver().run()
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(report)) as? [String: Any])
        for key in ["prompt", "answer", "promptTokenIDs", "generatedTokenIDs", "logits"] {
            XCTAssertNil(object[key])
        }
        XCTAssertEqual(object["fullModelNumericsVerified"] as? Bool, false)
        XCTAssertEqual(report.runtimeRevision, OrnithPerformanceEngine.swiftletRuntimeRevision)
        print("ORNITH_KERNEL_CHECK exactReplay=true failClosed=true noChatPayload=true syntheticOnly=true")
    }
}

private final class KernelCheckDriver {
    enum Change { case none, small, signedZero, nan, shape, error }
    var change: Change = .none
    var mode: PerformanceKernelMode = .original
    var resets: [PerformanceKernelMode] = []
    var calls: [(tokens: [Int], phase: PerformanceForwardMetrics.Phase)] = []
    var stepIndex = 0
    var cancelled = false
    var cancelAtCall: Int?

    func run(prompt: [Int] = [1, 2], steps: Int = 3) throws -> PerformanceKernelCheck {
        try PerformanceKernelProbe.run(prompt: prompt, sourceIdentity: "synthetic-checkpoint", decodeSteps: steps,
            shouldCancel: { self.cancelled }, resetMode: { mode in
                self.mode = mode; self.stepIndex = 0; self.resets.append(mode)
            }, forward: { tokens, phase in try self.forward(tokens, phase) })
    }

    func forward(_ tokens: [Int], _ phase: PerformanceForwardMetrics.Phase) throws -> PerformanceForwardResult {
        calls.append((tokens, phase))
        if calls.count == cancelAtCall { cancelled = true }
        var logits = [Float](repeating: 0, count: 8)
        logits[(stepIndex + 3) % 8] = 10
        if mode == .groupedExperts && stepIndex == 1 {
            switch change {
            case .none: break
            case .small: logits[0] = 0.125
            case .signedZero: logits[0] = -0.0
            case .nan: logits[0] = .nan
            case .shape: logits.append(0)
            case .error: throw OrnithPerformanceError.io("synthetic read failure")
            }
        }
        stepIndex += 1
        return .init(logits: logits, metrics: .init(phase: phase, inputTokens: tokens.count, completed: true,
            wallSeconds: 0.01, metalGPUSeconds: nil, gpuTimedCommandBuffers: 0, gpuUntimedCommandBuffers: 0,
            commandBuffers: 0, dispatches: 0, encodeSeconds: 0, blockingWaitSeconds: 0, expertFetchWallSeconds: 0,
            expertReadCalls: 0, expertReturnedReadBytes: 0, expertSummedReadSeconds: 0, expertCacheHits: 0,
            expertCacheMisses: 0, expertCacheEvictions: 0, expertCacheAllocatedBytes: 0, expertCacheBudgetBytes: 0,
            cacheReplacementCount: 0, deviceMetalAllocatedBytes: 0, endDevice: nil,
            sampledPeak: .init(maximumObservedFootprintBytes: 0, minimumAvailableProcessBytes: nil,
                               samples: 0, thermalStatesObserved: []), kernelMode: mode))
    }
}
