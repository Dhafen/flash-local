import Foundation
import MinirunKit
import XCTest
@testable import MinirunRunners

final class Qwen4ExpLocalParityTests: XCTestCase {
    private struct Fixture: Decodable {
        let fileName: String
        let fileByteCount: UInt64
        let sha256: String
        let tokens: [Int]
    }
    func testLogitComparisonRejectsWrongScaleEvenWithMatchingTopTokenAndRejectsNaN() throws {
        let wrong = try Qwen4ExpLocalParity.compare(position: 0, tokenID: 1, reference: [1, 2, 3], actual: [2, 4, 6])
        XCTAssertEqual(wrong.referenceTopTokenID, wrong.streamedTopTokenID)
        XCTAssertFalse(wrong.withinNumericalTolerance)
        XCTAssertEqual(wrong.relativeL2Error, 1, accuracy: 1e-10)
        let near = try Qwen4ExpLocalParity.compare(position: 0, tokenID: 1, reference: [1, 2, 3], actual: [1.00001, 2, 3])
        XCTAssertTrue(near.withinNumericalTolerance)
        XCTAssertThrowsError(try Qwen4ExpLocalParity.compare(position: 0, tokenID: 1, reference: [1], actual: [.nan]))
        XCTAssertThrowsError(try Qwen4ExpLocalParity.compare(position: 0, tokenID: 1, reference: [1], actual: [1, 2]))
    }

    func testSequentialLocalReferenceAndStreamedPassAgreeOnBoundedSyntheticGraph() throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["GGUF_HEADER_FIXTURE"] ?? env["TEST_RUNNER_GGUF_HEADER_FIXTURE"] else {
            throw XCTSkip("Set GGUF_HEADER_FIXTURE and generate the synthetic graph")
        }
        let folder = URL(fileURLWithPath: path).deletingLastPathComponent()
        let fixture = try JSONDecoder().decode(Fixture.self,
            from: Data(contentsOf: folder.appendingPathComponent("synthetic-qwen4exp.json")))
        XCTAssertLessThan(fixture.fileByteCount, 1 << 20)
        let artifact = try VerifiedGGUFArtifact.verify(insideRoot: folder,
            manifest: GGUFModelManifest(fileName: fixture.fileName, fileByteCount: fixture.fileByteCount, sha256: fixture.sha256))
        var phases: [Qwen4ExpLocalParity.Phase] = []
        let report = try Qwen4ExpLocalParity.run(artifact: artifact, tokenIDs: fixture.tokens,
            budgetBytes: 5_500_000_000, rawCacheBytes: 2 << 20, onProgress: { phase, _, _ in phases.append(phase) })
        XCTAssertTrue(report.passedAtTestedPositions)
        XCTAssertEqual(report.tokenIDs, fixture.tokens)
        XCTAssertEqual(report.positions.count, fixture.tokens.count)
        XCTAssertEqual(report.modelSHA256, fixture.sha256)
        XCTAssertEqual(phases, Array(repeating: .cpuReference, count: fixture.tokens.count + 1)
            + Array(repeating: .streamedMLX, count: fixture.tokens.count + 1))
        XCTAssertGreaterThan(report.cpuMaximumObservedFootprintBytes, 0)
        XCTAssertGreaterThan(report.streamedMaximumObservedFootprintBytes, 0)
        XCTAssertGreaterThan(report.readerBytes, 0)
        XCTAssertGreaterThan(report.projections.calls, 0)
        let restored = try JSONDecoder().decode(Qwen4ExpLocalParity.Report.self, from: JSONEncoder().encode(report))
        XCTAssertTrue(restored.passedAtTestedPositions)
        XCTAssertEqual(restored.tokenIDs, fixture.tokens)
        XCTAssertEqual(restored.schemaVersion, 1)
        XCTAssertNotNil(ISO8601DateFormatter().date(from: restored.timestamp))
        XCTAssertEqual(restored.timestamp, report.timestamp)
        XCTAssertEqual(restored.environment.operatingSystem, ProcessInfo.processInfo.operatingSystemVersionString)
        XCTAssertEqual(restored.environment.physicalMemoryBytes, ProcessInfo.processInfo.physicalMemory)
        #if targetEnvironment(simulator)
        XCTAssertEqual(restored.environment.platform, "iOSSimulator")
        #elseif os(iOS)
        XCTAssertEqual(restored.environment.platform, "iOSDevice")
        #elseif os(macOS)
        XCTAssertEqual(restored.environment.platform, "macOS")
        #endif
        XCTAssertThrowsError(try Qwen4ExpLocalParity.run(artifact: artifact, tokenIDs: []))
        XCTAssertThrowsError(try Qwen4ExpLocalParity.run(artifact: artifact, tokenIDs: Array(repeating: 1, count: 33)))
        XCTAssertThrowsError(try Qwen4ExpLocalParity.run(artifact: artifact, tokenIDs: [-1]))
        XCTAssertThrowsError(try Qwen4ExpLocalParity.run(artifact: artifact, tokenIDs: fixture.tokens, cancellationRequested: { true })) {
            XCTAssertTrue($0 is CancellationError)
        }
        print("REAM_LOCAL_PARITY positions=\(report.positions.count) cpuThenMLX=true allLogitsCompared=true passed=true syntheticOnly=true")
    }
}
