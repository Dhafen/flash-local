import Foundation
import MinirunKit
import XCTest
@testable import ModelAdapters

final class Qwen4ExpCPUReferenceTests: XCTestCase {
    private struct Fixture: Decodable {
        let fileName: String
        let fileByteCount: UInt64
        let sha256: String
    }
    private struct Golden: Decodable {
        let tokens: [Int]
        let logits: [[Float]]
    }
    private final class Flag: @unchecked Sendable {
        let lock = NSLock(); private var value = false
        func cancel() { lock.lock(); value = true; lock.unlock() }
        func requested() -> Bool { lock.lock(); defer { lock.unlock() }; return value }
    }
    func testMappedCPUBridgeMatchesUnchangedReferenceAndStopsBeforeLoadingWhenCancelled() throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["GGUF_HEADER_FIXTURE"] ?? env["TEST_RUNNER_GGUF_HEADER_FIXTURE"] else {
            throw XCTSkip("Set GGUF_HEADER_FIXTURE and run the synthetic reference")
        }
        let folder = URL(fileURLWithPath: path).deletingLastPathComponent()
        let reference = folder.appendingPathComponent("llama-reference-logits.json")
        guard FileManager.default.fileExists(atPath: reference.path) else { throw XCTSkip("Run scripts/run_synthetic_reference.py") }
        let fixture = try JSONDecoder().decode(Fixture.self,
            from: Data(contentsOf: folder.appendingPathComponent("synthetic-qwen4exp.json")))
        XCTAssertLessThan(fixture.fileByteCount, 1 << 20)
        let artifact = try VerifiedGGUFArtifact.verify(insideRoot: folder,
            manifest: GGUFModelManifest(fileName: fixture.fileName, fileByteCount: fixture.fileByteCount, sha256: fixture.sha256))
        let golden = try JSONDecoder().decode(Golden.self, from: Data(contentsOf: reference))
        let flag = Flag()
        let cpu = try Qwen4ExpCPUReference(artifact: artifact, contextTokens: 32,
            budgetBytes: 5_500_000_000, cancellationRequested: { flag.requested() })
        XCTAssertEqual(cpu.vocabularySize, golden.logits.first?.count)
        for (position, token) in golden.tokens.enumerated() {
            let actual = try cpu.logits(for: token)
            XCTAssertEqual(actual.count, golden.logits[position].count)
            let difference = zip(actual, golden.logits[position]).map { abs($0 - $1) }.max() ?? .infinity
            XCTAssertLessThanOrEqual(difference, 1e-5, "CPU mapped/lazy wrapper differs at position \(position)")
        }
        XCTAssertGreaterThan(cpu.maximumObservedFootprintBytes, 0)
        flag.cancel()
        XCTAssertThrowsError(try cpu.logits(for: 1)) { XCTAssertTrue($0 is CancellationError) }
        XCTAssertThrowsError(try Qwen4ExpCPUReference(artifact: artifact, cancellationRequested: { true })) {
            XCTAssertTrue($0 is CancellationError)
        }
        XCTAssertThrowsError(try Qwen4ExpCPUReference(artifact: artifact, contextTokens: 65536))
        XCTAssertThrowsError(try Qwen4ExpCPUReference(artifact: artifact, budgetBytes: 1))
        print("REAM_CPU_REFERENCE_BRIDGE positions=\(golden.tokens.count) samePinnedKernels=true lazyMode=on mmap=true noGPU=true")
    }
}
