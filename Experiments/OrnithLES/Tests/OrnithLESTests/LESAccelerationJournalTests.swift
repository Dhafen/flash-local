import Foundation
import XCTest
@testable import OrnithLES

final class LESAccelerationJournalTests: XCTestCase {
    private func withJournal(_ body: (URL, LESAccelerationJournal) throws -> Void) throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("acceleration-journal-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let journal = try LESAccelerationJournal(directory: root, sourceRevision: "test-source", appBuild: "test-build",
            candidate: .dflash, contextTokens: 8192)
        try body(root, journal)
    }
    func testInterruptedNativeOperationRecoversExactPhaseWithoutInventingCause() throws {
        try withJournal { root, journal in
            XCTAssertTrue(journal.recordNativePhase(14))
            let recovered = try XCTUnwrap(LESAccelerationJournal.recoverInterrupted(directory: root))
            XCTAssertEqual(recovered.lastPhase, "draft.kv-injection")
            XCTAssertEqual(recovered.outcome, "interrupted")
            XCTAssertEqual(recovered.contextTokens, 8192)
            XCTAssertEqual(recovered.appBuild, "test-build")
            XCTAssertEqual(recovered.terminationCause, "unknown; requires OS crash or jetsam report")
            XCTAssertNotNil(recovered.events.last?.footprintBytes)
            XCTAssertNil(try LESAccelerationJournal.recoverInterrupted(directory: root))
        }
    }
    func testCompletedAndCaughtFailuresAreNotReportedAsProcessTerminations() throws {
        for failed in [true, false] {
            try withJournal { root, journal in
                try journal.finish(failed: failed)
                XCTAssertNil(try LESAccelerationJournal.recoverInterrupted(directory: root))
            }
        }
    }
    func testJournalIsBoundedAndUnknownNativePhaseFailsClosed() throws {
        try withJournal { root, journal in
            for i in 0..<100 { try journal.record("test.\(i)") }
            XCTAssertFalse(journal.recordNativePhase(999))
            let value = try XCTUnwrap(LESAccelerationJournal.recoverInterrupted(directory: root))
            XCTAssertEqual(value.events.count, 24)
            XCTAssertEqual(value.lastPhase, "test.99")
            let data = try Data(contentsOf: journal.url)
            XCTAssertLessThan(data.count, 65536)
            XCTAssertThrowsError(try journal.record("prompt: user content"))
        }
    }
    func testNativeJournalWritesAreSuppressedDuringTimedForwardLoops() throws {
        try withJournal { _, journal in
            let control = LESControl(journal: journal)
            XCTAssertTrue(control.recordNativePhase(11))
            let before = try Data(contentsOf: journal.url)
            control.setNativeDiagnosticsEnabled(false)
            XCTAssertTrue(control.recordNativePhase(14))
            XCTAssertEqual(try Data(contentsOf: journal.url), before)
            control.setNativeDiagnosticsEnabled(true)
            XCTAssertTrue(control.recordNativePhase(15))
            XCTAssertNotEqual(try Data(contentsOf: journal.url), before)
        }
    }
    func testFailedWriteStopsBeforeNativeExecution() throws {
        try withJournal { _, journal in
            try FileManager.default.removeItem(at: journal.url)
            try FileManager.default.createDirectory(at: journal.url, withIntermediateDirectories: false)
            XCTAssertFalse(journal.recordNativePhase(10))
        }
    }
}
