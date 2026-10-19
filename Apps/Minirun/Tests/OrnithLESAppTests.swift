#if canImport(OrnithLES)
import Foundation
import XCTest
import MinirunKit
import OrnithLES
@testable import MinirunApp

@MainActor
final class OrnithLESAppTests: XCTestCase {
    func testFinishedAPILeaseDoesNotCancelAReusableNativeOwner() {
        let completed = LESControl(), lease = LESRequestLease(completed)
        lease.finish(); lease.cancel()
        XCTAssertFalse(completed.isCancellationRequested)
        let running = LESControl(), active = LESRequestLease(running)
        active.cancel(); active.finish(); active.cancel()
        XCTAssertTrue(running.isCancellationRequested)
    }
    func testAPICannotListenWithoutAdmittedModelAndPreviewsDoNotOpenSockets() async {
        let model = OrnithLESAppModel.reviewPreview()
        model.startAPI(); XCTAssertFalse(model.isAPIRunning); XCTAssertTrue(model.apiAccessKey.isEmpty)
        model.stopAPI(); await model.shutdown()
        XCTAssertTrue(model.apiURLs.isEmpty); XCTAssertFalse(model.shouldKeepAwake)
        let missing = OrnithLESAppModel.live()
        missing.startAPI(); XCTAssertFalse(missing.isAPIRunning); XCTAssertTrue(missing.apiAccessKey.isEmpty)
    }
    func testMissingModelDoesNotAlterExistingModelsOrEnableAnotherOwner() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("les-app-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        let sentinel = root.appendingPathComponent("keep.txt")
        let original = Data("preserve existing model and chat".utf8); try original.write(to: sentinel)
        let app = OrnithLESAppModel(modelsRoot: root, privateRoot: root.appendingPathComponent("private"), documents: root)
        XCTAssertTrue(app.canStart); app.start(); XCTAssertFalse(app.canStart)
        await app.shutdown()
        XCTAssertFalse(app.isBusy); XCTAssertTrue(app.canStart)
        XCTAssertTrue(app.runs.isEmpty); XCTAssertNil(app.numericalCheck)
        XCTAssertEqual(try Data(contentsOf: sentinel), original)
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("private").path))
        print("ORNITH_LES_APP visibleEntry=true guardedOwnership=true previewCannotInfer=true trainedInference=false")
    }
    func testPreviewDoesNotPrepareWeightsOrStartAWorker() async {
        let model = OrnithLESAppModel.reviewPreview()
        XCTAssertFalse(model.canStart); model.start(); model.stop(); await model.shutdown()
        XCTAssertFalse(model.isBusy); XCTAssertTrue(model.runs.isEmpty)
        XCTAssertNil(model.measurementURL); XCTAssertNil(model.numericalCheck)
    }
    func testMissingChatModelCannotGenerateOrChangeOrdinaryFiles() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("les-chat-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent("chat.json"), bytes = Data("keep ordinary history".utf8)
        try bytes.write(to: file)
        let app = OrnithLESAppModel(modelsRoot: root, privateRoot: root.appendingPathComponent("private"), documents: root)
        app.chatDraft = "Hej"; app.sendChat()
        XCTAssertTrue(app.chatMessages.isEmpty); XCTAssertFalse(app.canSendChat)
        app.activateChat(); XCTAssertTrue(app.isBusy); XCTAssertFalse(app.canStart)
        await app.shutdown()
        XCTAssertFalse(app.chatReady); XCTAssertFalse(app.isBusy); XCTAssertFalse(app.canSendChat)
        XCTAssertEqual(try Data(contentsOf: file), bytes)
        XCTAssertNil(app.chatSummary); XCTAssertNil(app.chatMeasurementURL)
        print("ORNITH_CHAT_APP missingModelRejected=true sameOwner=true ordinaryHistoryPreserved=true previewCannotInfer=true")
    }
}
#endif
