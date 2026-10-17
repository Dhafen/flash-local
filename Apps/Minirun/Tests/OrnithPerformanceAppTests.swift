import Foundation
import XCTest
@testable import MinirunApp

@MainActor
final class OrnithPerformanceAppTests: XCTestCase {
    func testAlternateRunnerExcludesGGUFUntilItsOwnerHasClosed() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let app = REAMAppModel(root: root)
        XCTAssertTrue(app.canOpenAlternateRunner)
        await app.pauseForAlternateRunner()
        XCTAssertTrue(app.alternateRunnerActive)
        XCTAssertFalse(app.canChat)
        XCTAssertFalse(app.canRunNumerics)
        XCTAssertFalse(app.canSelectModel)
        XCTAssertFalse(app.canOpenAlternateRunner)
        app.resumeAfterAlternateRunner()
        XCTAssertFalse(app.alternateRunnerActive)
        XCTAssertTrue(app.canOpenAlternateRunner)
    }

    func testMissingQPackDoesNotEnableSendOrTouchExistingModelsAndChat() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let documents = root.appendingPathComponent("Documents")
        let models = documents.appendingPathComponent("Models")
        try FileManager.default.createDirectory(at: models, withIntermediateDirectories: true)
        let existing = models.appendingPathComponent("existing-model.gguf")
        let sentinel = Data("keep existing models".utf8)
        try sentinel.write(to: existing)
        let privateRoot = root.appendingPathComponent("ApplicationSupport/OrnithPerformance")
        let app = OrnithPerformanceAppModel(privateRoot: privateRoot, documents: documents)
        app.prepare()
        XCTAssertNil(app.error)
        XCTAssertFalse(app.isReady)
        XCTAssertFalse(app.canSend)
        XCTAssertFalse(app.canCheckKernels)
        app.checkAndActivateKernels()
        app.useOriginalKernels()
        XCTAssertEqual(app.kernelMode, .original)
        XCTAssertNil(app.kernelCheck)
        XCTAssertFalse(app.hasInstalledFiles)
        XCTAssertEqual(app.expectedFileBytes, 19_509_024_201)
        app.draft = "A refused message"
        app.send()
        XCTAssertTrue(app.messages.isEmpty)
        app.saveMeasurement()
        XCTAssertFalse(FileManager.default.fileExists(atPath: documents.appendingPathComponent("Diagnostics").path))
        await app.shutdown()
        XCTAssertFalse(app.isBusy)
        XCTAssertEqual(try Data(contentsOf: existing), sentinel)
        XCTAssertFalse(FileManager.default.fileExists(atPath: privateRoot.appendingPathComponent("qpack-chat.json").path))
    }
    func testPreviewDoesNotRunKernelCheckOrEnableSending() async {
        let app = OrnithPerformanceAppModel.reviewPreview()
        XCTAssertTrue(app.isPreview)
        XCTAssertFalse(app.canCheckKernels)
        XCTAssertFalse(app.canSend)
        app.checkAndActivateKernels()
        app.useOriginalKernels()
        XCTAssertEqual(app.kernelMode, .original)
        XCTAssertNil(app.kernelCheck)
        XCTAssertFalse(app.isBusy)
        await app.shutdown()
        XCTAssertFalse(app.isBusy)
    }
}
