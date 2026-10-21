#if canImport(OrnithLES)
import Foundation
import XCTest
import MinirunKit
import OrnithLES
import SwiftUI
#if os(macOS)
import AppKit
#endif
@testable import MinirunApp

@MainActor
final class OrnithLESAppTests: XCTestCase {
    func testActiveOwnerBootstrapsModelStorageWithoutCreatingUnverifiedDerivedFiles() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("les-first-start-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let models = root.appendingPathComponent("Models"), privateRoot = root.appendingPathComponent("private")
        let app = OrnithLESAppModel(modelsRoot: models, privateRoot: privateRoot, documents: root)
        app.activateChat()
        for _ in 0..<100 {
            if !app.isBusy { break }
            try await Task.sleep(for: .milliseconds(20))
        }
        XCTAssertFalse(app.isBusy)
        XCTAssertTrue(FileManager.default.fileExists(atPath: models.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: privateRoot.path))
        XCTAssertFalse(app.chatReady); XCTAssertTrue(app.chatMessages.isEmpty)
        await app.shutdown()
    }
    #if os(macOS)
    func testLESRootDoesNotPrepareTheInvisibleLegacyModelOwner() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("inactive-legacy-" + UUID().uuidString)
        let legacy = REAMAppModel(root: root)
        let host = NSHostingView(rootView: REAMRootView(model: legacy))
        let window = NSWindow(contentRect: CGRect(x: 0, y: 0, width: 430, height: 900),
            styleMask: .borderless, backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false; window.contentView = host; window.orderFront(nil)
        defer { window.close(); try? FileManager.default.removeItem(at: root) }
        host.layoutSubtreeIfNeeded()
        // Give the real .task lifecycle an opportunity to run. The old root
        // creates this directory and starts verification even though hidden.
        try await Task.sleep(for: .milliseconds(250))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.path))
        XCTAssertFalse(legacy.isBusy); XCTAssertNil(legacy.error)
    }
    #endif
    func testSamplingSurvivesReasoningToggleAndClosingReopeningTheModel() throws {
        let suite = "OrnithSampling-" + UUID().uuidString
        let preferences = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { preferences.removePersistentDomain(forName: suite) }
        var model: OrnithLESAppModel? = .live(preferences: preferences)
        model?.chatOptions.temperature = 0.7; model?.chatOptions.topP = 0.93; model?.chatOptions.seed = 1234
        model?.chatThinkingEnabled = false
        XCTAssertEqual(model?.chatOptions.temperature, 0.7)
        model = nil
        let reopened = OrnithLESAppModel.live(preferences: preferences)
        XCTAssertEqual(reopened.chatOptions.temperature, 0.7)
        XCTAssertEqual(reopened.chatOptions.topP, 0.93); XCTAssertEqual(reopened.chatOptions.seed, 1234)
        XCTAssertFalse(reopened.chatThinkingEnabled)
        reopened.chatThinkingEnabled = true
        XCTAssertEqual(reopened.chatOptions.temperature, 0.7)
    }
    func testWorkProfilesOnlyChangeEditableSystemInstruction() {
        let root = FileManager.default.temporaryDirectory
        let model = OrnithLESAppModel(modelsRoot: root, privateRoot: root, documents: root)
        let sampling = model.chatOptions, batch = model.batchConfiguration
        model.applyProfile(.programming)
        XCTAssertEqual(model.systemPrompt, LocalChatProfile.programming.systemPrompt)
        XCTAssertEqual(model.chatOptions, sampling); XCTAssertEqual(model.batchConfiguration, batch)
        model.applyProfile(.securityReview)
        XCTAssertTrue(model.systemPrompt.contains("supplied evidence"))
        model.applyProfile(.unchanged); XCTAssertEqual(model.systemPrompt, "")
        XCTAssertFalse(model.chatReady); XCTAssertTrue(model.chatMessages.isEmpty)
    }
    func testPendingBatchSelectionCannotSilentlyUsePreviouslyLoadedGeometry() {
        let model = OrnithLESAppModel.reviewChatPreview(thinking: false)
        XCTAssertFalse(model.batchSettingsChanged)
        model.batchConfiguration.evaluationTokens = 32; model.batchConfiguration.physicalTokens = 8
        XCTAssertTrue(model.batchSettingsChanged)
        model.startAPI(); XCTAssertTrue(model.apiAccessKey.isEmpty)
        model.batchConfiguration = .init(); XCTAssertFalse(model.batchSettingsChanged)
    }
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
