import Foundation
import XCTest
@testable import MinirunApp

@MainActor
final class REAMAppModelTests: XCTestCase {
    func testMissingModelCannotCreateOrInventChatOutput() {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let model = REAMAppModel(root: root)
        model.prepare()
        XCTAssertFalse(model.hasFile); XCTAssertFalse(model.canChat)
        model.draft = "Do not invent an answer"
        model.send()
        XCTAssertTrue(model.messages.isEmpty)
        XCTAssertEqual(model.draft, "Do not invent an answer")
        XCTAssertFalse(model.isBusy)
    }

    func testWrongModelSizeNeverEnablesSendAndWorkerTearsDown() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let model = REAMAppModel(root: root)
        try Data([1, 2, 3]).write(to: root.appendingPathComponent(model.source.manifest.fileName))
        model.prepare()
        XCTAssertTrue(model.isBusy)
        await model.waitUntilIdle()
        XCTAssertTrue(model.hasFile); XCTAssertFalse(model.canChat); XCTAssertFalse(model.isBusy)
        XCTAssertNotNil(model.error)
        model.draft = "Hello"; model.send()
        XCTAssertTrue(model.messages.isEmpty)
    }

    func testPreviewNeverStartsServicesOrClaimsRealReadiness() {
        let model = REAMAppModel.reviewPreview()
        let initial = model.messages.map(\.text)
        model.prepare(); model.download(); model.verify(); model.draft = "Try"; model.send()
        XCTAssertTrue(model.isPreview); XCTAssertFalse(model.canChat); XCTAssertFalse(model.isBusy)
        XCTAssertEqual(model.messages.map(\.text), initial)
        XCTAssertNil(model.summary)
    }

    func testRestoresAndExplicitlyClearsOnlyItsOwnChatFile() throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let file = folder.appendingPathComponent("chat.json")
        let original = [REAMMessage(role: .user, text: "Public test fixture"),
                        REAMMessage(role: .assistant, text: "Interrupted fixture", interrupted: true)]
        try JSONEncoder().encode(original).write(to: file)
        let model = REAMAppModel(root: folder.appendingPathComponent("Models"), chatFile: file)
        model.prepare()
        XCTAssertEqual(model.messages.map(\.text), original.map(\.text))
        XCTAssertTrue(model.messages.last!.interrupted)
        model.clearMessages()
        XCTAssertTrue(model.messages.isEmpty)
        XCTAssertTrue(try JSONDecoder().decode([REAMMessage].self, from: Data(contentsOf: file)).isEmpty)
    }
}
