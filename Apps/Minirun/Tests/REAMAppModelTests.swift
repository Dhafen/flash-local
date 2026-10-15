import Foundation
import MinirunKit
import ModelAdapters
import MinirunRunners
import XCTest
@testable import MinirunApp

@MainActor
final class REAMAppModelTests: XCTestCase {
    private struct Fixture: Decodable {
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
        let decodedHex: String
    }

    func testVerifiedLocalModelGeneratesReferenceTextPersistsAndRecoversAfterStop() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["GGUF_HEADER_FIXTURE"] ?? env["TEST_RUNNER_GGUF_HEADER_FIXTURE"] else {
            throw XCTSkip("Set GGUF_HEADER_FIXTURE and run the synthetic text-session reference")
        }
        let fixtures = URL(fileURLWithPath: path).deletingLastPathComponent()
        let fixture = try JSONDecoder().decode(Fixture.self,
            from: Data(contentsOf: fixtures.appendingPathComponent("synthetic-qwen4exp-chat.json")))
        let reference = try JSONDecoder().decode(Reference.self,
            from: Data(contentsOf: fixtures.appendingPathComponent("qwen4exp-session-reference.json")))
        XCTAssertLessThan(fixture.fileByteCount, 1 << 20, "Only the untrained tiny fixture belongs in CI")
        XCTAssertEqual(fixture.messages.count, 1)
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let root = folder.appendingPathComponent("Models")
        try LocalModelStorage.prepare(root: root)
        defer { try? FileManager.default.removeItem(at: folder) }
        let localFile = root.appendingPathComponent(fixture.fileName)
        try FileManager.default.copyItem(at: fixtures.appendingPathComponent(fixture.fileName), to: localFile)
        let source = try PinnedGGUFDownload(url: URL(string: "https://example.invalid/synthetic.gguf")!,
            manifest: GGUFModelManifest(fileName: fixture.fileName, fileByteCount: fixture.fileByteCount, sha256: fixture.sha256))
        let file = folder.appendingPathComponent("chat.json")
        let model = REAMAppModel(root: root, source: source, chatFile: file,
            settings: REAMRunSettings(contextTokens: fixture.contextTokens, budgetBytes: 4_000_000_000,
                maximumNewTokens: fixture.maximumNewTokens, rawCacheBytes: 2 << 20))
        model.prepare(); await model.waitUntilIdle()
        XCTAssertNil(model.error); XCTAssertTrue(model.canChat)
        model.draft = fixture.messages[0].content; model.send(); model.stop()
        XCTAssertTrue(model.isBusy); XCTAssertFalse(model.canChat)
        await model.waitUntilIdle()
        XCTAssertFalse(model.isBusy); XCTAssertTrue(model.canChat)
        XCTAssertTrue(try XCTUnwrap(model.messages.last).interrupted)
        model.clearMessages()
        model.draft = fixture.messages[0].content; model.send()
        await model.waitUntilIdle()
        XCTAssertNil(model.error)
        let summary = try XCTUnwrap(model.summary)
        XCTAssertEqual(summary.promptTokenIDs, reference.promptTokenIDs)
        XCTAssertEqual(summary.generatedTokenIDs, reference.generatedTokenIDs)
        XCTAssertEqual(summary.modelSHA256, fixture.sha256)
        XCTAssertGreaterThan(summary.ioBytesRead, 0)
        XCTAssertGreaterThan(summary.maximumObservedFootprintBytes, 0)
        var bytes = Data(); var cursor = reference.decodedHex.startIndex
        while cursor < reference.decodedHex.endIndex {
            let next = reference.decodedHex.index(cursor, offsetBy: 2)
            bytes.append(try XCTUnwrap(UInt8(reference.decodedHex[cursor..<next], radix: 16))); cursor = next
        }
        XCTAssertEqual(model.messages.count, 2)
        XCTAssertEqual(model.messages.last?.text, String(decoding: bytes, as: UTF8.self))
        XCTAssertEqual(model.messages.last?.interrupted, false)
        let saved = try JSONDecoder().decode([REAMMessage].self, from: Data(contentsOf: file))
        XCTAssertEqual(saved.map(\.text), model.messages.map(\.text))
        model.saveMeasurement(); await model.waitUntilIdle()
        XCTAssertNil(model.error)
        let measurement = try XCTUnwrap(model.lastMeasurementURL)
        XCTAssertEqual(measurement.deletingLastPathComponent().lastPathComponent, "Diagnostics")
        let report = try JSONDecoder().decode(REAMRunReport.self, from: Data(contentsOf: measurement))
        XCTAssertEqual(report.generation.generatedTokenIDs, reference.generatedTokenIDs)
        XCTAssertEqual(report.generation.promptTokenIDs, reference.promptTokenIDs)
        XCTAssertEqual(report.generation.modelSHA256, fixture.sha256)
        XCTAssertEqual(report.contextTokens, fixture.contextTokens)
        XCTAssertEqual(report.budgetBytes, 4_000_000_000)
        XCTAssertEqual(report.modelFileBytes, fixture.fileByteCount)
        XCTAssertNil(report.modelProvenance, "Synthetic fixtures have no declared trained-model provenance")
        XCTAssertFalse(report.airplaneModeVerified)
        XCTAssertFalse(report.fullModelNumericsVerified)
        #if os(macOS)
        XCTAssertEqual(report.platform, "macOS", "A hosted Mac test must not label itself an iPhone measurement")
        #endif
        model.checkNumerics(); await model.waitUntilIdle()
        XCTAssertNil(model.error)
        let parityURL = try XCTUnwrap(model.lastNumericsURL)
        let parity = try JSONDecoder().decode(Qwen4ExpLocalParity.Report.self, from: Data(contentsOf: parityURL))
        XCTAssertTrue(parity.passedAtTestedPositions)
        XCTAssertEqual(parity.tokenIDs, reference.promptTokenIDs)
        XCTAssertEqual(parity.modelSHA256, fixture.sha256)
        XCTAssertNil(parity.modelProvenance)
        XCTAssertEqual(parity.positions.count, reference.promptTokenIDs.count)
        XCTAssertEqual(model.messages.map(\.text), saved.map(\.text), "A diagnostic must not invent chat turns")
        try FileManager.default.removeItem(at: localFile)
        model.refreshStorage()
        XCTAssertFalse(model.canChat)
        print("REAM_APP_SESSION_PARITY exactPromptIDs=true exactGeneratedIDs=true exactText=true saved=true stopRecovery=true")
    }

    func testMissingModelCannotCreateOrInventChatOutput() {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let model = REAMAppModel(root: root)
        model.prepare()
        XCTAssertFalse(model.hasFile); XCTAssertFalse(model.canChat)
        model.draft = "Do not invent an answer"
        model.send()
        model.saveMeasurement()
        XCTAssertTrue(model.messages.isEmpty)
        XCTAssertEqual(model.draft, "Do not invent an answer")
        XCTAssertFalse(model.isBusy)
        XCTAssertNil(model.lastMeasurementURL)
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
        model.prepare(); model.download(); model.verify(); model.draft = "Try"; model.send(); model.checkNumerics()
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
