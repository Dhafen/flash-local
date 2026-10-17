import Foundation
import MinirunKit
import ModelAdapters
import MinirunRunners
import XCTest
@testable import MinirunApp

@MainActor
final class REAMAppModelTests: XCTestCase {
    func testThinkingDefaultsToOrnithAndDoesNotChangeREAM() {
        let root = FileManager.default.temporaryDirectory
        let ornith = REAMAppModel(root: root)
        XCTAssertTrue(ornith.thinkingEnabled); XCTAssertTrue(ornith.supportsThinking)
        ornith.setThinkingEnabled(false); XCTAssertFalse(ornith.thinkingEnabled)
        let ream = REAMAppModel(root: root, source: .primaryREAM, settings: .initial)
        ream.setThinkingEnabled(true)
        XCTAssertFalse(ream.thinkingEnabled); XCTAssertFalse(ream.supportsThinking)
    }
    func testLegacyMessagesDecodeAndReasoningIsSeparateFromCanonicalHistory() throws {
        let old = try JSONSerialization.data(withJSONObject: [["id": UUID().uuidString,
            "role": "assistant", "text": "Legacy reply", "interrupted": false]])
        let restored = try JSONDecoder().decode([REAMMessage].self, from: old)
        XCTAssertEqual(restored.first?.text, "Legacy reply")
        XCTAssertNil(restored.first?.reasoning); XCTAssertNil(restored.first?.isThinking)
        let history: [REAMMessage] = [.init(role: .user, text: "First"),
            .init(role: .assistant, text: "55", reasoning: "Public synthetic reasoning"),
            .init(role: .user, text: "Stopped"),
            .init(role: .assistant, text: "", interrupted: true, reasoning: "Unfinished synthetic reasoning"),
            .init(role: .user, text: "Next")]
        let prompt = REAMMessage.generationPrompt(history)
        XCTAssertEqual(prompt.map(\.content), ["First", "55", "Stopped", "", "Next"])
        XCTAssertEqual(prompt[1].reasoningContent, "Public synthetic reasoning")
        XCTAssertEqual(prompt[3].reasoningContent, "Unfinished synthetic reasoning")
        XCTAssertEqual(history[3].reasoning, "Unfinished synthetic reasoning")
    }
    func testOrnithThinkingUsesCanonicalTokensAndStopsWithoutAnActiveIndicator() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["GGUF_HEADER_FIXTURE"] ?? env["TEST_RUNNER_GGUF_HEADER_FIXTURE"] else {
            throw XCTSkip("Set the generated synthetic fixture directory")
        }
        let fixtures = URL(fileURLWithPath: path).deletingLastPathComponent()
        let fixture = try JSONDecoder().decode(Fixture.self,
            from: Data(contentsOf: fixtures.appendingPathComponent("synthetic-ornith-chat.json")))
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try LocalModelStorage.prepare(root: root)
        defer { try? FileManager.default.removeItem(at: root) }
        XCTAssertLessThan(fixture.fileByteCount, 1 << 20)
        try FileManager.default.copyItem(at: fixtures.appendingPathComponent(fixture.fileName),
            to: root.appendingPathComponent(fixture.fileName))
        let manifest = try GGUFModelManifest(fileName: fixture.fileName, fileByteCount: fixture.fileByteCount, sha256: fixture.sha256)
        let verified = try VerifiedGGUFArtifact.verify(insideRoot: root, manifest: manifest)
        let (prompt, expected) = try autoreleasepool { () throws -> ([Int], [Int]) in
            let tokenizer = try CanonicalQwen35Tokenizer(artifact: verified)
            let prompt = try tokenizer.chat(fixture.messages, maximumTokens: fixture.contextTokens - 3, enableThinking: true)
            XCTAssertTrue(prompt.renderedText.hasSuffix("<think>\n"))
            let reference = try Qwen4ExpCPUReference(artifact: verified, contextTokens: fixture.contextTokens, budgetBytes: 4_000_000_000)
            var logits: [Float] = [], ids: [Int] = []
            for token in prompt.tokenIDs { logits = try reference.logits(for: token) }
            for step in 0..<3 {
                var best = 0
                for i in logits.indices where logits[i] > logits[best] { best = i }
                if tokenizer.isEndOfGeneration(best) { break }
                ids.append(best)
                if step < 2 { logits = try reference.logits(for: best) }
            }
            return (prompt.tokenIDs, ids)
        }
        let source = try PinnedGGUFDownload(url: URL(string: "https://example.invalid/synthetic.gguf")!, manifest: manifest)
        XCTAssertFalse(expected.isEmpty, "The synthetic thinking session must actually generate tokens")
        let model = REAMAppModel(root: root, source: source,
            settings: REAMRunSettings(contextTokens: fixture.contextTokens, budgetBytes: 4_000_000_000,
                maximumNewTokens: 3, rawCacheBytes: 2 << 20))
        model.prepare(); await model.waitUntilIdle()
        model.setThinkingEnabled(true); XCTAssertTrue(model.thinkingEnabled)
        model.draft = try XCTUnwrap(fixture.messages.first).content; model.send()
        XCTAssertNotEqual(model.messages.last?.isThinking, true, "A waiting prompt is not generated reasoning")
        await model.waitUntilIdle()
        XCTAssertNil(model.error); XCTAssertEqual(model.summary?.promptTokenIDs, prompt)
        XCTAssertEqual(model.summary?.generatedTokenIDs, expected)
        XCTAssertEqual(model.summary?.thinkingEnabled, true); XCTAssertNotEqual(model.messages.last?.isThinking, true)
        model.draft = "Next"; model.send(); model.stop(); await model.waitUntilIdle()
        XCTAssertNotEqual(model.messages.last?.isThinking, true)
        print("THINKING_CHAT canonicalPrompt=true sameReferenceTokens=true separateChannels=true stopClearsIndicator=true syntheticOnly=true")
    }
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

    func testEmptyInterruptedTurnsDoNotCreateConsecutiveUserRoles() {
        let history: [REAMMessage] = [
            .init(role: .system, text: "Public synthetic system"),
            .init(role: .user, text: "Completed question"),
            .init(role: .assistant, text: "Completed answer"),
            .init(role: .user, text: "Stopped before output"),
            .init(role: .assistant, text: "", interrupted: true),
            .init(role: .user, text: "Another stopped attempt"),
            .init(role: .assistant, text: "", interrupted: true),
            .init(role: .user, text: "Current question")
        ]
        XCTAssertEqual(REAMMessage.generationPrompt(history), [
            .init(role: .system, content: "Public synthetic system"),
            .init(role: .user, content: "Completed question"),
            .init(role: .assistant, content: "Completed answer"),
            .init(role: .user, content: "Current question")
        ])
        XCTAssertEqual(history.count, 8, "Context recovery must preserve visible failed attempts")
    }

    func testPromptKeepsRealPartialAnswersAndRecoversLegacyUnansweredUsers() {
        let history: [REAMMessage] = [
            .init(role: .user, text: "First question"),
            .init(role: .assistant, text: "A real partial answer", interrupted: true),
            .init(role: .user, text: "Old unanswered question"),
            .init(role: .user, text: "Current question")
        ]
        XCTAssertEqual(REAMMessage.generationPrompt(history), [
            .init(role: .user, content: "First question"),
            .init(role: .assistant, content: "A real partial answer"),
            .init(role: .user, content: "Current question")
        ])
        XCTAssertEqual(REAMMessage.generationPrompt([
            .init(role: .user, text: "Question"), .init(role: .assistant, text: ""),
            .init(role: .user, text: "Next")
        ]).map(\.role), [.user, .assistant, .user],
        "A completed empty answer is different from an interrupted placeholder")
    }

    func testRestoredEmptyFailuresGenerateWithoutClearingVisibleHistory() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["GGUF_HEADER_FIXTURE"] ?? env["TEST_RUNNER_GGUF_HEADER_FIXTURE"] else {
            throw XCTSkip("Set GGUF_HEADER_FIXTURE and run the synthetic text-session reference")
        }
        let fixtures = URL(fileURLWithPath: path).deletingLastPathComponent()
        let fixture = try JSONDecoder().decode(Fixture.self,
            from: Data(contentsOf: fixtures.appendingPathComponent("synthetic-qwen4exp-chat.json")))
        let reference = try JSONDecoder().decode(Reference.self,
            from: Data(contentsOf: fixtures.appendingPathComponent("qwen4exp-session-reference.json")))
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: folder) }
        let root = folder.appendingPathComponent("Models"), file = folder.appendingPathComponent("chat.json")
        try LocalModelStorage.prepare(root: root)
        try FileManager.default.copyItem(at: fixtures.appendingPathComponent(fixture.fileName),
            to: root.appendingPathComponent(fixture.fileName))
        let failed: [REAMMessage] = [
            .init(role: .user, text: "Stopped before output"),
            .init(role: .assistant, text: "", interrupted: true),
            .init(role: .user, text: "Stopped again"),
            .init(role: .assistant, text: "", interrupted: true)
        ]
        try JSONEncoder().encode(failed).write(to: file)
        let source = try PinnedGGUFDownload(url: URL(string: "https://example.invalid/synthetic.gguf")!,
            manifest: GGUFModelManifest(fileName: fixture.fileName, fileByteCount: fixture.fileByteCount, sha256: fixture.sha256))
        let model = REAMAppModel(root: root, source: source, chatFile: file,
            settings: REAMRunSettings(contextTokens: fixture.contextTokens, budgetBytes: 4_000_000_000,
                maximumNewTokens: fixture.maximumNewTokens, rawCacheBytes: 2 << 20))
        model.prepare(); await model.waitUntilIdle()
        XCTAssertTrue(model.canChat)
        model.draft = try XCTUnwrap(fixture.messages.first).content; model.send(); await model.waitUntilIdle()
        XCTAssertNil(model.error)
        XCTAssertEqual(model.summary?.promptTokenIDs, reference.promptTokenIDs)
        XCTAssertEqual(model.summary?.generatedTokenIDs, reference.generatedTokenIDs)
        XCTAssertEqual(model.messages.count, 6)
        XCTAssertEqual(Array(model.messages.prefix(4)).map(\.id), failed.map(\.id))
        print("CHAT_EMPTY_STOP_RECOVERY restored=true noClearRequired=true exactReferencePrompt=true visibleHistoryKept=true")
    }

    func testVerifiedLocalModelGeneratesReferenceTextPersistsAndRecoversAfterStop() async throws {
        try await checkVerifiedLocalModel(family: "qwen4exp")
    }
    func testOrnithAppRouteGeneratesAndRunsTheLocalCPUComparison() async throws {
        try await checkVerifiedLocalModel(family: "ornith")
        print("ORNITH_APP_SESSION exactPromptIDs=true exactGeneratedIDs=true localCPUComparison=true syntheticOnly=true")
    }
    private func checkVerifiedLocalModel(family: String) async throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["GGUF_HEADER_FIXTURE"] ?? env["TEST_RUNNER_GGUF_HEADER_FIXTURE"] else {
            throw XCTSkip("Set GGUF_HEADER_FIXTURE and run the synthetic text-session reference")
        }
        let fixtures = URL(fileURLWithPath: path).deletingLastPathComponent()
        let fixture = try JSONDecoder().decode(Fixture.self,
            from: Data(contentsOf: fixtures.appendingPathComponent("synthetic-\(family)-chat.json")))
        let reference = try JSONDecoder().decode(Reference.self,
            from: Data(contentsOf: fixtures.appendingPathComponent("\(family)-session-reference.json")))
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
        await model.waitUntilIdle()
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
        model.draft = "Next"; model.send(); await model.waitUntilIdle()
        XCTAssertNil(model.error)
        XCTAssertGreaterThan(try XCTUnwrap(model.summary?.reusedPromptTokens), 0,
                             "The app must keep one engine and enable exact prefix reuse between turns")
        let beforeDiagnostic = model.messages.map(\.text)
        model.checkNumerics(); await model.waitUntilIdle()
        XCTAssertNil(model.error)
        let parityURL = try XCTUnwrap(model.lastNumericsURL)
        let parity = try JSONDecoder().decode(Qwen4ExpLocalParity.Report.self, from: Data(contentsOf: parityURL))
        XCTAssertTrue(parity.passedAtTestedPositions)
        XCTAssertEqual(parity.tokenIDs, reference.promptTokenIDs)
        XCTAssertEqual(parity.modelSHA256, fixture.sha256)
        XCTAssertNil(parity.modelProvenance)
        XCTAssertEqual(parity.positions.count, reference.promptTokenIDs.count)
        XCTAssertEqual(model.messages.map(\.text), beforeDiagnostic, "A diagnostic must not invent chat turns")
        try FileManager.default.removeItem(at: localFile)
        model.refreshStorage()
        XCTAssertFalse(model.canChat)
        print("REAM_APP_SESSION_PARITY exactPromptIDs=true exactGeneratedIDs=true exactText=true saved=true stopRecovery=true")
    }

    func testModelSwitchKeepsSeparateConversationsAndRestoresTheExistingREAMChat() async throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: folder) }
        let chats = folder.appendingPathComponent("Chats"), root = folder.appendingPathComponent("Models")
        try FileManager.default.createDirectory(at: chats, withIntermediateDirectories: true)
        let ream = [REAMMessage(role: .user, text: "Saved REAM question"), REAMMessage(role: .assistant, text: "Saved REAM answer")]
        let ornith = [REAMMessage(role: .user, text: "Saved Ornith question"), REAMMessage(role: .assistant, text: "Saved Ornith answer")]
        let oldFile = chats.appendingPathComponent("chat.json"), newFile = chats.appendingPathComponent("ornith-chat.json")
        try JSONEncoder().encode(ream).write(to: oldFile)
        try JSONEncoder().encode(ornith).write(to: newFile)
        let preference = UserDefaults.standard.string(forKey: "FlashLocal.selectedChatModel")
        defer {
            if let preference { UserDefaults.standard.set(preference, forKey: "FlashLocal.selectedChatModel") }
            else { UserDefaults.standard.removeObject(forKey: "FlashLocal.selectedChatModel") }
        }
        let model = REAMAppModel(root: root, chatFile: newFile, modelSelectionDirectory: chats)
        model.prepare(); await model.waitUntilIdle()
        XCTAssertEqual(model.selectedModel, .ornith); XCTAssertEqual(model.contextTokens, 4096)
        XCTAssertEqual(model.messages.map(\.text), ornith.map(\.text))
        model.selectModel(.ream); await model.waitUntilIdle()
        XCTAssertEqual(model.selectedModel, .ream); XCTAssertEqual(model.contextTokens, 1024)
        XCTAssertEqual(model.messages.map(\.text), ream.map(\.text))
        XCTAssertEqual(model.source.manifest, .primaryREAM)
        model.selectModel(.ornith); await model.waitUntilIdle()
        XCTAssertEqual(model.messages.map(\.text), ornith.map(\.text))
        XCTAssertEqual(try JSONDecoder().decode([REAMMessage].self, from: Data(contentsOf: oldFile)).map(\.id), ream.map(\.id))
        XCTAssertEqual(try JSONDecoder().decode([REAMMessage].self, from: Data(contentsOf: newFile)).map(\.id), ornith.map(\.id))
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

    func testClearDoesNotHideChatPersistenceFailures() async throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let model = REAMAppModel(root: folder.appendingPathComponent("Models"), chatFile: folder)
        model.prepare(); await model.waitUntilIdle()
        model.clearMessages(); await model.waitUntilIdle()
        XCTAssertNotNil(model.error, "Writing JSON onto a directory must leave a visible persistence error")
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
        model.prepare(); model.download(); model.verify(); model.draft = "Try"; model.send(); model.checkNumerics()
        XCTAssertTrue(model.isPreview); XCTAssertFalse(model.canChat); XCTAssertFalse(model.isBusy)
        XCTAssertEqual(model.messages.map(\.text), initial)
        XCTAssertNil(model.summary)
    }

    func testRestoresAndExplicitlyClearsOnlyItsOwnChatFile() async throws {
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
        await model.waitUntilIdle()
        XCTAssertTrue(model.messages.isEmpty)
        XCTAssertTrue(try JSONDecoder().decode([REAMMessage].self, from: Data(contentsOf: file)).isEmpty)
    }
}
