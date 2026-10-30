import Foundation
import XCTest
@testable import FlintEngine
import MinirunKit

final class FlintStreamTests: XCTestCase {
    func testNumericalAdmissionRejectsWrongTop1NonFiniteOrDistantLogits() throws {
        let exact = try FlintNumericalCheck.compare(position: 3, reference: [1, 3, -2], candidate: [1, 3, -2])
        XCTAssertTrue(exact.passed)
        XCTAssertFalse(try FlintNumericalCheck.compare(position: 3, reference: [1, 3, -2], candidate: [4, 3, -2]).passed)
        XCTAssertFalse(try FlintNumericalCheck.compare(position: 3, reference: [1, 3, -2], candidate: [1, 3.2, -2]).passed)
        XCTAssertThrowsError(try FlintNumericalCheck.compare(position: 3, reference: [1, .nan], candidate: [1, 2]))
        XCTAssertThrowsError(try FlintNumericalCheck.compare(position: 3, reference: [1], candidate: [1, 2]))
    }
    func testNumericalAdmissionRequiresActualCompleteExpertSets() {
        let a = FlintRouteDecision(position: 3, layer: 48, experts: [1, 25, 127, 200, 250, 255])
        let reordered = FlintRouteDecision(position: 3, layer: 48, experts: a.experts.reversed())
        XCTAssertTrue(FlintNumericalCheck.sameRoutes([a], [reordered]))
        XCTAssertFalse(FlintNumericalCheck.sameRoutes([], []))
        XCTAssertFalse(FlintNumericalCheck.sameRoutes([a], [a, a]))
        XCTAssertFalse(FlintNumericalCheck.sameRoutes([a], [.init(position: 3, layer: 48, experts: [1, 25, 127, 200, 250, 254])]))
        XCTAssertFalse(FlintNumericalCheck.sameRoutes([a], [.init(position: 3, layer: 48, experts: [1, 25, 127, 200, 250, 256])]))
    }
    func testRouteDiagnosticsReportFirstDifferenceAndMissingRowsWithoutPromptData() throws {
        let reference: [FlintRouteDecision] = [
            .init(position: 1, layer: 2, experts: [1, 2, 3, 4, 5, 6]),
            .init(position: 0, layer: 3, experts: [6, 5, 4, 3, 2, 1]),
            .init(position: 0, layer: 2, experts: [1, 2, 3, 4, 5, 6]),
        ]
        let candidate: [FlintRouteDecision] = [
            .init(position: 1, layer: 2, experts: [1, 2, 3, 4, 5, 7]),
            .init(position: 0, layer: 2, experts: [6, 5, 4, 3, 2, 1]),
            .init(position: 1, layer: 3, experts: [1, 2, 3, 4, 5, 6]),
        ]
        let diagnostic = try XCTUnwrap(FlintNumericalCheck.diagnoseRoutes(reference, candidate))
        XCTAssertEqual(diagnostic.referenceCount, 3)
        XCTAssertEqual(diagnostic.candidateCount, 3)
        XCTAssertEqual(diagnostic.mismatchCount, 3)
        XCTAssertEqual(diagnostic.firstMismatch?.position, 0)
        XCTAssertEqual(diagnostic.firstMismatch?.layer, 3)
        XCTAssertEqual(diagnostic.firstMismatch?.referenceExperts, [1, 2, 3, 4, 5, 6])
        XCTAssertNil(diagnostic.firstMismatch?.candidateExperts)
        XCTAssertNil(FlintNumericalCheck.diagnoseRoutes([reference[0], reference[0]], candidate))
        let identical = try XCTUnwrap(FlintNumericalCheck.diagnoseRoutes(reference, reference.reversed()))
        XCTAssertEqual(identical.mismatchCount, 0)
        XCTAssertNil(identical.firstMismatch)
        let encoded = try JSONEncoder().encode(diagnostic)
        let text = try XCTUnwrap(String(data: encoded, encoding: .utf8))
        XCTAssertFalse(text.contains("prompt"))
        XCTAssertFalse(text.contains("tokenID"))
        let legacy = Data("{\"positions\":[],\"routedTokenLayers\":1,\"exactExpertSets\":false}".utf8)
        let decoded = try JSONDecoder().decode(FlintNumericalCheck.self, from: legacy)
        XCTAssertNil(decoded.routeDiagnostics)
        XCTAssertNil(decoded.matchedMetalControl)
        XCTAssertNil(decoded.q4F32Reference)
        XCTAssertFalse(decoded.passed)
    }
    func testMatchedControlNeverOverridesFailedAdmission() throws {
        let good = FlintNumericalCheck.Position(index: 0, maximumAbsoluteError: 0,
            rmsError: 0, cosineSimilarity: 1, sameTop1: true)
        let control = FlintNumericalCheck.MatchedMetalControl(positions: [good, good, good, good],
            exactExpertSets: true, routeDiagnostics: nil)
        let failed = FlintNumericalCheck(positions: [], routedTokenLayers: 1, exactExpertSets: false,
            routeDiagnostics: nil, matchedMetalControl: control)
        XCTAssertFalse(failed.passed)
        let decoded = try JSONDecoder().decode(FlintNumericalCheck.self, from: JSONEncoder().encode(failed))
        XCTAssertEqual(decoded.matchedMetalControl?.positions.count, 4)
        XCTAssertFalse(decoded.passed)
    }
    func testQ4F32ReferenceRequiresFourRealPositionsAndCompleteMatchingRoutes() throws {
        func positions(_ indices: [Int], top1: Bool = true) -> [FlintNumericalCheck.Position] {
            indices.map { .init(index: $0, maximumAbsoluteError: 0, rmsError: 0, cosineSimilarity: 1, sameTop1: top1) }
        }
        let diagnostics = FlintNumericalCheck.RouteDiagnostics(referenceCount: 556, candidateCount: 556,
            mismatchCount: 0, firstMismatch: nil)
        func admission(_ rows: [FlintNumericalCheck.Position], _ count: Int = 556,
                       _ exact: Bool = true, _ routes: FlintNumericalCheck.RouteDiagnostics? = nil,
                       serving: [FlintNumericalCheck.Position]? = nil) -> FlintNumericalCheck {
            let alternative = FlintNumericalCheck.Q4F32Reference(positions: rows, routedTokenLayers: count,
                exactExpertSets: exact, routeDiagnostics: routes, servingPositions: serving ?? rows)
            return FlintNumericalCheck(positions: [], routedTokenLayers: 0, exactExpertSets: false,
                routeDiagnostics: nil, q4F32Reference: alternative)
        }
        let valid = admission(positions([0, 3, 6, 11]), 556, true, diagnostics)
        XCTAssertTrue(valid.passed)
        XCTAssertTrue(try JSONDecoder().decode(FlintNumericalCheck.self, from: JSONEncoder().encode(valid)).passed)
        XCTAssertFalse(admission(positions([0, 3, 6, 11]), 556, true, nil).passed)
        XCTAssertFalse(admission(positions([0, 3, 6, 11]), 555, true, diagnostics).passed)
        XCTAssertFalse(admission(positions([0, 3, 6, 11]), 556, false, diagnostics).passed)
        XCTAssertFalse(admission(positions([0, 3, 6, 11], top1: false), 556, true, diagnostics).passed)
        XCTAssertFalse(admission(positions([0, 3, 3, 11]), 556, true, diagnostics).passed)
        XCTAssertFalse(admission(positions([0, 3, 6, 11]), 556, true, diagnostics,
            serving: positions([0, 3, 6, 11], top1: false)).passed)
        XCTAssertFalse(admission(positions([0, 3, 6, 11]), 556, true, diagnostics,
            serving: positions([0, 3, 6])).passed)
        let incomplete = FlintNumericalCheck.Q4F32Reference(positions: positions([0, 3, 6, 11]),
            routedTokenLayers: 556, exactExpertSets: true, routeDiagnostics: diagnostics)
        XCTAssertTrue(incomplete.referencePassed)
        XCTAssertFalse(incomplete.passed)
        XCTAssertFalse(try JSONDecoder().decode(FlintNumericalCheck.Q4F32Reference.self,
            from: JSONEncoder().encode(incomplete)).passed)
        let mismatch = FlintNumericalCheck.RouteDiagnostics(referenceCount: 556, candidateCount: 556,
            mismatchCount: 1, firstMismatch: .init(position: 0, layer: 6,
                referenceExperts: [1, 2, 3, 4, 5, 6], candidateExperts: [1, 2, 3, 4, 5, 7]))
        XCTAssertFalse(admission(positions([0, 3, 6, 11]), 556, true, mismatch).passed)
    }
    func testPrecisionOverridesAreDiagnosticOnly() throws {
        var config = FlintConfiguration()
        config.diagnosticF32KV = true
        XCTAssertThrowsError(try config.validate())
        config.diagnosticRouting = true
        XCTAssertNoThrow(try config.validate())
        config.backend = .cpuReference
        XCTAssertThrowsError(try config.validate())
        config.diagnosticF32KV = false
        config.diagnosticDisableFlash = true
        XCTAssertThrowsError(try config.validate())
        config.backend = .metal
        XCTAssertNoThrow(try config.validate())
        config.diagnosticQ4F32Reference = true
        XCTAssertThrowsError(try config.validate())
        config.backend = .cpuReference; config.diagnosticDisableFlash = false
        XCTAssertNoThrow(try config.validate())
        config.diagnosticRouting = false
        XCTAssertThrowsError(try config.validate())
    }
    func testReasoningAndUTF8AreIdenticalForEveryBytePartition() throws {
        let text = "<think>Räkna 👋: 50+5.</think>Svaret är 55.\n"
        let bytes = Array(text.utf8)
        for size in 1...bytes.count {
            var stream = FlintTextStream(), answer = "", reasoning = ""
            for start in stride(from: 0, to: bytes.count, by: size) {
                let delta = try stream.append(Array(bytes[start..<min(start + size, bytes.count)]))
                answer += delta.content; reasoning += delta.reasoning
            }
            let end = try stream.finish(); answer += end.content; reasoning += end.reasoning
            XCTAssertEqual(answer, "Svaret är 55.\n"); XCTAssertEqual(reasoning, "Räkna 👋: 50+5.")
        }
    }
    func testIncompleteAndInvalidUTF8NeverCorruptToolArguments() throws {
        for bytes: [UInt8] in [[0x80], [0xc0, 0x80], [0xed, 0xa0, 0x80], [0xf4, 0x90, 0x80, 0x80], [0xe2, 0x28]] {
            var stream = FlintTextStream()
            XCTAssertThrowsError(try stream.append(bytes))
        }
        var stream = FlintTextStream()
        XCTAssertEqual(try stream.append([0xf0, 0x9f]).content, "")
        XCTAssertThrowsError(try stream.finish())
    }
    func testTruncatedDelimiterAndUnfinishedReasoningAreNotFabricated() throws {
        var text = FlintTextStream()
        let delta = try text.append(Array("code: <thi".utf8))
        XCTAssertEqual(delta.content, "code: ")
        XCTAssertEqual(try text.finish().content, "<thi")
        var thinking = FlintTextStream()
        XCTAssertEqual(try thinking.append(Array("<think>still working".utf8)).reasoning, "still working")
        XCTAssertTrue(thinking.isThinking); XCTAssertEqual(try thinking.finish().content, "")
    }
    func testWirePromptReplaysReasoningAndToolHistoryWithoutQwenTags() throws {
        let caps = LocalModelCapabilities(family: .flint)
        let request = try LocalChatRequest(data: JSONSerialization.data(withJSONObject: ["model": caps.modelID,
            "messages": [["role": "user", "content": "50+5"]]]), capabilities: caps)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: FlintWirePrompt.encode(request)) as? [String: Any])
        XCTAssertEqual((json["messages"] as? [[String: String]])?.first?["content"], "50+5")
        XCTAssertEqual(json["add_generation_prompt"] as? Bool, true)
    }
    func testSlidingWindowPrefixReuseRejectsChangedOrShortenedHistory() {
        XCTAssertEqual(FlintPrefixPolicy.reusableCount(cached: [1, 2, 3], prompt: [1, 2, 3, 4]), 3)
        XCTAssertFalse(FlintPrefixPolicy.requiresReset(cached: [1, 2, 3], prompt: [1, 2, 3, 4]))
        XCTAssertTrue(FlintPrefixPolicy.requiresReset(cached: [1, 2, 3], prompt: [1, 2]))
        XCTAssertTrue(FlintPrefixPolicy.requiresReset(cached: [1, 2, 3], prompt: [1, 9, 3]))
    }
}
