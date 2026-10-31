import Foundation
import XCTest
@testable import FlintEngine
import MinirunKit

final class FlintStreamTests: XCTestCase {
    func testBuild251NumericalReportRemainsRejected() throws {
        // Numerical scalars and the first differing expert set from the real
        // build 251 diagnostic. No prompt, token IDs or model bytes are stored.
        let data = Data(#"""
        {"routedTokenLayers":556,"exactExpertSets":false,
         "positions":[
           {"index":0,"maximumAbsoluteError":1.6782722473144531,"rmsError":0.5795510330074217,"cosineSimilarity":0.999881317144799,"sameTop1":true},
           {"index":3,"maximumAbsoluteError":1.7575931549072266,"rmsError":0.43258388635475786,"cosineSimilarity":0.9999514611291594,"sameTop1":true},
           {"index":6,"maximumAbsoluteError":6.180698394775391,"rmsError":2.344590137277799,"cosineSimilarity":0.999769510204985,"sameTop1":false},
           {"index":11,"maximumAbsoluteError":3.409656524658203,"rmsError":1.200228399684716,"cosineSimilarity":0.9999459210680908,"sameTop1":true}],
         "routeDiagnostics":{"referenceCount":556,"candidateCount":556,"mismatchCount":128,
           "firstMismatch":{"position":0,"layer":6,"referenceExperts":[17,57,113,182,203,206],"candidateExperts":[17,30,57,113,182,203]}},
         "matchedMetalControl":{"exactExpertSets":false,"positions":[
           {"index":0,"maximumAbsoluteError":1.6803264617919922,"rmsError":0.5814009741508069,"cosineSimilarity":0.9998809951575803,"sameTop1":true},
           {"index":3,"maximumAbsoluteError":1.6944255828857422,"rmsError":0.4228427593457391,"cosineSimilarity":0.9999555303553055,"sameTop1":true},
           {"index":6,"maximumAbsoluteError":8.367809295654297,"rmsError":3.3707503307136055,"cosineSimilarity":0.9995613611955291,"sameTop1":false},
           {"index":11,"maximumAbsoluteError":3.4108505249023438,"rmsError":1.2011623351430916,"cosineSimilarity":0.9999459293800075,"sameTop1":true}],
           "routeDiagnostics":{"referenceCount":556,"candidateCount":556,"mismatchCount":130,
             "firstMismatch":{"position":0,"layer":6,"referenceExperts":[17,57,113,182,203,206],"candidateExperts":[17,30,57,113,182,203]}}},
         "q4F32Reference":{"routedTokenLayers":556,"exactExpertSets":false,"positions":[
           {"index":0,"maximumAbsoluteError":0.0058803558349609375,"rmsError":0.002154697755235025,"cosineSimilarity":0.999999998673728,"sameTop1":true},
           {"index":3,"maximumAbsoluteError":0.08395957946777344,"rmsError":0.014414634787969859,"cosineSimilarity":0.9999998886248521,"sameTop1":true},
           {"index":6,"maximumAbsoluteError":3.215679168701172,"rmsError":1.079167423025251,"cosineSimilarity":0.9998928087571591,"sameTop1":true},
           {"index":11,"maximumAbsoluteError":0.016693115234375,"rmsError":0.0035905684512366554,"cosineSimilarity":0.999999997724888,"sameTop1":true}],
           "routeDiagnostics":{"referenceCount":556,"candidateCount":556,"mismatchCount":2,
             "firstMismatch":{"position":6,"layer":35,"referenceExperts":[40,54,69,112,141,198],"candidateExperts":[40,54,69,102,112,141]}}}}
        """#.utf8)
        let report = try JSONDecoder().decode(FlintNumericalCheck.self, from: data)
        XCTAssertEqual(report.positions.map(\.index), [0, 3, 6, 11])
        XCTAssertEqual(report.positions[2].maximumAbsoluteError, 6.180698394775391)
        XCTAssertFalse(report.positions[2].sameTop1)
        XCTAssertEqual(report.routeDiagnostics?.mismatchCount, 128)
        XCTAssertEqual(report.matchedMetalControl?.routeDiagnostics?.mismatchCount, 130)
        XCTAssertEqual(report.q4F32Reference?.routeDiagnostics?.mismatchCount, 2)
        XCTAssertEqual(report.q4F32Reference?.routeDiagnostics?.firstMismatch?.layer, 35)
        XCTAssertFalse(try XCTUnwrap(report.q4F32Reference).referencePassed)
        XCTAssertNil(report.servingMode, "An observed mismatch must not admit chat or API inference")
        let reference = [FlintRouteDecision(position: 0, layer: 6, experts: [17,57,113,182,203,206])]
        let candidate = [FlintRouteDecision(position: 0, layer: 6, experts: [17,30,57,113,182,203])]
        let first = try XCTUnwrap(FlintNumericalCheck.diagnoseRoutes(reference, candidate))
        XCTAssertEqual(first.mismatchCount, 1)
        XCTAssertEqual(first.firstMismatch, report.routeDiagnostics?.firstMismatch)
        let exactCPU = [FlintRouteDecision(position: 6, layer: 35, experts: [40,54,69,112,141,198])]
        let normalMetal = [FlintRouteDecision(position: 6, layer: 35, experts: [40,54,69,102,112,141])]
        let later = try XCTUnwrap(FlintNumericalCheck.diagnoseRoutes(exactCPU, normalMetal))
        XCTAssertEqual(later.firstMismatch, report.q4F32Reference?.routeDiagnostics?.firstMismatch)
    }
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
        XCTAssertNil(decoded.metalFallbacks)
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
                exactExpertSets: exact, routeDiagnostics: routes, servingPositions: serving ?? rows,
                servingReferencePositions: rows)
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
    func testFallbackRequiresCompleteReferenceAndCallbackFreeServingProof() throws {
        let rows = [0, 3, 6, 11].map {
            FlintNumericalCheck.Position(index: $0, maximumAbsoluteError: 0, rmsError: 0, cosineSimilarity: 1, sameTop1: true)
        }
        let matching = FlintNumericalCheck.RouteDiagnostics(referenceCount: 556, candidateCount: 556,
            mismatchCount: 0, firstMismatch: nil)
        let complete = FlintNumericalCheck.Q4F32Reference(positions: rows, routedTokenLayers: 556,
            exactExpertSets: true, routeDiagnostics: matching, servingPositions: rows, servingReferencePositions: rows)
        let incomplete = FlintNumericalCheck.Q4F32Reference(positions: rows, routedTokenLayers: 556,
            exactExpertSets: true, routeDiagnostics: matching)
        let missingDirectCPUComparison = FlintNumericalCheck.Q4F32Reference(positions: rows, routedTokenLayers: 556,
            exactExpertSets: true, routeDiagnostics: matching, servingPositions: rows)
        var differentReference = rows
        differentReference[2] = .init(index: 6, maximumAbsoluteError: 0.11, rmsError: 0,
            cosineSimilarity: 1, sameTop1: true)
        let failedDirectCPUComparison = FlintNumericalCheck.Q4F32Reference(positions: rows, routedTokenLayers: 556,
            exactExpertSets: true, routeDiagnostics: matching, servingPositions: rows,
            servingReferencePositions: differentReference)
        let one = FlintNumericalCheck.MetalFallback(mode: .singleToken, outcome: .compared, comparison: complete)
        let unavailable = FlintNumericalCheck.MetalFallback(mode: .singleToken, outcome: .unavailable, comparison: nil)
        let precise = FlintNumericalCheck.MetalFallback(mode: .singleTokenF32KV, outcome: .compared, comparison: complete)
        func check(_ candidates: [FlintNumericalCheck.MetalFallback]) -> FlintNumericalCheck {
            .init(positions: [], routedTokenLayers: 0, exactExpertSets: false, routeDiagnostics: nil, metalFallbacks: candidates)
        }
        XCTAssertEqual(check([one]).servingMode, .singleToken)
        XCTAssertEqual(check([unavailable, precise]).servingMode, .singleTokenF32KV)
        XCTAssertEqual(try JSONDecoder().decode(FlintNumericalCheck.self, from: JSONEncoder().encode(check([unavailable, precise]))).servingMode, .singleTokenF32KV)
        XCTAssertFalse(check([.init(mode: .singleToken, outcome: .compared, comparison: incomplete)]).passed)
        XCTAssertFalse(check([.init(mode: .singleToken, outcome: .compared, comparison: missingDirectCPUComparison)]).passed)
        XCTAssertFalse(check([.init(mode: .singleToken, outcome: .compared, comparison: failedDirectCPUComparison)]).passed)
        XCTAssertFalse(check([.init(mode: .singleToken, outcome: .unavailable, comparison: complete)]).passed)
        XCTAssertFalse(check([precise]).passed, "A fallback cannot skip its prescribed predecessor")
        XCTAssertFalse(check([one, one]).passed, "Duplicate or unexpected modes fail closed")
        XCTAssertFalse(check([.init(mode: .standard, outcome: .compared, comparison: complete)]).passed)
        let differing = FlintNumericalCheck.RouteDiagnostics(referenceCount: 556, candidateCount: 556,
            mismatchCount: 2, firstMismatch: .init(position: 6, layer: 35,
                referenceExperts: [1, 2, 3, 4, 5, 6], candidateExperts: [1, 2, 3, 4, 5, 7]))
        let notEquivalent = FlintNumericalCheck.Q4F32Reference(positions: rows, routedTokenLayers: 556,
            exactExpertSets: true, routeDiagnostics: differing, servingPositions: rows, servingReferencePositions: rows)
        XCTAssertFalse(check([.init(mode: .singleToken, outcome: .compared, comparison: notEquivalent)]).passed)
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
    func testAdmittedServingModeUsesTheSameNumericalConfigurationAsItsTrace() throws {
        var requested = FlintConfiguration(); requested.contextTokens = 8192; requested.evaluationBatch = 128; requested.physicalBatch = 4
        let fastTrace = try FlintConfiguration.forMode(.singleToken, requested: requested, tracing: true)
        let fastServe = try FlintConfiguration.forMode(.singleToken, requested: requested, tracing: false)
        XCTAssertEqual(fastTrace.contextTokens, fastServe.contextTokens)
        XCTAssertEqual(fastTrace.evaluationBatch, fastServe.evaluationBatch)
        XCTAssertEqual(fastTrace.physicalBatch, 1); XCTAssertEqual(fastServe.physicalBatch, 1)
        XCTAssertFalse(fastTrace.diagnosticF32KV); XCTAssertFalse(fastServe.conservativeMetal)
        let preciseTrace = try FlintConfiguration.forMode(.singleTokenF32KV, requested: requested, tracing: true)
        let preciseServe = try FlintConfiguration.forMode(.singleTokenF32KV, requested: requested, tracing: false)
        XCTAssertTrue(preciseTrace.diagnosticRouting && preciseTrace.diagnosticF32KV && preciseTrace.diagnosticDisableFlash)
        XCTAssertTrue(!preciseServe.diagnosticRouting && preciseServe.conservativeMetal)
        XCTAssertEqual(preciseTrace.contextTokens, preciseServe.contextTokens)
        XCTAssertEqual(preciseTrace.evaluationBatch, preciseServe.evaluationBatch)
        XCTAssertEqual(preciseTrace.physicalBatch, preciseServe.physicalBatch)
        XCTAssertEqual(requested.physicalBatch, 4, "The requested configuration is not mutated")
        var invalid = preciseServe; invalid.physicalBatch = 2
        XCTAssertThrowsError(try invalid.validate())
        invalid = preciseServe; invalid.diagnosticRouting = true
        XCTAssertThrowsError(try invalid.validate())
        invalid = preciseServe; invalid.backend = .cpuReference
        XCTAssertThrowsError(try invalid.validate())
        XCTAssertThrowsError(try FlintConfiguration.forMode(.standard, requested: preciseServe, tracing: false))
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
