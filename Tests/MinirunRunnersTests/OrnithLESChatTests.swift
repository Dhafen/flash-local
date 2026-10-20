#if canImport(OrnithLES)
import Foundation
import XCTest
import OrnithLES
import ModelAdapters
import MinirunKit
@testable import MinirunRunners

final class OrnithLESChatTests: XCTestCase {
    func testBatchBenchmarkRejectsDifferentPartialOrMissingTokenOutputs() {
        let tokens = Array(0..<129)
        XCTAssertTrue(OrnithPrefillBenchmark.Report.same128TokenOutputs([tokens, tokens, tokens, tokens]))
        XCTAssertFalse(OrnithPrefillBenchmark.Report.same128TokenOutputs([tokens, tokens, tokens]))
        var changed = tokens; changed[128] = 17
        XCTAssertFalse(OrnithPrefillBenchmark.Report.same128TokenOutputs([tokens, changed, tokens, tokens]))
        let short = Array(tokens.prefix(127))
        XCTAssertFalse(OrnithPrefillBenchmark.Report.same128TokenOutputs([short, short, short, short]))
        print("ORNITH_PREFILL_COMPARISON exactOutputs=true min128Tokens=true partialRejected=true syntheticOnly=true")
    }
    func testDefaultSamplerIsIdenticalToExistingGreedyAndDoesNotChangeLogits() throws {
        var sampler = LocalTokenSampler(seed: 42)
        let options = LocalGenerationOptions()
        for step in 0..<128 {
            let logits = (0..<257).map { Float((($0 * 17 + step * 29) % 113) - 56) / 17 }
            let unchanged = logits
            XCTAssertEqual(try sampler.select(logits, options: options), try Qwen4ExpGreedyToken.select(logits))
            XCTAssertEqual(logits, unchanged)
        }
        XCTAssertEqual(options.thinkingBudget, 512); XCTAssertEqual(options.maximumNewTokens, 2048); XCTAssertEqual(options.temperature, 0)
        print("ORNITH_CHAT_SAMPLING greedyIdentical=true logitsUnchanged=true thinkingBudgetDefault512=true fixtures=128 trainedInference=false")
    }
    func testAdmittedChatUsesOnlyValidatedBatchAndBindsModelAndRuntime() throws {
        let serial: [String: Any] = ["positions": 4, "maximumAbsoluteError": 0.001, "relativeL2Error": 0.0001,
            "sameGreedy": true, "routingDigest": 19, "routingDecisions": 8000, "canonicalRoutingDigest": 23,
            "canonicalExpertSetDigest": 41, "routingObserved": true]
        let plain: [String: Any] = ["positions": 4, "maximumAbsoluteError": 0.001, "relativeL2Error": 0.0001,
            "sameGreedy": true, "routingObserved": false]
        for batch in [1, 4] {
            var batched = serial
            if batch == 1 { batched["maximumAbsoluteError"] = 2.0 }
            let proof: [String: Any] = ["referenceArithmetic": LESArithmeticCheck.arithmetic,
                "referenceRevision": Qwen4ExpCPUReference.sourceRevision, "forwardCount": 25,
                "referenceMetrics": ["denseCalls": 10000, "expertCalls": 3000, "rows": 1000000,
                    "maximumScratchBytes": 786432, "canonicalRoutingDigest": 23, "routingDecisions": 8000,
                    "canonicalExpertSetDigest": 41],
                "ordinaryCPUComparison": plain, "noLookahead": serial, "lookahead": serial, "batchedPrefill": batched,
                "callbackFreeNoLookahead": plain, "callbackFreeLookahead": plain,
                "callbackFreeBatchedPrefill": plain, "callbackFreePageCache": plain,
                "pageCachePrefillTokens": batch, "routingComparison": LESArithmeticCheck.expertSetRouting]
            let raw: [String: Any] = ["modelSHA256": "model", "referenceRevision": Qwen4ExpCPUReference.sourceRevision,
                "nativeRevision": "native", "noLookahead": serial, "lookahead": serial, "batchedPrefill": batched,
                "arithmeticCheck": proof]
            let check = try JSONDecoder().decode(LESNumericalCheck.self, from: JSONSerialization.data(withJSONObject: raw))
            XCTAssertTrue(check.nativeBenchmarkAdmitted)
            XCTAssertEqual(try OrnithLESChatGeneration.validatedPrefill(check, modelSHA256: "model", nativeRevision: "native"), batch)
            let selected = try OrnithLESChatGeneration.validatedBatch(.init(), proof: nil, baseline: check,
                modelSHA256: "model", nativeRevision: "native")
            XCTAssertEqual(selected.0, batch); XCTAssertEqual(selected.1, 4)
            var tuning = LESPrefillConfiguration(); tuning.evaluationTokens = 32; tuning.physicalTokens = 8
            XCTAssertThrowsError(try OrnithLESChatGeneration.validatedBatch(tuning, proof: nil, baseline: check,
                modelSHA256: "model", nativeRevision: "native"))
            var tunedObserved = serial, tunedFree = plain
            tunedObserved["configuredPhysicalBatchTokens"] = 8; tunedObserved["maximumPhysicalBatchTokens"] = 8
            tunedFree["configuredPhysicalBatchTokens"] = 8; tunedFree["maximumPhysicalBatchTokens"] = 8
            var tuned: [String: Any] = ["modelSHA256": "model", "nativeRevision": "native",
                "referenceRevision": Qwen4ExpCPUReference.sourceRevision,
                "configuration": ["evaluationTokens": 32, "physicalTokens": 8],
                "contextTokens": 4096, "promptTokens": 22, "forwardCount": 25,
                "referenceMetrics": proof["referenceMetrics"]!, "observed": tunedObserved, "callbackFree": tunedFree]
            func decodeProof() throws -> LESPrefillCheck {
                try JSONDecoder().decode(LESPrefillCheck.self, from: JSONSerialization.data(withJSONObject: tuned))
            }
            let admitted = try OrnithLESChatGeneration.validatedBatch(tuning, proof: decodeProof(), baseline: check,
                modelSHA256: "model", nativeRevision: "native")
            XCTAssertEqual(admitted.0, 32); XCTAssertEqual(admitted.1, 8)
            tuned["modelSHA256"] = "changed"
            XCTAssertThrowsError(try OrnithLESChatGeneration.validatedBatch(tuning, proof: decodeProof(), baseline: check,
                modelSHA256: "model", nativeRevision: "native"))
            tuned["modelSHA256"] = "model"; tuned["nativeRevision"] = "changed"
            XCTAssertThrowsError(try OrnithLESChatGeneration.validatedBatch(tuning, proof: decodeProof(), baseline: check,
                modelSHA256: "model", nativeRevision: "native"))
            tuned["nativeRevision"] = "native"; tuning.evaluationTokens = 16
            XCTAssertThrowsError(try OrnithLESChatGeneration.validatedBatch(tuning, proof: decodeProof(), baseline: check,
                modelSHA256: "model", nativeRevision: "native"))
            XCTAssertThrowsError(try OrnithLESChatGeneration.validatedPrefill(check, modelSHA256: "changed", nativeRevision: "native"))
            XCTAssertThrowsError(try OrnithLESChatGeneration.validatedPrefill(check, modelSHA256: "model", nativeRevision: "changed"))
        }
    }
    func testPrefixReuseRequiresCompleteExactTokenHistoryAndUsableLogits() {
        func reuse(_ previous: [Int], _ prompt: [Int], _ logits: Bool = true) -> Int {
            OrnithLESChatGeneration.reusablePrefix(previous: previous, prompt: prompt, hasLogits: logits)
        }
        XCTAssertEqual(reuse([1, 2, 3], [1, 2, 3, 4]), 3)
        XCTAssertEqual(reuse([1, 2, 3], [1, 2, 3]), 3)
        XCTAssertEqual(reuse([1, 2, 3], [1, 2, 3], false), 0)
        XCTAssertEqual(reuse([1, 2, 3], [1, 2, 4]), 0)
        XCTAssertEqual(reuse([1, 2, 3], [1, 2]), 0)
        XCTAssertEqual(reuse([], [1]), 0)
        // An interrupted state is discarded, never approximated by overlap.
        XCTAssertEqual(reuse([1, 9, 3], [1, 2, 3, 4]), 0)
    }
    func testChatRejectsUnadmittedAndMismatchedModelOrRuntimeChecks() throws {
        let raw: [String: Any] = ["modelSHA256": "model", "referenceRevision": "reference", "nativeRevision": "native",
            "noLookahead": ["positions": 4, "maximumAbsoluteError": 0, "relativeL2Error": 0, "sameGreedy": true],
            "lookahead": ["positions": 4, "maximumAbsoluteError": 0, "relativeL2Error": 0, "sameGreedy": true],
            "batchedPrefill": ["positions": 4, "maximumAbsoluteError": 0, "relativeL2Error": 0, "sameGreedy": true]]
        let check = try JSONDecoder().decode(LESNumericalCheck.self, from: JSONSerialization.data(withJSONObject: raw))
        for (model, native) in [("model", "native"), ("other", "native"), ("model", "other")] {
            XCTAssertThrowsError(try OrnithLESChatGeneration.validatedPrefill(check, modelSHA256: model, nativeRevision: native))
        }
        XCTAssertFalse(check.nativeBenchmarkAdmitted)
        print("ORNITH_CHAT_POLICY inProcessGate=true matchingModelAndRuntime=true exactPrefixOnly=true ordinaryEngineUnchanged=true syntheticOnly=true")
    }
}
#endif
