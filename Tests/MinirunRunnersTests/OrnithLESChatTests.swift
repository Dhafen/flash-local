#if canImport(OrnithLES)
import Foundation
import XCTest
import OrnithLES
import ModelAdapters
@testable import MinirunRunners

final class OrnithLESChatTests: XCTestCase {
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
