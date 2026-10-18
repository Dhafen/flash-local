import Foundation
import ModelAdapters

/// Independent precision investigation, separate from the original pinned
/// CPU comparison. A sampled projection or an observed-only success cannot
/// admit timing: the actual callback-free paths must pass the same thresholds.
public struct LESArithmeticCheck: Codable, Sendable {
    public let referenceArithmetic: String
    public let referenceRevision: String
    public let forwardCount: Int
    public let referenceMetrics: GGUFDequantizedReferenceMetrics
    public let ordinaryCPUComparison: LESLogitComparison
    public let noLookahead: LESLogitComparison
    public let lookahead: LESLogitComparison
    public let batchedPrefill: LESLogitComparison
    public let callbackFreeNoLookahead: LESLogitComparison
    public let callbackFreeLookahead: LESLogitComparison
    public let callbackFreeBatchedPrefill: LESLogitComparison
    public let callbackFreePageCache: LESLogitComparison
    public let pageCachePrefillTokens: Int
    public var diagnostics: LESNumericalDiagnostics? = nil
    public var diagnosticsUnavailable: Bool? = nil
    public static let arithmetic = "dequantized-weight-f32-double-v1"

    private var serialRoutingPassed: Bool {
        guard (4...67).contains(forwardCount) else { return false }
        return noLookahead.routingObserved == true && lookahead.routingObserved == true &&
            noLookahead.canonicalRoutingDigest != nil && noLookahead.routingDigest != nil &&
            noLookahead.routingDecisions == UInt64(forwardCount * 40 * 8) &&
            noLookahead.routingDecisions == referenceMetrics.routingDecisions &&
            noLookahead.canonicalRoutingDigest == referenceMetrics.canonicalRoutingDigest &&
            noLookahead.canonicalRoutingDigest == lookahead.canonicalRoutingDigest &&
            noLookahead.routingDigest == lookahead.routingDigest &&
            noLookahead.routingDecisions == lookahead.routingDecisions
    }
    public var batchedPrefillPassed: Bool {
        noLookahead.passed && lookahead.passed && serialRoutingPassed && batchedPrefill.passed &&
            batchedPrefill.routingObserved == true &&
            batchedPrefill.canonicalRoutingDigest == noLookahead.canonicalRoutingDigest &&
            batchedPrefill.routingDecisions == noLookahead.routingDecisions &&
            callbackFreeBatchedPrefill.passed && callbackFreeBatchedPrefill.callbackFree
    }
    public var prefillTokens: Int { batchedPrefillPassed ? 4 : 1 }
    public var passed: Bool {
        referenceArithmetic == Self.arithmetic &&
            referenceRevision == Qwen4ExpCPUReference.sourceRevision &&
            (4...67).contains(forwardCount) &&
            referenceMetrics.denseCalls >= UInt64(forwardCount * 40) &&
            referenceMetrics.expertCalls >= UInt64(forwardCount * 40 * 2) &&
            referenceMetrics.rows > 0 && (1...1_048_576).contains(referenceMetrics.maximumScratchBytes) &&
            noLookahead.passed && lookahead.passed && serialRoutingPassed &&
            callbackFreeNoLookahead.passed && callbackFreeNoLookahead.callbackFree &&
            callbackFreeLookahead.passed && callbackFreeLookahead.callbackFree &&
            callbackFreePageCache.passed && callbackFreePageCache.callbackFree &&
            pageCachePrefillTokens == prefillTokens
    }
}
