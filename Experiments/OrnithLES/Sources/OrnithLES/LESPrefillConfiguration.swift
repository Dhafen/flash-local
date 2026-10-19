import Foundation
import ModelAdapters

/// Load-time scheduling only. Context, routing and weights are unchanged.
public struct LESPrefillConfiguration: Codable, Equatable, Sendable {
    public var evaluationTokens: Int = 4
    /// Zero uses the existing in-process 1/4-token numerical admission.
    public var physicalTokens: Int = 0
    public init() {}
    public func validate() throws {
        guard [1, 4, 8, 16, 32].contains(evaluationTokens),
              [0, 1, 4, 8, 16].contains(physicalTokens),
              physicalTokens == 0 || physicalTokens <= evaluationTokens else {
            throw GGUFError.limit("Välj evaluation 1/4/8/16/32 och fysisk batch Auto/1/4/8/16, högst evaluation.")
        }
    }
    public func physicalTokens(checkedDefault: Int) throws -> Int {
        try validate()
        guard checkedDefault == 1 || checkedDefault == 4 else { throw GGUFError.invalid("Missing baseline batch admission") }
        return physicalTokens == 0 ? min(evaluationTokens, checkedDefault) : physicalTokens
    }
}

/// A short independent, original-weight control for exactly one requested
/// physical/logical geometry. Never accepted as general model validation.
public struct LESPrefillCheck: Codable, Sendable {
    public let modelSHA256, nativeRevision, referenceRevision: String
    public let configuration: LESPrefillConfiguration
    public let contextTokens, promptTokens, forwardCount: Int
    public let referenceMetrics: GGUFDequantizedReferenceMetrics
    public let observed, callbackFree: LESLogitComparison
    public var passed: Bool {
        guard (try? configuration.validate()) != nil,
              [1, 4, 8, 16].contains(configuration.physicalTokens),
              contextTokens == 4096, promptTokens >= configuration.physicalTokens,
              (16...64).contains(promptTokens), forwardCount == promptTokens + 3,
              referenceRevision == Qwen4ExpCPUReference.sourceRevision,
              referenceMetrics.denseCalls >= UInt64(forwardCount * 40),
              referenceMetrics.expertCalls >= UInt64(forwardCount * 40 * 2),
              referenceMetrics.rows > 0, (1...1_048_576).contains(referenceMetrics.maximumScratchBytes),
              referenceMetrics.routingDecisions == UInt64(forwardCount * 40 * 8),
              referenceMetrics.canonicalExpertSetDigest != nil else { return false }
        return observed.passed && observed.routingObserved == true &&
            observed.routingDecisions == referenceMetrics.routingDecisions &&
            observed.canonicalExpertSetDigest == referenceMetrics.canonicalExpertSetDigest &&
            observed.configuredPhysicalBatchTokens == configuration.physicalTokens &&
            observed.maximumPhysicalBatchTokens == configuration.physicalTokens &&
            callbackFree.passed && callbackFree.callbackFree &&
            callbackFree.configuredPhysicalBatchTokens == configuration.physicalTokens &&
            callbackFree.maximumPhysicalBatchTokens == configuration.physicalTokens
    }
}
