import Foundation

/// Actual executor observations. Expert reads exclude embedding mmap faults
/// and are not physical NAND traffic. No trained-quality or context claim.
public struct FlintGenerationReport: Codable, Sendable {
    public var schemaVersion = 1
    public var fullModelNumericsVerified = false
    public let modelSHA256, nativeRevision: String
    public let contextTokens: Int
    public let evaluationBatch, physicalBatch, cpuThreads: Int
    public let processMemoryBudgetBytes: UInt64
    public let platform, operatingSystem, sourceRevision: String
    public let appBuild: String?
    public let timestamp: Date
    public let promptTokenIDs, generatedTokenIDs: [Int32]
    public let reusedPromptTokens: Int
    public let prefillSeconds, decodeForwardSeconds, elapsedSeconds: Double
    public let timeToFirstTokenSeconds: Double?
    public let maximumObservedFootprintBytes, explicitExpertBytesRead: UInt64
    public let maximumThermalState: Int
    public let finishReason: String
    public var evaluatedPromptTokens: Int { promptTokenIDs.count - reusedPromptTokens }
    public var decodeTokensPerSecond: Double? {
        decodeForwardSeconds > 0 ? Double(generatedTokenIDs.count) / decodeForwardSeconds : nil
    }
}
