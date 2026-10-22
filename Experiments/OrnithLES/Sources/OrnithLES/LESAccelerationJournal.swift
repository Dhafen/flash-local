import Foundation
import StorageCore

/// Bounded, local phase journal written BEFORE entering native code. A normal
/// Swift error is catchable; a native abort or OS termination is not. Recovery
/// reports an interrupted operation, without guessing the cause of termination.
public final class LESAccelerationJournal: @unchecked Sendable {
    // Every mutation and file access is serialized by this lock. No UI state,
    // model data, prompts, tokens or credentials are retained by this object.
    private let lock = NSLock()
    public let url: URL
    public struct Event: Codable, Sendable {
        public let phase: String
        public let timestamp: Date
        public let footprintBytes: UInt64?
        public let availableBytes: UInt64?
    }
    public struct Snapshot: Codable, Sendable {
        public let schemaVersion: Int
        public let id: UUID
        public let sourceRevision, appBuild: String
        public let candidate: LESAccelerationBenchmark.Candidate
        public let contextTokens: Int
        public let started: Date
        public var outcome: String
        public var events: [Event]
        public var lastPhase: String? { events.last?.phase }
        public let terminationCause = "unknown; requires OS crash or jetsam report"
    }
    private var snapshot: Snapshot
    public init(directory: URL, sourceRevision: String, appBuild: String,
                candidate: LESAccelerationBenchmark.Candidate, contextTokens: Int) throws {
        guard directory.isFileURL, sourceRevision.utf8.count <= 128, appBuild.utf8.count <= 32,
              [8192, 16384, 32768, 65536].contains(contextTokens) else {
            throw CocoaError(.fileWriteInvalidFileName)
        }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        url = directory.appendingPathComponent("ornith-acceleration-journal.json")
        snapshot = .init(schemaVersion: 1, id: UUID(), sourceRevision: sourceRevision, appBuild: appBuild,
            candidate: candidate, contextTokens: contextTokens, started: Date(), outcome: "running", events: [])
        try record("preparing")
    }
    public func record(_ phase: String) throws {
        guard !phase.isEmpty, phase.utf8.count <= 96,
              phase.unicodeScalars.allSatisfy({ CharacterSet.alphanumerics.contains($0) || ".-_".unicodeScalars.contains($0) }) else {
            throw CocoaError(.fileWriteInvalidFileName)
        }
        lock.lock(); defer { lock.unlock() }
        let memory = ProcessFootprint.current()
        snapshot.events.append(.init(phase: phase, timestamp: Date(),
            footprintBytes: memory?.footprintBytes, availableBytes: memory?.availableBytes))
        snapshot.events = Array(snapshot.events.suffix(24))
        try Self.write(snapshot, to: url)
    }
    public func finish(failed: Bool) throws {
        lock.lock(); defer { lock.unlock() }
        snapshot.outcome = failed ? "failed" : "completed"
        try Self.write(snapshot, to: url)
    }
    private static func write(_ snapshot: Snapshot, to url: URL) throws {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(snapshot)
        guard data.count <= 65536 else { throw CocoaError(.fileWriteOutOfSpace) }
        try data.write(to: url, options: .atomic)
    }
    /// Must run before starting a new operation, never while a writer is live.
    public static func recoverInterrupted(directory: URL) throws -> Snapshot? {
        let url = directory.appendingPathComponent("ornith-acceleration-journal.json")
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let size = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? Int.max
        guard size <= 65536 else { throw CocoaError(.fileReadCorruptFile) }
        let decoder = JSONDecoder(); decoder.dateDecodingStrategy = .iso8601
        var value = try decoder.decode(Snapshot.self, from: Data(contentsOf: url))
        guard value.schemaVersion == 1, value.events.count <= 24,
              value.sourceRevision.utf8.count <= 128, value.appBuild.utf8.count <= 32,
              value.events.allSatisfy({ $0.phase.utf8.count <= 96 }) else { throw CocoaError(.fileReadCorruptFile) }
        guard value.outcome == "running" else { return nil }
        value.outcome = "interrupted"
        try write(value, to: url)
        return value
    }
    public func recordNativePhase(_ phase: Int32) -> Bool {
        let names: [Int32: String] = [1: "target.weights", 2: "target.context", 10: "draft.weights",
            11: "draft.context", 12: "draft.features", 13: "draft.fusion", 14: "draft.kv-injection",
            15: "draft.proposal", 16: "target.verification", 17: "target.rollback", 19: "target.forward"]
        guard let name = names[phase] else { return false }
        do { try record(name); return true } catch { return false }
    }
}
