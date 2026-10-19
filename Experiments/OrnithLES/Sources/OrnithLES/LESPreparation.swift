import Darwin
import Foundation
import FlashOrnithLES
import GGUFCore
import MinirunKit
import StorageCore

/// Prepared dense weights are a disposable derivative, never the installed
/// GGUF. Revision or integrity failures rebuild from the verified descriptor;
/// they must not permanently prevent an app update from starting the engine.
public enum LESPreparation {
    private static let lock = NSLock()
    private struct Seal: Codable {
        let sourceSHA256: String
        let sourceBytes: UInt64
        let denseSHA256: String
        let manifestSHA256: String
        let runtimeRevision: String
    }
    private static func digest(_ url: URL, control: LESControl) throws -> String {
        try FileDigestComputer.digest(ofFileAt: url, algorithm: .sha256,
            cancellationRequested: { control.shouldAbort() }).hex
    }
    private static func regularFile(_ url: URL, maximumBytes: UInt64) throws -> Bool {
        let value = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey])
        return value.isRegularFile == true && value.isSymbolicLink != true &&
            value.fileSize.map({ $0 > 0 && UInt64($0) <= maximumBytes }) == true
    }
    private static func valid(_ directory: URL, artifact: VerifiedGGUFArtifact, control: LESControl) throws -> Bool {
        guard !control.shouldAbort() else { throw CancellationError() }
        do {
            let sealURL = directory.appendingPathComponent("seal.json")
            let dense = directory.appendingPathComponent("dense.gguf")
            let manifest = directory.appendingPathComponent("manifest-template.json")
            guard try regularFile(sealURL, maximumBytes: 16 << 10),
                  try regularFile(dense, maximumBytes: artifact.manifest.fileByteCount),
                  try regularFile(manifest, maximumBytes: 16 << 20) else { return false }
            let seal = try JSONDecoder().decode(Seal.self, from: Data(contentsOf: sealURL))
            guard seal.sourceSHA256 == artifact.manifest.sha256,
                  seal.sourceBytes == artifact.manifest.fileByteCount,
                  seal.runtimeRevision == LESContext.revision else { return false }
            return try digest(dense, control: control) == seal.denseSHA256 &&
                digest(manifest, control: control) == seal.manifestSHA256
        } catch {
            // Cancellation/memory pressure is not evidence of bad cache data.
            guard !control.shouldAbort() else { throw CancellationError() }
            return false
        }
    }
    public static func prepare(artifact: VerifiedGGUFArtifact, privateRoot: URL, control: LESControl,
                               progress: @Sendable (String) -> Void = { _ in },
                               onAdmission: (LESMemoryAdmission) -> Void = { _ in }) throws -> URL {
        lock.lock(); defer { lock.unlock() }
        try artifact.assertUnchanged()
        guard privateRoot.isFileURL else { throw GGUFError.invalid("LES preparation requires a private file directory") }
        guard !control.shouldAbort() else { throw CancellationError() }
        guard LESContext.runtimeMemory.liveContexts == 0 else {
            throw GGUFError.invalid("Close the active LES context before preparing its files")
        }
        let hash = artifact.manifest.sha256
        guard hash.utf8.count == 64, hash.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else {
            throw GGUFError.invalid("Invalid verified model digest")
        }
        let fm = FileManager.default
        try fm.createDirectory(at: privateRoot, withIntermediateDirectories: true)
        var root = URL(fileURLWithPath: privateRoot.resolvingSymlinksInPath().standardizedFileURL.path, isDirectory: true)
        var excluded = URLResourceValues(); excluded.isExcludedFromBackup = true
        try root.setResourceValues(excluded)
        let final = root.appendingPathComponent("gguf-" + hash, isDirectory: true)
        // Fixed per-model transaction names bound interrupted writes to one
        // staging copy. They cannot collide with the original model or reports.
        let stage = root.appendingPathComponent("stage-" + hash, isDirectory: true)
        let previous = root.appendingPathComponent("previous-" + hash, isDirectory: true)
        func exists(_ url: URL) -> Bool { fm.fileExists(atPath: url.path) }
        func requireOwnedDirectory(_ url: URL) throws {
            guard [final, stage, previous].contains(url), url.deletingLastPathComponent().pathComponents == root.pathComponents else {
                throw GGUFError.invalid("Preparation transaction escaped its private directory")
            }
            let value = try url.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
            guard value.isDirectory == true, value.isSymbolicLink != true,
                  url.resolvingSymlinksInPath().deletingLastPathComponent().pathComponents == root.pathComponents else {
                throw GGUFError.invalid("Preparation cache must be a private nonsymlink directory")
            }
        }
        func removeOwned(_ url: URL) throws {
            if exists(url) { try requireOwnedDirectory(url); try fm.removeItem(at: url) }
        }
        for url in [final, stage, previous] where exists(url) { try requireOwnedDirectory(url) }
        if !exists(final), exists(previous) { try fm.moveItem(at: previous, to: final) }
        if exists(final) {
            progress("Kontrollerar snabbmotorns förberedda filer…")
            if try valid(final, artifact: artifact, control: control) {
                try artifact.assertUnchanged()
                try? removeOwned(stage); try? removeOwned(previous)
                return final
            }
            progress("Bygger om snabbmotorns förberedda filer från din lokala GGUF…")
        }
        func publish() throws -> URL {
            try artifact.assertUnchanged()
            guard !control.shouldAbort() else { throw CancellationError() }
            try removeOwned(previous)
            if exists(final) { try fm.moveItem(at: final, to: previous) }
            do { try fm.moveItem(at: stage, to: final) }
            catch {
                if !exists(final), exists(previous) { try? fm.moveItem(at: previous, to: final) }
                throw error
            }
            // A killed process can leave this backup; the next valid reuse
            // removes it. Publication success must not be reversed by cleanup.
            try? removeOwned(previous)
            return final
        }
        if exists(stage) {
            if try valid(stage, artifact: artifact, control: control) { return try publish() }
            try removeOwned(stage)
        }
        let plan = try LESBenchmark.plan(artifact, context: 32, depth: 4,
            phase: "preparation", onAdmission: onAdmission)
        guard !control.shouldAbort() else { throw CancellationError() }
        let available = try root.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
            .volumeAvailableCapacityForImportantUsage
        guard let available, available > 0, UInt64(available) > plan.denseBytes + (64 << 20) else {
            throw GGUFError.limit("LES preparation needs space for a new dense cache; the original GGUF and previous cache are retained")
        }
        try fm.createDirectory(at: stage, withIntermediateDirectories: false)
        defer { try? removeOwned(stage) }
        let fd = try artifact.duplicateForReading(); defer { Darwin.close(fd) }
        let status = flo_prepare(fd, stage.path, { user in
            guard let user else { return true }
            return Unmanaged<LESControl>.fromOpaque(user).takeUnretainedValue().shouldAbort()
        }, Unmanaged.passUnretained(control).toOpaque())
        guard status == 0 else { throw LESContext.failure() }
        let seal = Seal(sourceSHA256: hash, sourceBytes: artifact.manifest.fileByteCount,
            denseSHA256: try digest(stage.appendingPathComponent("dense.gguf"), control: control),
            manifestSHA256: try digest(stage.appendingPathComponent("manifest-template.json"), control: control),
            runtimeRevision: LESContext.revision)
        try JSONEncoder().encode(seal).write(to: stage.appendingPathComponent("seal.json"), options: .atomic)
        guard try valid(stage, artifact: artifact, control: control) else {
            throw GGUFError.invalid("Rebuilt LES prepared files failed integrity verification")
        }
        return try publish()
    }
}
