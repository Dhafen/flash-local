import CryptoKit
import Foundation
import SwiftletCore

public struct QPackInstallProgress: Sendable {
    public enum Phase: String, Sendable { case downloading, verifying, complete }
    public let phase: Phase
    public let completedSourceBytes: Int
    public let totalSourceBytes: Int
}

public struct VerifiedQPackInstall: Sendable {
    public let directory: URL
    public let sourceIdentity: String
    public let verifiedSourceBytes: Int
    // Only the verifier constructs this handle, after complete byte checks.
    init(directory: URL, sourceIdentity: String, verifiedSourceBytes: Int) {
        self.directory = directory
        self.sourceIdentity = sourceIdentity
        self.verifiedSourceBytes = verifiedSourceBytes
    }
}

/// Lossless placement into Swiftlet QPack v1. The trained checkpoint is never
/// written to disk: only final dense/expert files, small metadata and a journal.
/// Network windows and restart hashing are independently bounded to 8 MiB.
public actor QPackInstaller {
    struct Journal: Codable {
        let sourceIdentity: String
        var completed: [String: Int]
    }
    struct Receipt: Codable {
        let sourceIdentity: String
        let sourceSHA256: [String: String]
        let fileSHA256: [String: String]
        let verifiedSourceBytes: Int
    }
    struct Manifest: Codable {
        let magic = "QPACK"
        let version = 1
        let modelName: String
        let sourceCheckpoint: String
        let quantBits = 4
        let quantGroupSize = 64
        let files: [String: Int]
    }

    public let plan: QPackInstallPlan
    public let root: URL
    private let source: any QPackByteSource
    private let windowBytes: Int
    private let diskReserveBytes: UInt64
    private var active = false

    public init(plan: QPackInstallPlan, privateRoot: URL, source: any QPackByteSource,
                windowBytes: Int = 8 << 20, diskReserveBytes: UInt64 = 256 << 20) throws {
        guard windowBytes > 0, windowBytes <= 8 << 20 else {
            throw OrnithPerformanceError.invalidSource("installer window exceeds 8 MiB")
        }
        self.plan = plan
        root = privateRoot
        self.source = source
        self.windowBytes = windowBytes
        self.diskReserveBytes = diskReserveBytes
    }

    public var finalDirectory: URL {
        root.appendingPathComponent("ornith-\(plan.source.manifest.revision.prefix(12)).qpack", isDirectory: true)
    }

    public func install(onProgress: @Sendable (QPackInstallProgress) -> Void = { _ in }) async throws -> VerifiedQPackInstall {
        guard !active else { throw OrnithPerformanceError.io("installer already active") }
        active = true
        defer { active = false }
        try Task.checkCancellation()
        try QPackFiles.ensureDirectory(root)
        let fm = FileManager.default
        if fm.fileExists(atPath: finalDirectory.path) {
            return try verifyInstalled(at: finalDirectory, onProgress: onProgress)
        }
        let staging = root.appendingPathComponent(".ornith-\(plan.source.manifest.revision.prefix(12)).installing", isDirectory: true)
        try QPackFiles.ensureDirectory(staging)
        try QPackFiles.ensureDirectory(staging.appendingPathComponent("packed_experts", isDirectory: true))
        let identity = try plan.source.manifest.identity()
        let journalURL = staging.appendingPathComponent(".install-progress.json")
        var journal: Journal
        if fm.fileExists(atPath: journalURL.path) {
            journal = try JSONDecoder().decode(Journal.self, from: readSmall(journalURL, maximum: 1 << 20))
            guard journal.sourceIdentity == identity,
                  Set(journal.completed.keys).isSubset(of: Set(plan.source.manifest.shards.map(\.name))) else {
                throw OrnithPerformanceError.integrity("partial installation belongs to another source")
            }
        } else {
            // Refuse ambiguous nonempty output rather than trusting stale data.
            let entries = try fm.contentsOfDirectory(atPath: staging.path)
            guard entries.allSatisfy({ $0 == "packed_experts" }),
                  (try fm.contentsOfDirectory(atPath: staging.appendingPathComponent("packed_experts").path)).isEmpty else {
                throw OrnithPerformanceError.integrity("partial output has no source-bound journal")
            }
            journal = Journal(sourceIdentity: identity, completed: [:])
            try writeJSON(journal, to: journalURL)
        }
        try checkDiskSpace(staging: staging)
        let files = try QPackFiles(directory: staging, sizes: plan.fileBytes, create: true)
        try files.write(plan.densePrefix, range: 0..<plan.densePrefix.count, file: "model.safetensors", offset: 0)
        var auxiliarySizes: [String: Int] = [:]
        for auxiliary in plan.source.manifest.auxiliary {
            try Task.checkCancellation()
            let url = staging.appendingPathComponent(auxiliary.name)
            var data: Data?
            if let prior = try? readSmall(url, maximum: auxiliary.byteCount),
               prior.count == auxiliary.byteCount, digest(prior) == auxiliary.sha256 { data = prior }
            if data == nil {
                if auxiliary.name == "config.json" { data = plan.source.configData }
                else { data = try await fetch(name: auxiliary.name, range: nil, total: auxiliary.byteCount) }
            }
            guard let data, data.count == auxiliary.byteCount, digest(data) == auxiliary.sha256 else {
                throw OrnithPerformanceError.integrity("auxiliary hash mismatch: \(auxiliary.name)")
            }
            try data.write(to: url, options: [.atomic])
            auxiliarySizes[auxiliary.name] = data.count
        }

        let total = try plan.source.manifest.shards.reduce(0) { try checkedAdd($0, $1.byteCount) }
        var verifiedSourceBytes = 0
        for (index, shard) in plan.source.manifest.shards.enumerated() {
            let completed = journal.completed[shard.name] ?? 0
            guard completed >= 0, completed <= shard.byteCount else {
                throw OrnithPerformanceError.integrity("invalid resume position")
            }
            // Reconstruct the old prefix from final destinations. CryptoKit
            // SHA state is not serialized; no unchecked resume/hash shortcut.
            var hasher = SHA256()
            var cursor = 0
            while cursor < completed {
                try Task.checkCancellation()
                let end = min(try checkedAdd(cursor, windowBytes), completed)
                hasher.update(data: try reconstruct(shard: index, range: cursor..<end, files: files))
                cursor = end
            }
            while cursor < shard.byteCount {
                try Task.checkCancellation()
                let end = min(try checkedAdd(cursor, windowBytes), shard.byteCount)
                let data = try await fetch(name: shard.name, range: cursor..<end, total: shard.byteCount)
                guard data.count == end - cursor else { throw OrnithPerformanceError.integrity("short source window") }
                try checkHeaderBytes(data, shard: index, offset: cursor)
                var touched = Set<String>()
                for piece in try plan.pieces(shard: index, range: cursor..<end) {
                    let lo = piece.sourceRange.lowerBound - cursor
                    try files.write(data, range: lo..<(lo + piece.sourceRange.count), file: piece.file,
                                    offset: piece.destinationOffset)
                    touched.insert(piece.file)
                }
                // Checkpoint after writes, never before them. A lost journal
                // only causes a bounded range to be written again.
                try files.synchronize(touched)
                hasher.update(data: data)
                cursor = end
                journal.completed[shard.name] = cursor
                try writeJSON(journal, to: journalURL)
                onProgress(.init(phase: .downloading, completedSourceBytes: verifiedSourceBytes + cursor,
                                 totalSourceBytes: total))
            }
            let actual = hasher.finalize().map { String(format: "%02x", $0) }.joined()
            guard actual == shard.sha256 else {
                journal.completed[shard.name] = 0
                try writeJSON(journal, to: journalURL)
                throw OrnithPerformanceError.integrity("complete source shard SHA-256 mismatch: \(shard.name)")
            }
            verifiedSourceBytes = try checkedAdd(verifiedSourceBytes, shard.byteCount)
        }
        try files.synchronize(Set(plan.fileBytes.keys))
        onProgress(.init(phase: .verifying, completedSourceBytes: verifiedSourceBytes, totalSourceBytes: total))
        // Verify the converter, not just the arriving network stream. Read the
        // final scattered bytes back in original source order and require the
        // publisher's full-shard SHA-256 for every reconstructed shard.
        try verifyReconstructedSource(files: files)
        let layoutURL = staging.appendingPathComponent("packed_experts/layout.json")
        try plan.layoutData.write(to: layoutURL, options: [.atomic])
        var manifestSizes = plan.fileBytes.merging(auxiliarySizes, uniquingKeysWith: { _, new in new })
        manifestSizes["packed_experts/layout.json"] = plan.layoutData.count
        let manifest = Manifest(modelName: "qwen3_5_moe", sourceCheckpoint:
            "https://huggingface.co/\(plan.source.manifest.repository)/tree/\(plan.source.manifest.revision)", files: manifestSizes)
        let manifestURL = staging.appendingPathComponent("manifest.json")
        try writeJSON(manifest, to: manifestURL)
        _ = try Qpack.verify(containerDir: staging)
        onProgress(.init(phase: .verifying, completedSourceBytes: verifiedSourceBytes, totalSourceBytes: total))
        // Bind a local integrity receipt to all final files as actually read
        // back from disk, including tokenizer/template/layout and manifest.
        var hashes: [String: String] = [:]
        for name in manifestSizes.keys.sorted() + ["manifest.json"] {
            try Task.checkCancellation()
            hashes[name] = try fileHash(staging.appendingPathComponent(name))
        }
        let receipt = Receipt(sourceIdentity: identity,
            sourceSHA256: Dictionary(uniqueKeysWithValues: plan.source.manifest.shards.map { ($0.name, $0.sha256) }),
            fileSHA256: hashes, verifiedSourceBytes: verifiedSourceBytes)
        try writeJSON(receipt, to: staging.appendingPathComponent("flash-local-integrity.json"))
        // Same parent volume: this moves the directory, never copies weights.
        try fm.moveItem(at: staging, to: finalDirectory)
        onProgress(.init(phase: .complete, completedSourceBytes: verifiedSourceBytes, totalSourceBytes: total))
        return VerifiedQPackInstall(directory: finalDirectory, sourceIdentity: identity, verifiedSourceBytes: verifiedSourceBytes)
    }

    public func verifyInstalled(at directory: URL,
        onProgress: @Sendable (QPackInstallProgress) -> Void = { _ in }
    ) throws -> VerifiedQPackInstall {
        let identity = try plan.source.manifest.identity()
        let receipt = try JSONDecoder().decode(Receipt.self,
            from: readSmall(directory.appendingPathComponent("flash-local-integrity.json"), maximum: 1 << 20))
        let total = try plan.source.manifest.shards.reduce(0) { try checkedAdd($0, $1.byteCount) }
        var expected = Set(plan.fileBytes.keys)
        expected.formUnion(plan.source.manifest.auxiliary.map(\.name))
        expected.formUnion(["packed_experts/layout.json", "manifest.json"])
        guard receipt.sourceIdentity == identity, receipt.verifiedSourceBytes == total,
              Set(receipt.fileSHA256.keys) == expected,
              receipt.sourceSHA256 == Dictionary(uniqueKeysWithValues:
                plan.source.manifest.shards.map { ($0.name, $0.sha256) }) else {
            throw OrnithPerformanceError.integrity("receipt belongs to another source/layout")
        }
        try QPackFiles.ensureDirectory(directory)
        _ = try Qpack.verify(containerDir: directory)
        let files = try QPackFiles(directory: directory, sizes: plan.fileBytes, create: false)
        var prefix = Data(count: plan.densePrefix.count)
        try files.read(into: &prefix, range: 0..<prefix.count, file: "model.safetensors", offset: 0)
        guard prefix == plan.densePrefix else {
            throw OrnithPerformanceError.integrity("dense tensor header changed")
        }
        guard try readSmall(directory.appendingPathComponent("packed_experts/layout.json"), maximum: 1 << 20) == plan.layoutData else {
            throw OrnithPerformanceError.integrity("QPack layout changed")
        }
        for name in expected.sorted() {
            try Task.checkCancellation()
            guard try fileHash(directory.appendingPathComponent(name)) == receipt.fileSHA256[name] else {
                throw OrnithPerformanceError.integrity("installed file changed: \(name)")
            }
        }
        for auxiliary in plan.source.manifest.auxiliary {
            guard receipt.fileSHA256[auxiliary.name] == auxiliary.sha256 else {
                throw OrnithPerformanceError.integrity("source tokenizer/config changed")
            }
        }
        try verifyReconstructedSource(files: files)
        onProgress(.init(phase: .complete, completedSourceBytes: total, totalSourceBytes: total))
        return VerifiedQPackInstall(directory: directory, sourceIdentity: identity, verifiedSourceBytes: total)
    }

    private func reconstruct(shard: Int, range: Range<Int>, files: QPackFiles) throws -> Data {
        var data = Data(count: range.count)
        let header = plan.source.sourceHeaders[shard]
        let hi = min(range.upperBound, header.count)
        if range.lowerBound < hi {
            data.replaceSubrange(0..<(hi - range.lowerBound), with: header[range.lowerBound..<hi])
        }
        for piece in try plan.pieces(shard: shard, range: range) {
            let lo = piece.sourceRange.lowerBound - range.lowerBound
            try files.read(into: &data, range: lo..<(lo + piece.sourceRange.count),
                           file: piece.file, offset: piece.destinationOffset)
        }
        return data
    }

    private func verifyReconstructedSource(files: QPackFiles) throws {
        for (index, shard) in plan.source.manifest.shards.enumerated() {
            var cursor = 0
            var hash = SHA256()
            while cursor < shard.byteCount {
                try Task.checkCancellation()
                let end = min(try checkedAdd(cursor, windowBytes), shard.byteCount)
                hash.update(data: try reconstruct(shard: index, range: cursor..<end, files: files))
                cursor = end
            }
            let actual = hash.finalize().map { String(format: "%02x", $0) }.joined()
            guard actual == shard.sha256 else {
                throw OrnithPerformanceError.integrity("converted source bytes differ: \(shard.name)")
            }
        }
    }

    private func checkHeaderBytes(_ data: Data, shard: Int, offset: Int) throws {
        let expected = plan.source.sourceHeaders[shard]
        let end = min(offset + data.count, expected.count)
        if offset < end, data.prefix(end - offset) != expected[offset..<end] {
            throw OrnithPerformanceError.integrity("source header changed")
        }
    }

    private func fetch(name: String, range: Range<Int>?, total: Int) async throws -> Data {
        for attempt in 0..<3 {
            do { return try await source.data(name: name, range: range, totalByteCount: total) }
            catch let error as URLError where attempt < 2 && [.timedOut, .networkConnectionLost,
                .notConnectedToInternet, .cannotConnectToHost].contains(error.code) {
                try Task.checkCancellation()
                try await Task.sleep(nanoseconds: UInt64(attempt + 1) * 500_000_000)
            }
        }
        throw OrnithPerformanceError.io("network retry exhausted")
    }

    private func checkDiskSpace(staging: URL) throws {
        let attributes = try FileManager.default.attributesOfFileSystem(forPath: root.path)
        guard let free = attributes[.systemFreeSize] as? NSNumber else {
            throw OrnithPerformanceError.io("cannot determine free internal storage")
        }
        let partial = try QPackFiles.allocatedBytes(in: staging, names: Array(plan.fileBytes.keys))
        let auxiliary = try plan.source.manifest.auxiliary.reduce(0) { try checkedAdd($0, $1.byteCount) }
        let needed = UInt64(try checkedAdd(plan.finalWeightFileBytes, auxiliary))
        guard free.uint64Value >= (needed > partial ? needed - partial : 0),
              free.uint64Value - (needed > partial ? needed - partial : 0) >= diskReserveBytes else {
            throw OrnithPerformanceError.io("insufficient free internal storage for final QPack and installation reserve")
        }
    }

    private func fileHash(_ url: URL) throws -> String {
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        var hasher = SHA256()
        while let data = try file.read(upToCount: 1 << 20), !data.isEmpty {
            try Task.checkCancellation()
            hasher.update(data: data)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private func readSmall(_ url: URL, maximum: Int) throws -> Data {
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        let data = try file.read(upToCount: maximum + 1) ?? Data()
        guard data.count <= maximum else { throw OrnithPerformanceError.integrity("oversized local metadata") }
        return data
    }

    private func writeJSON<T: Encodable>(_ value: T, to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        try encoder.encode(value).write(to: url, options: [.atomic])
    }
}
