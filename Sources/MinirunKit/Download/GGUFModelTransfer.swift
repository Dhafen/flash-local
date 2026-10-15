import Darwin
import Foundation
import GGUFCore

public struct PinnedGGUFDownload: Sendable {
    public let url: URL
    public let manifest: GGUFModelManifest
    public init(url: URL, manifest: GGUFModelManifest) throws {
        guard url.scheme == "https", url.user == nil, url.password == nil, url.query == nil,
              manifest.fileByteCount <= 80_000_000_000 else { throw GGUFError.invalid("pinned model download") }
        self.url = url; self.manifest = manifest
    }
    public static let primaryREAM = try! PinnedGGUFDownload(
        url: URL(string: "https://huggingface.co/RemySkye/Qwen3.8-Flash-Next-REAM-60Pct-GGUF/resolve/28a55fecdca51a498900aeb34a4e7e8338977cd3/Qwen3.8-Flash-Next-REAM-60Pct-HQ-PLE4of16-56GiB.gguf")!,
        manifest: .primaryREAM)
}

/// Resumable bounded ranges written directly to the final local volume. Only
/// fully fsynced chunks enter the receipt. Resume rehashes those chunks before
/// requesting more bytes. The final name is published atomically after full
/// SHA-256/GGUF verification, then verified again to bind a fresh capability to
/// the renamed inode. No second full-size file is created.
public struct GGUFModelTransfer: Sendable {
    public enum Phase: String, Sendable { case checkingResume, downloading, verifying, ready }
    public struct Progress: Sendable {
        public let phase: Phase
        public let completedBytes: UInt64
        public let totalBytes: UInt64
    }
    private struct Receipt: Codable {
        var version = 1
        let source: String
        let sha256: String
        let totalBytes: UInt64
        let chunkBytes: Int
        var committedBytes: UInt64 = 0
        var etag: String?
        var hashes: [String] = []
    }
    public let chunkBytes: Int
    public let reservedFreeBytes: UInt64
    public init(chunkBytes: Int = 8 << 20, reservedFreeBytes: UInt64 = 2 << 30) throws {
        guard (4096...GGUFHTTPRangeTransport.maximumChunkBytes).contains(chunkBytes) else {
            throw GGUFError.limit("download chunk must be between 4 KiB and 8 MiB")
        }
        self.chunkBytes = chunkBytes; self.reservedFreeBytes = reservedFreeBytes
    }

    public func download(insideRoot root: URL, source: PinnedGGUFDownload = .primaryREAM,
                         transport: any GGUFRangeTransport = GGUFHTTPRangeTransport(),
                         cancellationRequested: @escaping @Sendable () -> Bool = { false },
                         onProgress: @escaping @Sendable (Progress) -> Void = { _ in }) async throws -> VerifiedGGUFArtifact {
        defer { transport.close() }
        try LocalModelStorage.prepare(root: root)
        let directory = Darwin.open(root.path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard directory >= 0 else { throw Self.posix() }
        defer { _ = Darwin.close(directory) }
        let manifest = source.manifest
        let partial = manifest.fileName + ".partial", receiptName = partial + ".json"
        var installed = stat()
        if fstatat(directory, manifest.fileName, &installed, AT_SYMLINK_NOFOLLOW) == 0 {
            throw GGUFError.invalid("model file already exists; verify it instead of overwriting it")
        }
        guard errno == ENOENT else { throw Self.posix() }
        let fd = openat(directory, partial, O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC, mode_t(0o600))
        guard fd >= 0 else { throw Self.posix() }
        defer { _ = Darwin.close(fd) }
        guard flock(fd, LOCK_EX | LOCK_NB) == 0 else { throw GGUFError.invalid("model download is already active") }
        let size = try Self.size(fd)
        var receipt: Receipt
        if let data = try Self.readReceipt(directory: directory, name: receiptName) {
            receipt = try JSONDecoder().decode(Receipt.self, from: data)
            let expectedBytes = min(UInt64(receipt.hashes.count) * UInt64(chunkBytes), manifest.fileByteCount)
            guard receipt.version == 1, receipt.source == source.url.absoluteString,
                  receipt.sha256 == manifest.sha256, receipt.totalBytes == manifest.fileByteCount,
                  receipt.chunkBytes == chunkBytes, receipt.hashes.count <= 20000,
                  UInt64(receipt.hashes.count) <= (manifest.fileByteCount + UInt64(chunkBytes) - 1) / UInt64(chunkBytes),
                  receipt.committedBytes == expectedBytes, size >= expectedBytes,
                  receipt.committedBytes == 0 || receipt.etag != nil,
                  receipt.hashes.allSatisfy({ $0.utf8.count == 64 }) else {
                throw GGUFError.invalid("resume receipt does not match this model/file")
            }
        } else {
            guard size == 0 else { throw GGUFError.invalid("partial model has no verified resume receipt") }
            receipt = Receipt(source: source.url.absoluteString, sha256: manifest.sha256,
                              totalBytes: manifest.fileByteCount, chunkBytes: chunkBytes)
        }
        func check() throws {
            if cancellationRequested() || Task.isCancelled { throw CancellationError() }
        }
        for (i, digest) in receipt.hashes.enumerated() {
            try check()
            let offset = UInt64(i) * UInt64(chunkBytes)
            let count = Int(min(UInt64(chunkBytes), manifest.fileByteCount - offset))
            let data = try Self.read(fd: fd, offset: offset, count: count)
            guard FileDigestComputer.sha256(of: data) == digest else { throw GGUFError.invalid("partial model chunk changed") }
            onProgress(.init(phase: .checkingResume, completedBytes: offset + UInt64(count), totalBytes: receipt.committedBytes))
        }
        // A crash may leave bytes from an uncommitted chunk. Only this owned
        // staging file is truncated, to the last fsynced, hash-checked receipt.
        guard ftruncate(fd, off_t(receipt.committedBytes)) == 0 else { throw Self.posix() }
        while receipt.committedBytes < manifest.fileByteCount {
            try check()
            let remaining = manifest.fileByteCount - receipt.committedBytes
            let values = try root.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey, .volumeAvailableCapacityKey])
            guard let available = values.volumeAvailableCapacityForImportantUsage ?? values.volumeAvailableCapacity.map(Int64.init),
                  available >= 0, UInt64(available) >= remaining,
                  UInt64(available) - remaining >= reservedFreeBytes else {
                throw GGUFError.limit("not enough free space for the model and storage reserve")
            }
            let end = min(receipt.committedBytes + UInt64(chunkBytes), manifest.fileByteCount)
            let chunk = try await transport.fetch(url: source.url, range: receipt.committedBytes..<end,
                                                   totalBytes: manifest.fileByteCount, etag: receipt.etag)
            try check()
            guard chunk.bytes.count == Int(end - receipt.committedBytes),
                  (2...1024).contains(chunk.etag.utf8.count), chunk.etag.hasPrefix("\""), chunk.etag.hasSuffix("\""),
                  receipt.etag == nil || receipt.etag == chunk.etag else {
                throw GGUFError.invalid("download chunk size or identity changed")
            }
            try Self.write(fd: fd, data: chunk.bytes, offset: receipt.committedBytes)
            guard fsync(fd) == 0 else { throw Self.posix() }
            receipt.committedBytes = end; receipt.etag = chunk.etag
            receipt.hashes.append(FileDigestComputer.sha256(of: chunk.bytes))
            try Self.atomicReceipt(try JSONEncoder().encode(receipt), directory: directory, name: receiptName)
            onProgress(.init(phase: .downloading, completedBytes: end, totalBytes: manifest.fileByteCount))
        }
        try check()
        onProgress(.init(phase: .verifying, completedBytes: 0, totalBytes: manifest.fileByteCount))
        let stagingManifest = try GGUFModelManifest(fileName: partial, fileByteCount: manifest.fileByteCount, sha256: manifest.sha256)
        let staged = try VerifiedGGUFArtifact.verify(insideRoot: root, manifest: stagingManifest,
                                                    cancellationRequested: cancellationRequested)
        try staged.assertUnchanged(); try check()
        // RENAME_EXCL preserves an already installed model even if one appeared
        // while this transfer was in progress. APFS rename adds no file copy.
        guard renameatx_np(directory, partial, directory, manifest.fileName, UInt32(RENAME_EXCL)) == 0 else { throw Self.posix() }
        let result = try VerifiedGGUFArtifact.verify(insideRoot: root, manifest: manifest,
                                                    cancellationRequested: cancellationRequested)
        _ = unlinkat(directory, receiptName, 0)
        onProgress(.init(phase: .ready, completedBytes: manifest.fileByteCount, totalBytes: manifest.fileByteCount))
        return result
    }

    private static func size(_ fd: Int32) throws -> UInt64 {
        var value = stat()
        guard fstat(fd, &value) == 0 else { throw posix() }
        guard value.st_mode & mode_t(S_IFMT) == mode_t(S_IFREG), value.st_size >= 0, value.st_nlink == 1 else {
            throw GGUFError.invalid("non-regular or multiply linked model staging file")
        }
        return UInt64(value.st_size)
    }
    private static func read(fd: Int32, offset: UInt64, count: Int) throws -> Data {
        var data = Data(count: count)
        try data.withUnsafeMutableBytes { bytes in
            var filled = 0
            while filled < count {
                let n = pread(fd, bytes.baseAddress! + filled, count - filled, off_t(offset) + off_t(filled))
                if n < 0, errno == EINTR { continue }
                guard n > 0 else { throw GGUFError.truncated }; filled += n
            }
        }
        return data
    }
    private static func write(fd: Int32, data: Data, offset: UInt64 = 0) throws {
        try data.withUnsafeBytes { bytes in
            var written = 0
            while written < bytes.count {
                let n = pwrite(fd, bytes.baseAddress! + written, bytes.count - written, off_t(offset) + off_t(written))
                if n < 0, errno == EINTR { continue }
                guard n > 0 else { throw posix() }; written += n
            }
        }
    }
    private static func readReceipt(directory: Int32, name: String) throws -> Data? {
        let fd = openat(directory, name, O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
        if fd < 0, errno == ENOENT { return nil }
        guard fd >= 0 else { throw posix() }; defer { _ = Darwin.close(fd) }
        let count = try size(fd)
        guard count <= 1 << 20 else { throw GGUFError.limit("resume receipt exceeds 1 MiB") }
        return try read(fd: fd, offset: 0, count: Int(count))
    }
    private static func atomicReceipt(_ data: Data, directory: Int32, name: String) throws {
        guard data.count <= 1 << 20 else { throw GGUFError.limit("resume receipt exceeds 1 MiB") }
        let temporary = ".gguf-receipt-" + UUID().uuidString
        let fd = openat(directory, temporary, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, mode_t(0o600))
        guard fd >= 0 else { throw posix() }
        defer { _ = Darwin.close(fd); _ = unlinkat(directory, temporary, 0) }
        try write(fd: fd, data: data)
        guard fsync(fd) == 0, renameat(directory, temporary, directory, name) == 0 else { throw posix() }
    }
    private static func posix() -> POSIXError { POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
}
