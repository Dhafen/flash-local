import Darwin
import Foundation
import GGUFCore

public struct GGUFModelManifest: Sendable, Equatable {
    public let fileName: String
    public let fileByteCount: UInt64
    public let sha256: String

    public init(fileName: String, fileByteCount: UInt64, sha256: String) throws {
        guard !fileName.isEmpty, fileName != ".", fileName != "..",
              !fileName.contains("/"), !fileName.contains("\\"), !fileName.contains("\0"),
              fileByteCount > 0, sha256.utf8.count == 64,
              sha256.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else {
            throw GGUFError.invalid("GGUF manifest")
        }
        self.fileName = fileName
        self.fileByteCount = fileByteCount
        self.sha256 = sha256
    }

    /// LFS SHA-256 from the pinned Hugging Face tree; not a locally invented
    /// checksum. The header-only tests do not prove payload integrity.
    public static let primaryREAM = try! GGUFModelManifest(
        fileName: "Qwen3.8-Flash-Next-REAM-60Pct-HQ-PLE4of16-56GiB.gguf",
        fileByteCount: 56_115_609_536,
        sha256: "d18d0a0e017946aa54f063b2abd92ef96a0d7bd64948cefe9ac3b0ff535dc5fb")
}

/// Capability for exactly the file that was hashed, not a path that can be
/// reopened later. Verification is synchronous and cancellable: call off the
/// main actor. No inference is enabled merely by obtaining this capability.
public final class VerifiedGGUFArtifact: @unchecked Sendable {
    public let index: GGUFIndex
    public let manifest: GGUFModelManifest
    private let descriptor: Int32
    private let stamp: Stamp

    private struct Stamp: Equatable {
        let device: dev_t
        let inode: ino_t
        let bytes: off_t
        let modifiedSeconds: Int
        let modifiedNanoseconds: Int
        let changedSeconds: Int
        let changedNanoseconds: Int

        init(_ value: stat) {
            device = value.st_dev; inode = value.st_ino; bytes = value.st_size
            modifiedSeconds = value.st_mtimespec.tv_sec
            modifiedNanoseconds = value.st_mtimespec.tv_nsec
            changedSeconds = value.st_ctimespec.tv_sec
            changedNanoseconds = value.st_ctimespec.tv_nsec
        }
    }

    private init(descriptor: Int32, stamp: Stamp, index: GGUFIndex, manifest: GGUFModelManifest) {
        self.descriptor = descriptor; self.stamp = stamp
        self.index = index; self.manifest = manifest
    }

    deinit { _ = Darwin.close(descriptor) }

    public static func verify(
        insideRoot root: URL, manifest: GGUFModelManifest,
        cancellationRequested: @escaping @Sendable () -> Bool = { false }
    ) throws -> VerifiedGGUFArtifact {
        guard root.isFileURL else { throw GGUFError.invalid("non-file model root") }
        let directory = Darwin.open(root.path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard directory >= 0 else { throw posixError() }
        defer { _ = Darwin.close(directory) }
        let descriptor = Darwin.openat(directory, manifest.fileName, O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
        guard descriptor >= 0 else { throw posixError() }
        var transferred = false
        defer { if !transferred { _ = Darwin.close(descriptor) } }
        let initial = try status(descriptor)
        guard initial.st_mode & mode_t(S_IFMT) == mode_t(S_IFREG), initial.st_size >= 0,
              UInt64(initial.st_size) == manifest.fileByteCount else {
            throw GGUFError.invalid("model file type/size disagrees with manifest")
        }
        let stamp = Stamp(initial)
        let digest = try FileDigestComputer.digest(ofOpenFileDescriptor: descriptor, algorithm: .sha256,
                                                   cancellationRequested: cancellationRequested)
        guard digest.matches(manifest.sha256), Stamp(try status(descriptor)) == stamp else {
            throw GGUFError.invalid("model digest mismatch or file changed during verification")
        }
        // Metadata prefix is bounded separately from the hasher's 4 MiB chunk.
        // Partial prefixes are grown only on a truncated-header result.
        let limit = min(UInt64(GGUFParseLimits().headerBytes), manifest.fileByteCount)
        var count = min(UInt64(64 * 1024), limit)
        var parsedIndex: GGUFIndex?
        while parsedIndex == nil {
            if cancellationRequested() { throw CancellationError() }
            var prefix = Data(count: Int(count))
            try prefix.withUnsafeMutableBytes { bytes in
                var filled = 0
                while filled < bytes.count {
                    let read = Darwin.pread(descriptor, bytes.baseAddress! + filled,
                                            bytes.count - filled, off_t(filled))
                    if read < 0, errno == EINTR { continue }
                    guard read > 0 else { throw GGUFError.truncated }
                    filled += read
                }
            }
            do {
                parsedIndex = try GGUFIndex.parse(header: prefix, fileByteCount: manifest.fileByteCount)
            } catch GGUFError.truncated where count < limit {
                count = min(count * 2, limit)
            }
        }
        guard let index = parsedIndex else { throw GGUFError.truncated }
        guard Stamp(try status(descriptor)) == stamp else {
            throw GGUFError.invalid("model changed during metadata inspection")
        }
        transferred = true
        return VerifiedGGUFArtifact(descriptor: descriptor, stamp: stamp, index: index, manifest: manifest)
    }

    /// Call before/after each streamed operation, and before publishing logits.
    public func assertUnchanged() throws {
        guard Stamp(try Self.status(descriptor)) == stamp else {
            throw GGUFError.invalid("verified model file changed")
        }
    }

    /// TileReader takes ownership of this duplicate. The verified file is never
    /// reopened by pathname, including after a Files rename/replacement.
    public func duplicateForReading() throws -> Int32 {
        try assertUnchanged()
        let duplicate = fcntl(descriptor, F_DUPFD_CLOEXEC, 0)
        guard duplicate >= 0 else { throw Self.posixError() }
        return duplicate
    }

    /// Load the large tokenizer tables only when a runner actually needs them.
    /// Discovery/layout inspection keeps using `index`, which omits them.
    /// The bytes come from the hashed descriptor, with the same 32 MiB header
    /// and one-million retained-element bounds as the GGUF parser.
    public func tokenizerMetadata() throws -> GGUFIndex {
        try assertUnchanged()
        let count = index.headerByteCount
        guard count > 0, count <= GGUFParseLimits().headerBytes else {
            throw GGUFError.limit("verified tokenizer header")
        }
        var header = Data(count: count)
        try header.withUnsafeMutableBytes { bytes in
            var filled = 0
            while filled < count {
                let n = Darwin.pread(descriptor, bytes.baseAddress! + filled,
                                     count - filled, off_t(filled))
                if n < 0, errno == EINTR { continue }
                guard n > 0 else { throw GGUFError.truncated }
                filled += n
            }
        }
        var limits = GGUFParseLimits()
        limits.largeArrayKeys = ["tokenizer.ggml.tokens", "tokenizer.ggml.token_type", "tokenizer.ggml.merges"]
        let result = try GGUFIndex.parse(header: header, fileByteCount: manifest.fileByteCount, limits: limits)
        try assertUnchanged()
        return result
    }

    private static func status(_ descriptor: Int32) throws -> stat {
        var value = stat()
        guard fstat(descriptor, &value) == 0 else { throw posixError() }
        return value
    }

    private static func posixError() -> NSError {
        NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
    }
}
