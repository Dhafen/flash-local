import Darwin
import Foundation

/// Owns only the files of one private staging container. Bounded positional
/// reads/writes support reconstructing the original shard hash on resume.
final class QPackFiles {
    let directory: URL
    private var descriptors: [String: Int32] = [:]
    private let sizes: [String: Int]

    init(directory: URL, sizes: [String: Int], create: Bool) throws {
        self.directory = directory
        self.sizes = sizes
        for (name, bytes) in sizes.sorted(by: { $0.key < $1.key }) {
            guard name == "model.safetensors" || name.range(of: #"^packed_experts/layer_[0-9]{2}\.bin$"#,
                                                           options: .regularExpression) != nil,
                  bytes > 0 else { throw OrnithPerformanceError.io("invalid owned file") }
            let path = directory.appendingPathComponent(name).path
            let flags = (create ? O_RDWR | O_CREAT : O_RDONLY) | O_NOFOLLOW
            let fd = open(path, flags, 0o600)
            guard fd >= 0 else { throw OrnithPerformanceError.io("cannot open \(name)") }
            var info = stat()
            guard fstat(fd, &info) == 0, info.st_mode & S_IFMT == S_IFREG, info.st_nlink == 1 else {
                close(fd)
                throw OrnithPerformanceError.io("not a private regular file: \(name)")
            }
            if create {
                guard ftruncate(fd, off_t(bytes)) == 0 else {
                    close(fd)
                    throw OrnithPerformanceError.io("cannot size \(name)")
                }
            } else if info.st_size != bytes {
                close(fd)
                throw OrnithPerformanceError.integrity("file length changed: \(name)")
            }
            descriptors[name] = fd
        }
    }

    deinit { for fd in descriptors.values { close(fd) } }

    func write(_ data: Data, range: Range<Int>, file: String, offset: Int) throws {
        guard let fd = descriptors[file], offset >= 0, range.lowerBound >= 0,
              range.upperBound <= data.count, range.count <= (sizes[file] ?? 0) - offset else {
            throw OrnithPerformanceError.io("write outside owned file")
        }
        try data.withUnsafeBytes { raw in
            var done = 0
            while done < range.count {
                let n = pwrite(fd, raw.baseAddress!.advanced(by: range.lowerBound + done),
                               range.count - done, off_t(offset + done))
                if n < 0 && errno == EINTR { continue }
                guard n > 0 else { throw OrnithPerformanceError.io("short positional write") }
                done += n
            }
        }
    }

    func read(into data: inout Data, range: Range<Int>, file: String, offset: Int) throws {
        guard let fd = descriptors[file], offset >= 0, range.lowerBound >= 0,
              range.upperBound <= data.count, range.count <= (sizes[file] ?? 0) - offset else {
            throw OrnithPerformanceError.io("read outside owned file")
        }
        try data.withUnsafeMutableBytes { raw in
            var done = 0
            while done < range.count {
                let n = pread(fd, raw.baseAddress!.advanced(by: range.lowerBound + done),
                              range.count - done, off_t(offset + done))
                if n < 0 && errno == EINTR { continue }
                guard n > 0 else { throw OrnithPerformanceError.io("short positional read") }
                done += n
            }
        }
    }

    func synchronize(_ files: Set<String>) throws {
        for name in files {
            guard let fd = descriptors[name], fsync(fd) == 0 else {
                throw OrnithPerformanceError.io("cannot synchronize output")
            }
        }
    }

    static func allocatedBytes(in directory: URL, names: [String]) throws -> UInt64 {
        var bytes: UInt64 = 0
        for name in names {
            var info = stat()
            let url = directory.appendingPathComponent(name)
            if lstat(url.path, &info) != 0 {
                if errno == ENOENT { continue }
                throw OrnithPerformanceError.io("cannot inspect partial output")
            }
            guard info.st_mode & S_IFMT == S_IFREG, info.st_nlink == 1, info.st_blocks >= 0 else {
                throw OrnithPerformanceError.io("partial output is not a private file")
            }
            let (next, overflow) = bytes.addingReportingOverflow(UInt64(info.st_blocks) * 512)
            guard !overflow else { throw OrnithPerformanceError.arithmeticOverflow }
            bytes = next
        }
        return bytes
    }

    static func ensureDirectory(_ url: URL) throws {
        var isDirectory: ObjCBool = false
        let fm = FileManager.default
        if fm.fileExists(atPath: url.path, isDirectory: &isDirectory) {
            let values = try url.resourceValues(forKeys: [.isSymbolicLinkKey])
            guard isDirectory.boolValue, values.isSymbolicLink != true else {
                throw OrnithPerformanceError.io("staging path is not an owned directory")
            }
        } else {
            try fm.createDirectory(at: url, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        }
    }
}
