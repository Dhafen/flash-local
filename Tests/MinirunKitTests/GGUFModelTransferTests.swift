import Foundation
import GGUFCore
import XCTest
@testable import MinirunKit

final class GGUFModelTransferTests: XCTestCase {
    func testPinnedHTTPSOriginHonorsBoundedRangesAndIfRangeThroughURLSession() async throws {
        let env = ProcessInfo.processInfo.environment
        guard (env["FLASH_LOCAL_TEST_RANGE"] ?? env["TEST_RUNNER_FLASH_LOCAL_TEST_RANGE"]) == "1" else {
            throw XCTSkip("Opt in to the eight-byte published-header HTTP test with FLASH_LOCAL_TEST_RANGE=1")
        }
        let source = PinnedGGUFDownload.primaryREAM
        let transport = GGUFHTTPRangeTransport()
        defer { transport.close() }
        let first = try await transport.fetch(url: source.url, range: 0..<4,
            totalBytes: source.manifest.fileByteCount, etag: nil)
        XCTAssertEqual(first.bytes, Data("GGUF".utf8))
        let second = try await transport.fetch(url: source.url, range: 4..<8,
            totalBytes: source.manifest.fileByteCount, etag: first.etag)
        XCTAssertEqual(second.bytes, Data([3, 0, 0, 0]))
        XCTAssertEqual(second.etag, first.etag)
        transport.close()
        do {
            _ = try await transport.fetch(url: source.url, range: 0..<4,
                totalBytes: source.manifest.fileByteCount, etag: first.etag)
            XCTFail("Closed transport must refuse another request")
        } catch { XCTAssertTrue(error is CancellationError) }
        print("REAM_URLSESSION_RANGE bytes=8 exact206=true sameETag=true closed=true noTensorPayload=true")
    }

    private final class Flag: @unchecked Sendable {
        private let lock = NSLock(); private var value = false
        func set() { lock.lock(); value = true; lock.unlock() }
        func get() -> Bool { lock.lock(); defer { lock.unlock() }; return value }
    }
    private final class Transport: GGUFRangeTransport, @unchecked Sendable {
        let bytes: Data
        let etag: String
        let shortBody: Bool
        private let lock = NSLock()
        private var requests: [(Range<UInt64>, String?)] = []
        private var closed = false
        init(_ bytes: Data, etag: String = "\"version-one\"", shortBody: Bool = false) {
            self.bytes = bytes; self.etag = etag; self.shortBody = shortBody
        }
        var recorded: [(Range<UInt64>, String?)] { lock.lock(); defer { lock.unlock() }; return requests }
        func close() { lock.lock(); closed = true; lock.unlock() }
        func fetch(url: URL, range: Range<UInt64>, totalBytes: UInt64, etag: String?) async throws -> GGUFRangeChunk {
            try response(range: range, total: totalBytes, expected: etag)
        }
        private func response(range: Range<UInt64>, total: UInt64, expected: String?) throws -> GGUFRangeChunk {
            lock.lock(); defer { lock.unlock() }
            guard !closed else { throw CancellationError() }
            requests.append((range, expected))
            XCTAssertEqual(total, UInt64(bytes.count))
            var data = Data(bytes[Int(range.lowerBound)..<Int(range.upperBound)])
            if shortBody { data.removeLast() }
            return GGUFRangeChunk(bytes: data, etag: etag)
        }
    }
    private func fixture() throws -> (URL, Data, PinnedGGUFDownload) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        var bytes = Data("GGUF".utf8)
        func put(_ value: UInt64, _ count: Int) {
            for i in 0..<count { bytes.append(UInt8(truncatingIfNeeded: value >> (i * 8))) }
        }
        put(3, 4); put(1, 8); put(0, 8)
        let name = Data("test.weight".utf8); put(UInt64(name.count), 8); bytes.append(name)
        put(2, 4); put(4096, 8); put(1, 8); put(0, 4); put(0, 8)
        while bytes.count % 32 != 0 { bytes.append(0) }
        bytes.append(Data(repeating: 0, count: 16384))
        let manifest = try GGUFModelManifest(fileName: "model.gguf", fileByteCount: UInt64(bytes.count),
            sha256: FileDigestComputer.sha256(of: bytes))
        return (root, bytes, try PinnedGGUFDownload(url: URL(string: "https://example.invalid/pinned-model")!, manifest: manifest))
    }

    func testCompleteDownloadIsVerifiedPublishedWithoutAnotherFullSizeCopy() async throws {
        let (root, bytes, source) = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        let transport = Transport(bytes)
        let artifact = try await GGUFModelTransfer(chunkBytes: 4096, reservedFreeBytes: 0)
            .download(insideRoot: root, source: source, transport: transport)
        XCTAssertEqual(artifact.manifest, source.manifest)
        XCTAssertEqual(try Data(contentsOf: root.appendingPathComponent("model.gguf")), bytes)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: root.path), ["model.gguf"])
        XCTAssertEqual(transport.recorded.count, 5)
        XCTAssertNil(transport.recorded.first?.1)
        XCTAssertTrue(transport.recorded.dropFirst().allSatisfy { $0.1 == "\"version-one\"" })
        try artifact.assertUnchanged()
    }

    func testCancellationResumeChecksCommittedChunksAndDiscardsOnlyUncommittedTail() async throws {
        let (root, bytes, source) = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        let flag = Flag(), first = Transport(bytes)
        do {
            _ = try await GGUFModelTransfer(chunkBytes: 4096, reservedFreeBytes: 0).download(
                insideRoot: root, source: source, transport: first, cancellationRequested: { flag.get() },
                onProgress: { if $0.phase == .downloading { flag.set() } })
            XCTFail("Expected cancellation")
        } catch { XCTAssertTrue(error is CancellationError) }
        XCTAssertEqual(first.recorded.count, 1)
        let staging = root.appendingPathComponent("model.gguf.partial")
        let handle = try FileHandle(forWritingTo: staging)
        try handle.seekToEnd(); try handle.write(contentsOf: Data([9, 8, 7])); try handle.close()
        let resumed = Transport(bytes)
        _ = try await GGUFModelTransfer(chunkBytes: 4096, reservedFreeBytes: 0)
            .download(insideRoot: root, source: source, transport: resumed)
        XCTAssertEqual(resumed.recorded.first?.0.lowerBound, 4096)
        XCTAssertEqual(resumed.recorded.first?.1, "\"version-one\"")
        XCTAssertEqual(try Data(contentsOf: root.appendingPathComponent("model.gguf")), bytes)
    }

    func testCorruptedResumeChunkRefusesBeforeAnyNetworkRequest() async throws {
        let (root, bytes, source) = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        let flag = Flag()
        do {
            _ = try await GGUFModelTransfer(chunkBytes: 4096, reservedFreeBytes: 0).download(
                insideRoot: root, source: source, transport: Transport(bytes), cancellationRequested: { flag.get() },
                onProgress: { if $0.phase == .downloading { flag.set() } })
        } catch { XCTAssertTrue(error is CancellationError) }
        let handle = try FileHandle(forWritingTo: root.appendingPathComponent("model.gguf.partial"))
        try handle.write(contentsOf: Data([99])); try handle.close()
        let resumed = Transport(bytes)
        do {
            _ = try await GGUFModelTransfer(chunkBytes: 4096, reservedFreeBytes: 0)
                .download(insideRoot: root, source: source, transport: resumed)
            XCTFail("Expected corrupt chunk refusal")
        } catch { XCTAssertTrue(resumed.recorded.isEmpty) }
    }

    func testChangedETagAndShortBodiesDoNotCommitBytes() async throws {
        for short in [false, true] {
            let (root, bytes, source) = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
            let transport = Transport(bytes, shortBody: short)
            let flag = Flag()
            if !short {
                do {
                    _ = try await GGUFModelTransfer(chunkBytes: 4096, reservedFreeBytes: 0).download(
                        insideRoot: root, source: source, transport: transport, cancellationRequested: { flag.get() },
                        onProgress: { if $0.phase == .downloading { flag.set() } })
                } catch { XCTAssertTrue(error is CancellationError) }
            }
            let invalid = short ? transport : Transport(bytes, etag: "\"changed\"")
            do {
                _ = try await GGUFModelTransfer(chunkBytes: 4096, reservedFreeBytes: 0)
                    .download(insideRoot: root, source: source, transport: invalid)
                XCTFail("Expected bad response refusal")
            } catch {
                XCTAssertEqual(try FileDigestComputer.byteCount(ofFileAt: root.appendingPathComponent("model.gguf.partial")), short ? 0 : 4096)
                XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("model.gguf").path))
            }
        }
    }

    func testFullHashMismatchAndExistingModelNeverPublishOrOverwrite() async throws {
        let (root, bytes, source) = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        let wrong = try GGUFModelManifest(fileName: "model.gguf", fileByteCount: UInt64(bytes.count), sha256: String(repeating: "0", count: 64))
        do {
            _ = try await GGUFModelTransfer(chunkBytes: 4096, reservedFreeBytes: 0).download(insideRoot: root,
                source: PinnedGGUFDownload(url: source.url, manifest: wrong), transport: Transport(bytes))
            XCTFail("Expected full SHA-256 refusal")
        } catch { XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("model.gguf").path)) }
        let installed = root.appendingPathComponent("model.gguf")
        try Data([1, 2, 3]).write(to: installed)
        do {
            _ = try await GGUFModelTransfer(chunkBytes: 4096, reservedFreeBytes: 0)
                .download(insideRoot: root, source: source, transport: Transport(bytes))
            XCTFail("Expected existing file refusal")
        } catch { XCTAssertEqual(try Data(contentsOf: installed), Data([1, 2, 3])) }
    }

    func testSymlinkAndInsufficientDiskReserveRefuseBeforeNetwork() async throws {
        let (root, bytes, source) = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        let transport = Transport(bytes)
        do {
            _ = try await GGUFModelTransfer(chunkBytes: 4096, reservedFreeBytes: UInt64.max)
                .download(insideRoot: root, source: source, transport: transport)
            XCTFail("Expected free-space refusal")
        } catch { XCTAssertTrue(transport.recorded.isEmpty) }
        try FileManager.default.removeItem(at: root.appendingPathComponent("model.gguf.partial"))
        let target = root.appendingPathComponent("untouched")
        try Data([42]).write(to: target)
        try FileManager.default.createSymbolicLink(at: root.appendingPathComponent("model.gguf.partial"), withDestinationURL: target)
        do {
            _ = try await GGUFModelTransfer(chunkBytes: 4096, reservedFreeBytes: 0)
                .download(insideRoot: root, source: source, transport: Transport(bytes))
            XCTFail("Expected no-follow refusal")
        } catch { XCTAssertEqual(try Data(contentsOf: target), Data([42])) }
    }

    func testHTTPHeaderAdmissionRejectsIgnoredRangeChangedIdentityAndCompression() throws {
        func response(_ headers: [String: String], status: Int = 206) -> HTTPURLResponse {
            HTTPURLResponse(url: URL(string: "https://example.invalid/model")!, statusCode: status, httpVersion: "HTTP/1.1", headerFields: headers)!
        }
        let valid = ["Content-Range": "bytes 0-4095/9000", "Content-Length": "4096", "ETag": "\"opaque\""]
        XCTAssertEqual(try GGUFHTTPRangeTransport.admit(response(valid), range: 0..<4096, total: 9000, etag: nil), "\"opaque\"")
        XCTAssertThrowsError(try GGUFHTTPRangeTransport.admit(response(valid, status: 200), range: 0..<4096, total: 9000, etag: nil))
        for key in ["Content-Range", "Content-Length", "ETag", "Content-Encoding"] {
            var changed = valid; changed[key] = key == "Content-Encoding" ? "gzip" : "invalid"
            XCTAssertThrowsError(try GGUFHTTPRangeTransport.admit(response(changed), range: 0..<4096, total: 9000, etag: nil))
        }
        XCTAssertThrowsError(try GGUFHTTPRangeTransport.admit(response(valid), range: 0..<4096, total: 9000, etag: "\"older\""))
    }
}
