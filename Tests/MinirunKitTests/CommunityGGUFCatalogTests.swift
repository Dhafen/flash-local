import Foundation
import XCTest
@testable import MinirunKit

final class CommunityGGUFCatalogTests: XCTestCase {
    private let revision = String(repeating: "a", count: 40)
    private let digest = String(repeating: "b", count: 64)

    private func fixture(_ files: [[String: Any]]) throws -> Data {
        try JSONSerialization.data(withJSONObject: files)
    }

    func testDiscoversOnlySHA256PinnedGGUFWithValidSize() throws {
        let data = try fixture([
            ["type": "file", "path": "models/Q4_K_M.gguf", "size": 1234,
             "lfs": ["oid": digest, "size": 1234]],
            ["type": "file", "path": "model.safetensors", "size": 1000,
             "lfs": ["oid": digest, "size": 1000]],
            ["type": "file", "path": "unpinned.gguf", "size": 1000],
            ["type": "file", "path": "bad.gguf", "lfs": ["oid": "not-a-checksum", "size": 2000]],
            ["type": "file", "path": "../escape.gguf", "lfs": ["oid": digest, "size": 900]],
            ["type": "file", "path": "models/Q4_K_M.gguf", "lfs": ["oid": digest, "size": 1234]]
        ])
        let entries = try CommunityGGUFCatalog.parseTree(data, repository: "dphn/Dolphin3.0-Llama3.1-8B-GGUF", revision: revision)
        XCTAssertEqual(entries.count, 1)
        let entry = try XCTUnwrap(entries.first)
        XCTAssertEqual(entry.name, "Q4_K_M.gguf")
        XCTAssertEqual(entry.installedName, "community-" + digest + ".gguf")
        let source = try entry.pinnedSource()
        XCTAssertEqual(source.manifest.fileByteCount, 1234)
        XCTAssertEqual(source.manifest.sha256, digest)
        XCTAssertTrue(source.url.absoluteString.contains("/resolve/" + revision + "/models/Q4_K_M.gguf"))
    }

    func testRejectsUntrustedRepositoryAndRevision() throws {
        let valid = try fixture([["type": "file", "path": "test.gguf",
                                  "lfs": ["oid": digest, "size": 8192]]])
        for repository in ["invalid", "../evil", "name/../bad", "owner/%2e", "owner//repo", "owner/repo?x=1"] {
            XCTAssertThrowsError(try CommunityGGUFCatalog.parseTree(valid, repository: repository, revision: revision))
        }
        XCTAssertThrowsError(try CommunityGGUFCatalog.parseTree(valid, repository: "owner/repo", revision: "main"))
        XCTAssertThrowsError(try CommunityGGUFCatalog.parseTree(valid, repository: "owner/repo",
                                                                 revision: String(repeating: "g", count: 40)))
    }

    func testRejectsUnexpectedModelPathsAndOversizedPayload() throws {
        let rows: [[String: Any]] = [
            ["type": "file", "path": "a/../bad.gguf", "lfs": ["oid": digest, "size": 100]],
            ["type": "file", "path": "bad%2f.gguf", "lfs": ["oid": digest, "size": 100]],
            ["type": "file", "path": "big.gguf", "lfs": ["oid": digest, "size": 80_000_000_001]],
            ["type": "file", "path": "none.gguf", "lfs": ["oid": digest, "size": 0]]
        ]
        XCTAssertTrue(try CommunityGGUFCatalog.parseTree(fixture(rows), repository: "owner/repo",
                                                           revision: revision).isEmpty)
    }

    func testUnverifiedLocalFileIsNotAdmittedToCommunityLibrary() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let item = try XCTUnwrap(CommunityGGUFCatalog.parseTree(fixture([
            ["type": "file", "path": "example.gguf", "lfs": ["oid": digest, "size": 100]]
        ]), repository: "owner/repo", revision: revision).first)
        XCTAssertTrue(CommunityGGUFCatalog.installed(in: root).isEmpty)
        try Data("model".utf8).write(to: root.appendingPathComponent(item.installedName))
        XCTAssertThrowsError(try VerifiedGGUFArtifact.verify(
            insideRoot: root, manifest: item.pinnedSource().manifest))
        XCTAssertTrue(CommunityGGUFCatalog.installed(in: root).isEmpty,
                      "An unverified local file must not be shown as installed")
    }
}
