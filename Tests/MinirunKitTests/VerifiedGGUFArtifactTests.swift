import Darwin
import Foundation
import GGUFCore
import XCTest
@testable import MinirunKit

final class VerifiedGGUFArtifactTests: XCTestCase {
    func testPrimaryProvenanceMatchesPinnedDownloadAndLicenseSource() throws {
        let manifest = GGUFModelManifest.primaryREAM
        let provenance = try XCTUnwrap(manifest.provenance)
        let expectedURL = provenance.artifact.url + "/resolve/" + provenance.artifact.revision + "/" + manifest.fileName
        XCTAssertEqual(PinnedGGUFDownload.primaryREAM.url.absoluteString, expectedURL)
        XCTAssertEqual(provenance.license.sourceURL,
            provenance.inspectedBaseModel.url + "/resolve/" + provenance.inspectedBaseModel.revision + "/LICENSE")
        for source in [provenance.artifact, provenance.inspectedSourceModel, provenance.inspectedBaseModel] {
            XCTAssertEqual(URL(string: source.url)?.host, "huggingface.co")
            XCTAssertEqual(source.revision.count, 40)
        }
        let restored = try JSONDecoder().decode(GGUFModelProvenance.self,
            from: JSONEncoder().encode(provenance))
        XCTAssertEqual(restored, provenance)
        let local = try GGUFModelManifest(fileName: "synthetic.gguf", fileByteCount: 32,
            sha256: String(repeating: "0", count: 64))
        XCTAssertNil(local.provenance, "A local fixture must not inherit a trained-model source/license")
    }

    private func withModel(vocabularyCount: Int = 0, _ body: (URL, GGUFModelManifest) throws -> Void) throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        var data = Data("GGUF".utf8)
        func put(_ number: UInt64, _ bytes: Int) {
            for i in 0..<bytes { data.append(UInt8(truncatingIfNeeded: number >> (i * 8))) }
        }
        put(3, 4); put(1, 8); put(vocabularyCount == 0 ? 0 : 1, 8)
        if vocabularyCount > 0 {
            let key = Data("tokenizer.ggml.tokens".utf8)
            put(UInt64(key.count), 8); data.append(key); put(9, 4); put(8, 4)
            put(UInt64(vocabularyCount), 8)
            for i in 0..<vocabularyCount {
                let token = Data("token-\(i)".utf8); put(UInt64(token.count), 8); data.append(token)
            }
        }
        let name = Data("matrix".utf8); put(UInt64(name.count), 8); data.append(name)
        put(2, 4); put(4, 8); put(2, 8); put(0, 4); put(0, 8)
        data.append(Data(repeating: 0, count: ((data.count + 31) / 32 * 32) - data.count))
        data.append(Data(repeating: 0, count: 32))
        let manifest = try GGUFModelManifest(fileName: "model.fixture", fileByteCount: UInt64(data.count),
                                             sha256: FileDigestComputer.sha256(of: data))
        try data.write(to: root.appendingPathComponent(manifest.fileName))
        try body(root, manifest)
    }

    func testVerifiedCapabilityDuplicatesTheHashedDescriptor() throws {
        try withModel { root, manifest in
            let artifact = try VerifiedGGUFArtifact.verify(insideRoot: root, manifest: manifest)
            XCTAssertEqual(artifact.index.tensors["matrix"]?.byteCount, 32)
            let descriptor = try artifact.duplicateForReading()
            defer { _ = close(descriptor) }
            var bytes = [UInt8](repeating: 0, count: 4)
            let count = bytes.withUnsafeMutableBytes { pread(descriptor, $0.baseAddress!, 4, 0) }
            XCTAssertEqual(count, 4)
            XCTAssertEqual(bytes, Array("GGUF".utf8))
            try artifact.assertUnchanged()
        }
    }

    func testWrongDigestAndSizeDoNotProduceCapability() throws {
        try withModel { root, manifest in
            let wrongHash = try GGUFModelManifest(fileName: manifest.fileName,
                fileByteCount: manifest.fileByteCount, sha256: String(repeating: "0", count: 64))
            XCTAssertThrowsError(try VerifiedGGUFArtifact.verify(insideRoot: root, manifest: wrongHash))
            let wrongSize = try GGUFModelManifest(fileName: manifest.fileName,
                fileByteCount: manifest.fileByteCount + 1, sha256: manifest.sha256)
            XCTAssertThrowsError(try VerifiedGGUFArtifact.verify(insideRoot: root, manifest: wrongSize))
        }
    }

    func testTokenizerTablesAreOptInAndReadOnlyFromVerifiedDescriptor() throws {
        try withModel(vocabularyCount: 5000) { root, manifest in
            let artifact = try VerifiedGGUFArtifact.verify(insideRoot: root, manifest: manifest)
            XCTAssertEqual(artifact.index.metadata["tokenizer.ggml.tokens"], .omittedArray(type: 8, count: 5000))
            let loaded = try artifact.tokenizerMetadata()
            XCTAssertEqual(loaded.metadata["tokenizer.ggml.tokens"]?.elements?.count, 5000)
            XCTAssertEqual(loaded.metadata["tokenizer.ggml.tokens"]?.elements?.last?.text, "token-4999")
            XCTAssertEqual(loaded.tensors, artifact.index.tensors)
        }
    }

    func testMutationInvalidatesExistingCapability() throws {
        try withModel { root, manifest in
            let artifact = try VerifiedGGUFArtifact.verify(insideRoot: root, manifest: manifest)
            let handle = try FileHandle(forWritingTo: root.appendingPathComponent(manifest.fileName))
            try handle.seekToEnd(); try handle.write(contentsOf: Data([1])); try handle.close()
            XCTAssertThrowsError(try artifact.assertUnchanged())
            XCTAssertThrowsError(try artifact.duplicateForReading())
            XCTAssertThrowsError(try artifact.tokenizerMetadata())
        }
    }

    func testNoFollowRejectsModelSymlink() throws {
        try withModel { root, manifest in
            let file = root.appendingPathComponent(manifest.fileName)
            let moved = root.appendingPathComponent("target")
            try FileManager.default.moveItem(at: file, to: moved)
            try FileManager.default.createSymbolicLink(at: file, withDestinationURL: moved)
            XCTAssertThrowsError(try VerifiedGGUFArtifact.verify(insideRoot: root, manifest: manifest))
        }
    }

    func testCancellationAndManifestPathTraversalAreRejected() throws {
        try withModel { root, manifest in
            XCTAssertThrowsError(try VerifiedGGUFArtifact.verify(insideRoot: root, manifest: manifest,
                                                                 cancellationRequested: { true }))
            for name in ["../model", ".", "..", "/absolute", "nested/file", "nested\\file"] {
                XCTAssertThrowsError(try GGUFModelManifest(fileName: name, fileByteCount: 32, sha256: manifest.sha256))
            }
        }
    }
}
