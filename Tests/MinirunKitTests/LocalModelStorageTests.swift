import Foundation
import XCTest
import iOSBench
@testable import MinirunKit

final class LocalModelStorageTests: XCTestCase {
    private func withDirectory(_ body: (URL) throws -> Void) throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: url) }
        try body(url)
    }

    private func manager(root: URL?, coding: ScriptedURLCoding) -> StorageManager {
        StorageManager(store: SecurityScopedLocationStore(coding: coding, persistence: InMemoryBookmarkPersistence()),
            bookmarkLedger: InMemoryBookmarkLedger(), coding: coding, appOwnedModelRoot: root)
    }

    func testAppOwnedRootWorksWithoutSecurityScopeAndIsExcludedFromBackup() throws {
        try withDirectory { documents in
            let root = documents.appendingPathComponent("Models", isDirectory: true)
            let coding = ScriptedURLCoding(); coding.refusedURLs.insert(root)
            let storage = manager(root: root, coding: coding)
            XCTAssertEqual(storage.knownLocations(), [.appLocalModels])
            XCTAssertEqual(storage.bookmark(.appLocalModels)?.recordedPath, root.path)
            let scope = try storage.resolve(.appLocalModels)
            XCTAssertFalse(scope.isScoped)
            XCTAssertTrue(try root.resourceValues(forKeys: [.isDirectoryKey]).isDirectory == true)
            XCTAssertTrue(try root.resourceValues(forKeys: [.isExcludedFromBackupKey]).isExcludedFromBackup == true)
            scope.release()
            XCTAssertEqual(coding.stopCount, 0, "An unissued security scope must never be stopped")
        }
    }

    func testNoImplicitAuthorityInDefaultOrInjectedPreviewStorage() {
        let coding = ScriptedURLCoding()
        let storage = manager(root: nil, coding: coding)
        XCTAssertFalse(storage.knownLocations().contains(.appLocalModels))
        XCTAssertNil(storage.bookmark(.appLocalModels))
        XCTAssertThrowsError(try storage.resolve(.appLocalModels))
    }

    func testRootPreparationIsIdempotentAndPreservesFiles() throws {
        try withDirectory { documents in
            let root = documents.appendingPathComponent("Models", isDirectory: true)
            try LocalModelStorage.prepare(root: root)
            let file = root.appendingPathComponent("small-fixture")
            try Data([1, 2, 3]).write(to: file)
            try LocalModelStorage.prepare(root: root)
            XCTAssertEqual(try Data(contentsOf: file), Data([1, 2, 3]))
        }
    }

    func testSymlinkAndRegularFileCannotBecomeTheModelRoot() throws {
        try withDirectory { documents in
            let target = documents.appendingPathComponent("outside", isDirectory: true)
            try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
            let root = documents.appendingPathComponent("Models")
            try FileManager.default.createSymbolicLink(at: root, withDestinationURL: target)
            XCTAssertThrowsError(try LocalModelStorage.prepare(root: root))
            try FileManager.default.removeItem(at: root)
            try Data([1]).write(to: root)
            XCTAssertThrowsError(try LocalModelStorage.prepare(root: root))
            XCTAssertEqual(try Data(contentsOf: root), Data([1]))
        }
    }

    func testPerJobLocalBookmarkDoesNotGrantPrefixSiblingOrEscapedSymlink() throws {
        try withDirectory { documents in
            let root = documents.appendingPathComponent("Models", isDirectory: true)
            try LocalModelStorage.prepare(root: root)
            let model = root.appendingPathComponent("model", isDirectory: true)
            let sibling = documents.appendingPathComponent("Models-evil", isDirectory: true)
            try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
            try FileManager.default.createDirectory(at: sibling, withIntermediateDirectories: true)
            let escaped = root.appendingPathComponent("escaped")
            try FileManager.default.createSymbolicLink(at: escaped, withDestinationURL: sibling)
            let coding = ScriptedURLCoding()
            // The fake resolves a path-only URL. Refuse every potential grant;
            // any success must come solely from the explicit app-owned root.
            for url in [model, sibling, escaped] { coding.refusedURLs.insert(URL(fileURLWithPath: url.path)) }
            let storage = manager(root: root, coding: coding)
            let result = try storage.resolveBookmarkData(Data(model.path.utf8))
            XCTAssertFalse(result.scope.isScoped)
            result.scope.release()
            XCTAssertThrowsError(try storage.resolveBookmarkData(Data(sibling.path.utf8)))
            XCTAssertThrowsError(try storage.resolveBookmarkData(Data(escaped.path.utf8)))
        }
    }
}
