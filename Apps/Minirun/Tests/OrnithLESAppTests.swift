#if canImport(OrnithLES)
import Foundation
import XCTest
@testable import MinirunApp

@MainActor
final class OrnithLESAppTests: XCTestCase {
    func testMissingModelDoesNotAlterExistingModelsOrEnableAnotherOwner() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("les-app-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        let sentinel = root.appendingPathComponent("keep.txt")
        let original = Data("preserve existing model and chat".utf8); try original.write(to: sentinel)
        let app = OrnithLESAppModel(modelsRoot: root, privateRoot: root.appendingPathComponent("private"), documents: root)
        XCTAssertTrue(app.canStart); app.start(); XCTAssertFalse(app.canStart)
        await app.shutdown()
        XCTAssertFalse(app.isBusy); XCTAssertTrue(app.canStart)
        XCTAssertTrue(app.runs.isEmpty); XCTAssertNil(app.numericalCheck)
        XCTAssertEqual(try Data(contentsOf: sentinel), original)
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("private").path))
        print("ORNITH_LES_APP visibleEntry=true guardedOwnership=true previewCannotInfer=true trainedInference=false")
    }
    func testPreviewDoesNotPrepareWeightsOrStartAWorker() async {
        let model = OrnithLESAppModel.reviewPreview()
        XCTAssertFalse(model.canStart); model.start(); model.stop(); await model.shutdown()
        XCTAssertFalse(model.isBusy); XCTAssertTrue(model.runs.isEmpty)
        XCTAssertNil(model.measurementURL); XCTAssertNil(model.numericalCheck)
    }
}
#endif
