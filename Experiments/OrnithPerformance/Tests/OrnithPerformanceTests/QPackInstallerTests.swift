import Foundation
import SwiftletCore
import XCTest
@testable import OrnithPerformance

final class QPackInstallerTests: XCTestCase {
    func testDirectInstallPreservesEveryExpertAndEightBitRouterWithoutRawShardCopy() async throws {
        let fixture = try SyntheticMLX()
        let plan = try fixture.plan()
        let root = temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let source = FixtureByteSource(fixture)
        let installer = try QPackInstaller(plan: plan, privateRoot: root, source: source, windowBytes: 4093)
        let result = try await installer.install()
        let reader = try QpackExpertReader(containerDir: result.directory)
        for layer in 0..<4 {
            for expert in 0..<8 {
                var blob = Data(count: plan.layout.expertStride)
                try blob.withUnsafeMutableBytes { try reader.readExpert(layer: layer, expert: expert, into: $0.baseAddress!) }
                for section in plan.layout.sections {
                    let name = "language_model.model.layers.\(layer).mlp.switch_mlp." + section.name
                    let original = try XCTUnwrap(fixture.tensors[name])
                    XCTAssertEqual(blob.subdata(in: section.offset..<(section.offset + section.size)),
                                   original.subdata(in: (expert * section.size)..<((expert + 1) * section.size)))
                }
            }
        }
        let checkpoint = try Checkpoint(dir: result.directory)
        for layer in 0..<4 {
            let name = "model.layers.\(layer).mlp.gate"
            XCTAssertEqual(checkpoint.quantSpec(for: name)?.bits, 8)
            XCTAssertEqual(try checkpoint.rawTensor(name + ".weight").bytes,
                           fixture.tensors["language_model." + name + ".weight"])
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: result.directory.appendingPathComponent("model-00001-of-00001.safetensors").path))
        let requests = await source.requests
        XCTAssertTrue(requests.compactMap(\.range).allSatisfy { $0.count <= 4093 })
        _ = try await installer.verifyInstalled(at: result.directory)
        print("ORNITH_QPACK_INSTALL exactExpertBytes=true mixedRouterBitsPreserved=true sourceSHA256Reconstructed=true rawShardCopy=false syntheticOnly=true")
    }

    func testInterruptedInstallResumesFromConvertedBytesAndDoesNotRedownloadPrefix() async throws {
        let fixture = try SyntheticMLX()
        let plan = try fixture.plan()
        let root = temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let prefixWindows = plan.source.sourceHeaders[0].count / 4093 + 5
        let first = try QPackInstaller(plan: plan, privateRoot: root,
            source: FixtureByteSource(fixture, failAfterRanges: prefixWindows), windowBytes: 4093)
        do { _ = try await first.install(); XCTFail("interruption expected") }
        catch FixtureByteSource.Failure.interrupted { }
        let firstFinal = await first.finalDirectory
        XCTAssertFalse(FileManager.default.fileExists(atPath: firstFinal.path))
        let resumedSource = FixtureByteSource(fixture)
        let resumed = try QPackInstaller(plan: plan, privateRoot: root, source: resumedSource, windowBytes: 4093)
        let result = try await resumed.install()
        let requests = await resumedSource.requests.compactMap(\.range)
        XCTAssertEqual(requests.first?.lowerBound, prefixWindows * 4093)
        XCTAssertGreaterThan(prefixWindows * 4093, plan.source.sourceHeaders[0].count)
        XCTAssertEqual(result.verifiedSourceBytes, fixture.manifest.shards[0].byteCount)
    }

    func testTamperedConvertedFileIsRejectedEvenAfterUpdatingTheLocalReceiptHash() async throws {
        let fixture = try SyntheticMLX()
        let root = temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let installer = try QPackInstaller(plan: fixture.plan(), privateRoot: root, source: FixtureByteSource(fixture))
        let result = try await installer.install()
        let file = result.directory.appendingPathComponent("packed_experts/layer_00.bin")
        var bytes = try Data(contentsOf: file)
        bytes[0] ^= 1
        try bytes.write(to: file)
        let receiptURL = result.directory.appendingPathComponent("flash-local-integrity.json")
        var receipt = try JSONSerialization.jsonObject(with: Data(contentsOf: receiptURL)) as! [String: Any]
        var hashes = receipt["fileSHA256"] as! [String: String]
        hashes["packed_experts/layer_00.bin"] = digest(bytes)
        receipt["fileSHA256"] = hashes
        try JSONSerialization.data(withJSONObject: receipt).write(to: receiptURL)
        do { _ = try await installer.verifyInstalled(at: result.directory); XCTFail("source mismatch expected") }
        catch let error as OrnithPerformanceError {
            XCTAssertEqual(error, .integrity("converted source bytes differ: model-00001-of-00001.safetensors"))
        }
    }

    func testPackedMixedPrecisionGraphMatchesCPUAndPrefillPlacementControls() async throws {
        let fixture = try SyntheticMLX()
        let root = temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let installer = try QPackInstaller(plan: fixture.plan(), privateRoot: root, source: FixtureByteSource(fixture))
        let verified = try await installer.install()
        let raw = root.appendingPathComponent("synthetic-reference")
        try FileManager.default.createDirectory(at: raw, withIntermediateDirectories: true)
        try fixture.config.write(to: raw.appendingPathComponent("config.json"))
        try fixture.files["model.safetensors"]!.write(to: raw.appendingPathComponent("model.safetensors"))
        let cpu = try QwenCPUModel(modelDir: raw)
        let cpuState = QwenCPUModel.DecodeState()
        let gpu = try QwenMetalModel(modelDir: verified.directory, cacheBudgetGB: 0.002, retainCPUKVMirror: false)
        let gpuState = QwenCPUModel.DecodeState()
        gpu.prefillMode = .layerMajor(chunkTokens: 4)
        var maxDifference: Float = 0
        for tokens in [[1, 5, 9, 42, 7], [23], [88], [101]] {
            let expected = try cpu.step(tokens, state: cpuState)
            let actual = try gpu.step(tokens, state: gpuState)
            XCTAssertEqual(expected.count, actual.count)
            let difference = zip(expected, actual).map { abs($0 - $1) }.max() ?? 0
            maxDifference = max(maxDifference, difference)
            XCTAssertLessThan(difference, 0.0005)
            XCTAssertEqual(expected.indices.max(by: { expected[$0] < expected[$1] }),
                           actual.indices.max(by: { actual[$0] < actual[$1] }))
        }
        XCTAssertEqual(gpuState.position, cpuState.position)
        print("ORNITH_QPACK_GRAPH mixedAffineG64=true routerBits=8 expertBits=4 maxAbs=\(maxDifference) sameGreedy=true syntheticOnly=true")
    }

    func temporary() -> URL { FileManager.default.temporaryDirectory.appendingPathComponent("ornith-test-\(UUID().uuidString)") }
}
