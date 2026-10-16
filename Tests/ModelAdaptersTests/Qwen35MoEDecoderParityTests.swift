import Foundation
import GGUFCore
import MLX
import MinirunKit
import StorageCore
import XCTest
@testable import ModelAdapters

final class Qwen35MoEDecoderParityTests: XCTestCase {
    private struct Manifest: Decodable {
        let fileName: String
        let fileByteCount: UInt64
        let sha256: String
        let tokens: [UInt64]
        let contextTokens: Int
    }
    private struct Reference: Decodable {
        let tokens: [UInt64]
        let logits: [[Float]]
        let checkpoints: [[String: [Float]]]?
    }
    private func folder() throws -> URL {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["GGUF_HEADER_FIXTURE"] ?? env["TEST_RUNNER_GGUF_HEADER_FIXTURE"] else {
            throw XCTSkip("Run the pinned synthetic CPU reference first")
        }
        return URL(fileURLWithPath: path).deletingLastPathComponent()
    }
    private func artifact(mixed: Bool) throws -> (VerifiedGGUFArtifact, Manifest, Reference) {
        let folder = try folder(), suffix = mixed ? "-mixed" : ""
        let manifest = try JSONDecoder().decode(Manifest.self,
            from: Data(contentsOf: folder.appendingPathComponent("synthetic-ornith\(suffix).json")))
        XCTAssertLessThan(manifest.fileByteCount, 32 << 20, "Only generated random weights may run in CI")
        let reference = try JSONDecoder().decode(Reference.self,
            from: Data(contentsOf: folder.appendingPathComponent("llama-ornith\(suffix)-reference-logits.json")))
        let artifact = try VerifiedGGUFArtifact.verify(insideRoot: folder,
            manifest: GGUFModelManifest(fileName: manifest.fileName, fileByteCount: manifest.fileByteCount, sha256: manifest.sha256))
        return (artifact, manifest, reference)
    }
    func testOrnithFP32AndMixedGraphsMatchPinnedCPUReference() throws {
        for mixed in [false, true] {
            let (artifact, manifest, reference) = try artifact(mixed: mixed)
            let decoder = try Qwen35MoEDecoder(artifact: artifact, contextTokens: manifest.contextTokens,
                budgetBytes: 5_500_000_000, rawCacheBytes: 2 << 20)
            XCTAssertEqual(decoder.layout.layers, 4); XCTAssertEqual(decoder.layout.mtpLayers, 1)
            XCTAssertEqual(reference.tokens, manifest.tokens)
            XCTAssertEqual(reference.logits.count, manifest.tokens.count)
            var normalLogits: [[Float]] = []
            for (position, token) in manifest.tokens.enumerated() {
                let actual = try decoder.logits(for: token).asArray(Float.self), expected = reference.logits[position]
                normalLogits.append(actual)
                XCTAssertEqual(actual.count, expected.count)
                XCTAssertTrue(expected.contains { abs($0) > 1e-6 })
                let maximum = zip(actual, expected).map { abs($0.0 - $0.1) }.max() ?? .infinity
                let squared = zip(actual, expected).reduce(0.0) { $0 + pow(Double($1.0) - Double($1.1), 2) }
                let norm = expected.reduce(0.0) { $0 + Double($1) * Double($1) }
                let relative = sqrt(squared / max(norm, 1e-20))
                print("ORNITH_LOGIT_PARITY mixed=\(mixed) position=\(position) maxAbs=\(maximum) relativeL2=\(relative) syntheticOnly=true")
                XCTAssertLessThanOrEqual(maximum, 0.01)
                XCTAssertLessThanOrEqual(relative, 1e-3)
            }
            XCTAssertEqual(decoder.position, 5)
            XCTAssertGreaterThan(decoder.readStatistics.bytesRead, 0)
            if mixed {
                let probe = try GGUFWeightSource(artifact: artifact, rawCacheBytes: 2 << 20)
                defer { probe.shutdown() }
                decoder.reset()
                var observed: [String: [Float]] = [:]
                #if DEBUG
                decoder.onParityCheckpoint = { observed[$0] = $1 }
                #endif
                for (position, token) in manifest.tokens.enumerated() {
                    observed.removeAll(keepingCapacity: true)
                    let shadow = try decoder.logits(for: token).asArray(Float.self)
                    for (a, b) in zip(shadow, normalLogits[position]) {
                        XCTAssertEqual(a, b, accuracy: 1e-7, "Checkpoint observer changed normal logits")
                    }
                    let expected = try XCTUnwrap(reference.checkpoints?[position])
                    let required = ["model.input_embed", "l_out-0", "l_out-1", "l_out-2", "l_out-3", "result_norm"]
                    for name in required { XCTAssertNotNil(expected[name]); XCTAssertNotNil(observed[name]) }
                    for name in observed.keys.sorted() {
                        let gpu = try XCTUnwrap(observed[name])
                        let cpu = try XCTUnwrap(expected[name], "Missing CPU checkpoint \(name)")
                        XCTAssertEqual(cpu.count, gpu.count)
                        let maximum = zip(cpu, gpu).map { abs($0.0 - $0.1) }.max() ?? 0
                        let squared = zip(cpu, gpu).reduce(0.0) { $0 + pow(Double($1.0) - Double($1.1), 2) }
                        let norm = cpu.reduce(0.0) { $0 + Double($1) * Double($1) }
                        print("ORNITH_CHECKPOINT position=\(position) name=\(name) maxAbs=\(maximum) relativeL2=\(sqrt(squared / max(norm, 1e-20)))")
                        let tensorName: String?
                        if name == "result_norm" { tensorName = "output.weight" }
                        else if name.hasPrefix("attn_norm-"), let layer = Int(name.dropFirst("attn_norm-".count)) {
                            tensorName = "blk.\(layer).\(layer == 3 ? "attn_q.weight" : "attn_qkv.weight")"
                        } else if name.hasPrefix("attn_post_norm-"), let layer = Int(name.dropFirst("attn_post_norm-".count)) {
                            tensorName = "blk.\(layer).ffn_gate_exps.weight"
                        } else if name.hasPrefix("final_output-"), let layer = Int(name.dropFirst("final_output-".count)) {
                            tensorName = "blk.\(layer).ssm_out.weight"
                        } else { tensorName = nil }
                        if let tensorName {
                            let type = try XCTUnwrap(artifact.index.tensors[tensorName]).quantization
                            let a = try probe.canonicalActivations(MLXArray(cpu, [1, cpu.count]), type: type, columns: cpu.count).asArray(Float.self)
                            let b = try probe.canonicalActivations(MLXArray(gpu, [1, gpu.count]), type: type, columns: gpu.count).asArray(Float.self)
                            let differences = a.indices.filter { abs(a[$0] - b[$0]) > 1e-7 }
                            print("ORNITH_ACTIVATION_BOUNDARIES position=\(position) name=\(name) changedElements=\(differences.count) first=\(Array(differences.prefix(8))) rawPairs=\(differences.prefix(8).map { [cpu[$0], gpu[$0], a[$0], b[$0]] })")
                        }
                    }
                }
                #if DEBUG
                decoder.onParityCheckpoint = nil
                #endif
            }
        }
    }
    func testOrnithCheckpointHeadElisionAndFailedStepRecovery() throws {
        let (artifact, manifest, _) = try artifact(mixed: true)
        let full = try Qwen35MoEDecoder(artifact: artifact, contextTokens: manifest.contextTokens,
            budgetBytes: 5_500_000_000, rawCacheBytes: 2 << 20)
        let lean = try Qwen35MoEDecoder(artifact: artifact, contextTokens: manifest.contextTokens,
            budgetBytes: 5_500_000_000, rawCacheBytes: 2 << 20, residentCacheBytes: 16 << 20)
        for token in manifest.tokens.prefix(3) { _ = try full.logits(for: token); try lean.consume(token) }
        let checkpoint = try XCTUnwrap(lean.checkpoint())
        XCTAssertGreaterThan(checkpoint.payloadBytes, 0)
        XCTAssertNil(try lean.checkpoint(maximumBytes: 0))
        XCTAssertThrowsError(try full.restore(checkpoint))
        var following: [[Float]] = []
        for token in manifest.tokens.dropFirst(3) {
            let expected = try full.logits(for: token).asArray(Float.self)
            let actual = try lean.logits(for: token).asArray(Float.self)
            following.append(actual)
            for (a, b) in zip(expected, actual) { XCTAssertEqual(a, b, accuracy: 1e-7) }
        }
        XCTAssertLessThan(lean.projectionStatistics.logicalWeightBytes, full.projectionStatistics.logicalWeightBytes)
        XCTAssertGreaterThan(lean.residentCacheStatistics.evictableHits ?? 0, 0)
        XCTAssertGreaterThan(lean.residentCacheStatistics.evictablePayloadBytes ?? 0, 0)
        XCTAssertLessThanOrEqual(lean.residentCacheStatistics.residentPayloadBytes, 16 << 20)
        XCTAssertThrowsError(try lean.logits(for: 1, cancellationRequested: { true }))
        XCTAssertThrowsError(try lean.logits(for: 1))
        try lean.restore(checkpoint)
        for (token, expected) in zip(manifest.tokens.dropFirst(3), following) {
            let actual = try lean.logits(for: token).asArray(Float.self)
            for (a, b) in zip(expected, actual) { XCTAssertEqual(a, b, accuracy: 1e-7) }
        }
        lean.reset()
        let ioBefore = lean.readStatistics.bytesRead
        lean.footprintSnapshot = { nil }
        XCTAssertThrowsError(try lean.logits(for: 1))
        XCTAssertEqual(lean.readStatistics.bytesRead, ioBefore)
        lean.footprintSnapshot = { ProcessFootprint.current() }
        XCTAssertThrowsError(try lean.logits(for: 1), "Lost telemetry must not silently repair failed state")
        lean.reset(); _ = try lean.logits(for: 1)
        XCTAssertEqual(lean.position, 1)
    }
}
