import Foundation
import GGUFCore
import MLX
import MinirunKit
import StorageCore
import XCTest
@testable import ModelAdapters

final class Qwen4ExpDecoderParityTests: XCTestCase {
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

    func testCompleteSyntheticGraphMatchesPinnedLlamaCPPLogits() throws {
        try checkCompleteGraph(mixed: false)
    }

    func testCompleteMixedQuantizedGraphMatchesPinnedLlamaCPPLogits() throws {
        try checkCompleteGraph(mixed: true)
    }

    func testPromptConsumptionKeepsRecurrentStateAndLastLogitsWhileSkippingUnusedOutputHeads() throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["GGUF_HEADER_FIXTURE"] ?? env["TEST_RUNNER_GGUF_HEADER_FIXTURE"] else {
            throw XCTSkip("Run the synthetic reference harness first")
        }
        let folder = URL(fileURLWithPath: path).deletingLastPathComponent()
        let manifest = try JSONDecoder().decode(Manifest.self,
            from: Data(contentsOf: folder.appendingPathComponent("synthetic-qwen4exp-mixed.json")))
        let artifact = try VerifiedGGUFArtifact.verify(insideRoot: folder,
            manifest: GGUFModelManifest(fileName: manifest.fileName, fileByteCount: manifest.fileByteCount, sha256: manifest.sha256))
        let full = try Qwen4ExpDecoder(artifact: artifact, contextTokens: manifest.contextTokens,
            budgetBytes: 5_500_000_000, rawCacheBytes: 2 << 20)
        let lean = try Qwen4ExpDecoder(artifact: artifact, contextTokens: manifest.contextTokens,
            budgetBytes: 5_500_000_000, rawCacheBytes: 2 << 20, residentCacheBytes: 4 << 20)
        let prefix = Array(manifest.tokens.prefix(3))
        for token in prefix { _ = try full.logits(for: token); try lean.consume(token) }
        XCTAssertEqual(full.position, lean.position)
        let head = try XCTUnwrap(artifact.index.tensors["output.weight"])
        let savedBytes = full.projectionStatistics.logicalWeightBytes - lean.projectionStatistics.logicalWeightBytes
        XCTAssertGreaterThanOrEqual(savedBytes, UInt64(prefix.count) * head.byteCount)
        for token in manifest.tokens.dropFirst(prefix.count) {
            let expected = try full.logits(for: token).asArray(Float.self)
            let actual = try lean.logits(for: token).asArray(Float.self)
            for (a,b) in zip(expected,actual) { XCTAssertEqual(a,b,accuracy:1e-7) }
        }
        XCTAssertGreaterThan(lean.residentCacheStatistics.hits, 0)
        XCTAssertLessThanOrEqual(lean.residentCacheStatistics.residentPayloadBytes, 4 << 20)
        print("GGUF_PROMPT_HEAD_ELISION prefix=\(prefix.count) savedLogicalWeightBytes=\(savedBytes) followingLogitsUnchanged=true")
    }

    private func checkCompleteGraph(mixed: Bool) throws {
        let environment = ProcessInfo.processInfo.environment
        guard let header = environment["GGUF_HEADER_FIXTURE"] ?? environment["TEST_RUNNER_GGUF_HEADER_FIXTURE"] else {
            throw XCTSkip("Set GGUF_HEADER_FIXTURE and run the synthetic reference harness first")
        }
        let folder = URL(fileURLWithPath: header).deletingLastPathComponent()
        let manifestURL = folder.appendingPathComponent(mixed ? "synthetic-qwen4exp-mixed.json" : "synthetic-qwen4exp.json")
        let referenceURL = folder.appendingPathComponent(mixed ? "llama-mixed-reference-logits.json" : "llama-reference-logits.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path),
              FileManager.default.fileExists(atPath: referenceURL.path) else {
            throw XCTSkip("Generate tiny model and pinned CPU reference; trained weights are unnecessary")
        }
        let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: manifestURL))
        let reference = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: referenceURL))
        XCTAssertEqual(reference.tokens, manifest.tokens)
        XCTAssertEqual(reference.logits.count, manifest.tokens.count)
        let expected = try GGUFModelManifest(fileName: manifest.fileName,
            fileByteCount: manifest.fileByteCount, sha256: manifest.sha256)
        let artifact = try VerifiedGGUFArtifact.verify(insideRoot: folder, manifest: expected)
        if mixed {
            XCTAssertEqual(Set(artifact.index.tensors.values.map { $0.quantization.rawValue }),
                           Set(GGUFQuantization.allCases.map { $0.rawValue }))
        }
        let decoder = try Qwen4ExpDecoder(artifact: artifact, contextTokens: manifest.contextTokens,
            budgetBytes: 5_500_000_000, rawCacheBytes: 2 << 20)
        var checkpoints: [String: [Float]] = [:]
        var normalLogits: [[Float]] = []
        let roundingProbe: GGUFWeightSource?
        if mixed { roundingProbe = try GGUFWeightSource(artifact: artifact, rawCacheBytes: 2 << 20) }
        else { roundingProbe = nil }
        defer { roundingProbe?.shutdown() }
        for (position, token) in manifest.tokens.enumerated() {
            let actual = try decoder.logits(for: token).asArray(Float.self)
            normalLogits.append(actual)
            let target = reference.logits[position]
            XCTAssertEqual(actual.count, target.count)
            XCTAssertTrue(target.contains { abs($0) > 1e-6 }, "Zero logits are not an informative fixture")
            var maximum: Float = 0, errorSquared = 0.0, referenceSquared = 0.0
            for (a, b) in zip(actual, target) {
                maximum = max(maximum, abs(a - b))
                errorSquared += pow(Double(a) - Double(b), 2)
                referenceSquared += pow(Double(b), 2)
            }
            let relative = sqrt(errorSquared / max(referenceSquared, 1e-20))
            print("QWEN4EXP_LOGIT_PARITY mixed=\(mixed) position=\(position) maxAbs=\(maximum) relativeL2=\(relative)")
            XCTAssertLessThanOrEqual(maximum, 0.01)
            XCTAssertLessThanOrEqual(relative, 1e-3)
        }
        XCTAssertEqual(decoder.position, manifest.tokens.count)
        if mixed {
            XCTAssertGreaterThan(decoder.projectionStatistics.activationRoundingCalls, 0)
            XCTAssertGreaterThan(decoder.projectionStatistics.activationRoundingSeconds, 0)
        } else {
            XCTAssertEqual(decoder.projectionStatistics.activationRoundingCalls, 0)
        }
        print("GGUF_FULL_GRAPH_PROBE mixed=\(mixed) " + String(decoding:
            try JSONEncoder().encode(decoder.projectionStatistics), as: UTF8.self))
        if mixed {
            decoder.reset()
            #if DEBUG
            decoder.onParityCheckpoint = { checkpoints[$0] = $1 }
            #endif
            for (position, token) in manifest.tokens.enumerated() {
                checkpoints.removeAll(keepingCapacity: true)
                let shadow = try decoder.logits(for: token).asArray(Float.self)
                for (a, b) in zip(shadow, normalLogits[position]) {
                    XCTAssertEqual(a, b, accuracy: 1e-7, "Checkpoint observer changed streamed logits")
                }
                let referenceCheckpoints = try XCTUnwrap(reference.checkpoints?[position])
                for name in ["l_last-0", "l_last-1", "l_last-2", "l_last-3", "result_norm"] {
                    let cpu = try XCTUnwrap(referenceCheckpoints[name], "Missing CPU checkpoint \(name)")
                    let gpu = try XCTUnwrap(checkpoints[name], "Missing streamed checkpoint \(name)")
                    XCTAssertEqual(cpu.count, gpu.count)
                    let maxError = zip(cpu, gpu).map { pair in abs(pair.0 - pair.1) }.max() ?? 0
                    let squared = zip(cpu, gpu).reduce(0.0) { $0 + pow(Double($1.0) - Double($1.1), 2) }
                    let norm = cpu.reduce(0.0) { $0 + Double($1) * Double($1) }
                    print("REAM_CHECKPOINT position=\(position) name=\(name) maxAbs=\(maxError) relativeL2=\(sqrt(squared / max(norm, 1e-20)))")
                    if name == "result_norm", let probe = roundingProbe {
                        let type = try XCTUnwrap(artifact.index.tensors["output.weight"]).quantization
                        let a = try probe.canonicalActivations(MLXArray(cpu, [1, cpu.count]), type: type, columns: cpu.count).asArray(Float.self)
                        let b = try probe.canonicalActivations(MLXArray(gpu, [1, gpu.count]), type: type, columns: gpu.count).asArray(Float.self)
                        let differences = a.indices.filter { abs(a[$0] - b[$0]) > 1e-7 }
                        print("REAM_ACTIVATION_BOUNDARIES position=\(position) changedElements=\(differences.count) first=\(Array(differences.prefix(8))) rawPairs=\(differences.prefix(8).map { [cpu[$0], gpu[$0], a[$0], b[$0]] })")
                    }
                }
            }
            #if DEBUG
            decoder.onParityCheckpoint = nil
            #endif
        }
        decoder.reset()
        let firstAgain = try decoder.logits(for: manifest.tokens[0]).asArray(Float.self)
        for (a, b) in zip(firstAgain, reference.logits[0]) { XCTAssertEqual(a, b, accuracy: 0.01) }
        decoder.reset()
        XCTAssertThrowsError(try decoder.logits(for: 1, cancellationRequested: { true }))
        XCTAssertThrowsError(try decoder.logits(for: 1), "Partial state requires explicit reset")
        decoder.reset()
        _ = try decoder.logits(for: 1)
        XCTAssertEqual(decoder.position, 1)

        // Lose observability at the first expert checkpoint, after the GDN
        // state/history have changed. Entry and the layer checkpoint succeed.
        // Further steps must remain refused until explicit reset.
        decoder.reset()
        var observations = 0
        decoder.footprintSnapshot = {
            observations += 1
            return observations < 3 ? ProcessFootprint.current() : nil
        }
        XCTAssertThrowsError(try decoder.logits(for: 1)) { error in
            XCTAssertTrue(error.localizedDescription.contains("physical footprint unavailable"))
        }
        XCTAssertEqual(observations, 3)
        XCTAssertEqual(decoder.position, 0)
        decoder.footprintSnapshot = { ProcessFootprint.current() }
        XCTAssertThrowsError(try decoder.logits(for: 1), "Restoring telemetry does not repair partial state")
        decoder.reset()
        let recovered = try decoder.logits(for: manifest.tokens[0]).asArray(Float.self)
        for (a, b) in zip(recovered, reference.logits[0]) { XCTAssertEqual(a, b, accuracy: 0.01) }

        // Reject both the configured process limit and live OS headroom before
        // touching weights. These observations exercise admission, not allocation.
        for reading in [
            ProcessFootprint(footprintBytes: 5_500_000_000, peakFootprintBytes: 5_500_000_000, availableBytes: nil),
            ProcessFootprint(footprintBytes: 1, peakFootprintBytes: 1, availableBytes: 1)
        ] {
            decoder.reset()
            decoder.footprintSnapshot = { reading }
            let bytesBefore = decoder.readStatistics
            XCTAssertThrowsError(try decoder.logits(for: 1)) { error in
                XCTAssertTrue(error.localizedDescription.contains("live process memory admission"))
            }
            XCTAssertEqual(decoder.readStatistics.bytesRead, bytesBefore.bytesRead)
            XCTAssertEqual(decoder.position, 0)
        }
    }
}
