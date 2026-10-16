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
    func testCPUAlignedNormBoundsAndTransferBenchmark() throws {
        let columns = 2048
        let x = MLXArray((0..<(2 * columns)).map { Float(($0 * 17) % 109 - 54) / 32 }, [2, columns])
        let gamma = MLXArray((0..<columns).map { 0.8 + Float($0 % 13) / 32 })
        var aligned: [Double] = [], metal: [Double] = []
        for sample in -1..<3 {
            let start = MonotonicClock.now()
            let a = try Qwen35CPUAlignedNorm.rms(x, weight: gamma, eps: 1e-6).asArray(Float.self)
            let middle = MonotonicClock.now()
            let b = K3Norm.rms(x, weight: gamma, eps: 1e-6).asArray(Float.self)
            let end = MonotonicClock.now()
            for (actual, other) in zip(a, b) { XCTAssertEqual(actual, other, accuracy: 1e-6) }
            if sample >= 0 {
                aligned.append(MonotonicClock.seconds(from: start, to: middle))
                metal.append(MonotonicClock.seconds(from: middle, to: end))
            }
        }
        XCTAssertThrowsError(try Qwen35CPUAlignedNorm.rms(x, weight: MLXArray.ones([1]), eps: 1e-6))
        XCTAssertThrowsError(try Qwen35CPUAlignedNorm.rms(x, weight: gamma, eps: 0))
        XCTAssertThrowsError(try Qwen35CPUAlignedNorm.rms(MLXArray([Float.nan]), weight: MLXArray.ones([1]), eps: 1e-6))
        XCTAssertThrowsError(try Qwen35CPUAlignedNorm.rms(MLXArray.zeros([65537]), weight: MLXArray.ones([65537]), eps: 1e-6))
        let record: [String: Any] = ["syntheticOnly":true, "physicalPhone":false,
            "sourceRevision":BuildInfo.gitRevision, "os":ProcessInfo.processInfo.operatingSystemVersionString,
            "physicalMemoryBytes":ProcessInfo.processInfo.physicalMemory, "shape":[2,columns],
            "maximumElements":65536, "includesTransfers":true, "warmup":1,
            "samples":["cpuAlignedSeconds":aligned,"metalSeconds":metal]]
        print("ORNITH_NORM_BENCH " + String(decoding: try JSONSerialization.data(withJSONObject: record, options: .sortedKeys), as: UTF8.self))
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
            XCTAssertEqual(decoder.projectionStatistics.canonicalCPUKVCalls, 10)
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
                    let attentionInput = try XCTUnwrap(expected["attn_norm-3"])
                    let projectedValue = try probe.linearCanonicalKV(MLXArray(attentionInput, [1, attentionInput.count]),
                        tensor: "blk.3.attn_v.weight").asArray(Float.self)
                    let cpuValue = try XCTUnwrap(expected["Vcur-3"])
                    XCTAssertEqual(projectedValue.map(\.bitPattern), cpuValue.map(\.bitPattern), "Pinned small projection arithmetic")
                    if position == 0 {
                        let input = MLXArray(attentionInput, [1, attentionInput.count])
                        var canonical: [Double] = [], metal: [Double] = []
                        for sample in -1..<3 {
                            let start = MonotonicClock.now()
                            let a = try probe.linearCanonicalKV(input, tensor: "blk.3.attn_v.weight").asArray(Float.self)
                            let middle = MonotonicClock.now()
                            let b = try probe.linear(input, tensor: "blk.3.attn_v.weight").asArray(Float.self)
                            let end = MonotonicClock.now()
                            XCTAssertEqual(a.map(\.bitPattern), cpuValue.map(\.bitPattern))
                            for (actual, other) in zip(a, b) { XCTAssertEqual(actual, other, accuracy: 1e-6) }
                            if sample >= 0 {
                                canonical.append(MonotonicClock.seconds(from: start, to: middle))
                                metal.append(MonotonicClock.seconds(from: middle, to: end))
                            }
                        }
                        let record: [String: Any] = ["syntheticOnly":true,"physicalPhone":false,
                            "sourceRevision":BuildInfo.gitRevision,"os":ProcessInfo.processInfo.operatingSystemVersionString,
                            "physicalMemoryBytes":ProcessInfo.processInfo.physicalMemory,"shape":[8,256],
                            "maximumWeightBytes":4 << 20,"maximumOutputRows":1024,"warmup":1,
                            "includesFileIOAndCopies":true,"residentPayloadBytes":0,
                            "samples":["canonicalCPUSeconds":canonical,"packedMetalSeconds":metal]]
                        print("ORNITH_KV_PROJECTION_BENCH " + String(decoding: try JSONSerialization.data(withJSONObject: record, options: .sortedKeys), as: UTF8.self))
                    }
                    // Use the CPU's own observed input, not the streamed input,
                    // to isolate reduction/rounding from upstream graph error.
                    for layer in 0..<4 {
                        let source = layer == 0 ? "model.input_embed" : "l_out-\(layer - 1)"
                        for (inputName, normName, tensorName) in [
                            (source, "attn_norm-\(layer)", "blk.\(layer).attn_norm.weight"),
                            ("attn_residual-\(layer)", "attn_post_norm-\(layer)", "blk.\(layer).post_attention_norm.weight")
                        ] {
                            let input = try XCTUnwrap(expected[inputName]), target = try XCTUnwrap(expected[normName])
                            let normalized = try Qwen35CPUAlignedNorm.rms(MLXArray(input, [1, input.count]),
                                weight: probe.smallTensor(tensorName), eps: decoder.layout.epsilon).asArray(Float.self)
                            XCTAssertEqual(normalized.map(\.bitPattern), target.map(\.bitPattern), "CPU norm arithmetic at \(normName)")
                        }
                    }
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
                        if name.hasPrefix("Vcur-") {
                            let a = cpu.map { Float16($0).bitPattern }, b = gpu.map { Float16($0).bitPattern }
                            let changes = a.indices.filter { a[$0] != b[$0] }
                            print("ORNITH_KV_ROUNDING position=\(position) name=\(name) changedHalfElements=\(changes.count) rawPairs=\(changes.prefix(8).map { [cpu[$0], gpu[$0]] })")
                        }
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
    func testPairedAndSeparateGateUpMatchEverySyntheticGraphLogitBitForBit() throws {
        for mixed in [false, true] {
            let (artifact, manifest, _) = try artifact(mixed: mixed)
            let paired = try Qwen35MoEDecoder(artifact: artifact, contextTokens: manifest.contextTokens,
                budgetBytes: 5_500_000_000, rawCacheBytes: 2 << 20, residentCacheBytes: 16 << 20)
            let control = try Qwen35MoEDecoder(artifact: artifact, contextTokens: manifest.contextTokens,
                budgetBytes: 5_500_000_000, rawCacheBytes: 2 << 20, residentCacheBytes: 16 << 20,
                pairGateUpProjections: false)
            for token in manifest.tokens {
                let a = try paired.logits(for: token).asArray(Float.self)
                let b = try control.logits(for: token).asArray(Float.self)
                XCTAssertEqual(a.map(\.bitPattern), b.map(\.bitPattern))
            }
            XCTAssertGreaterThan(paired.projectionStatistics.pairedProjectionBatches ?? 0, 0)
            XCTAssertEqual(control.projectionStatistics.pairedProjectionBatches, 0)
            XCTAssertEqual(paired.projectionStatistics.calls, control.projectionStatistics.calls)
            XCTAssertEqual(paired.projectionStatistics.logicalWeightBytes, control.projectionStatistics.logicalWeightBytes)
            XCTAssertLessThan(paired.projectionStatistics.evaluationBatches ?? UInt64.max,
                              control.projectionStatistics.evaluationBatches ?? 0)
            XCTAssertLessThanOrEqual(paired.residentCacheStatistics.residentPayloadBytes, 16 << 20)
            print("ORNITH_PAIRED_GRAPH mixed=\(mixed) exactLogitBits=true logicalWorkUnchanged=true fewerEvaluations=true syntheticOnly=true")
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
