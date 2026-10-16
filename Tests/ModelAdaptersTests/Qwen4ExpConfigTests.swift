import Foundation
import XCTest
import GGUFCore
@testable import ModelAdapters

final class Qwen4ExpConfigTests: XCTestCase {
    private func index(ratios: [UInt64] = Array(repeating: 0, count: 48),
                       vocabularies: [UInt64]? = nil) throws -> GGUFIndex {
        var integers: [String: UInt64] = [
            "block_count": 48, "embedding_length": 2560, "expert_count": 308,
            "expert_used_count": 10, "full_attention_interval": 4,
            "attention.head_count_kv": 2, "attention.key_length": 256,
            "attention.value_length": 256, "ssm.time_step_rank": 48,
            "ssm.state_size": 128, "hyper_connection.count": 4,
            "context_length": 262144, "ple.eos_token_id": 248044,
            "ple.ngram_size": 3, "ple.heads_per_ngram": 8,
            "embedding_length_per_layer_input": 160,
        ]
        // These are the published artifact's compact head ranges, not the
        // unpruned checkpoint's head table.
        let arrays: [String: [UInt64]] = [
            "attention.compress_ratios": ratios,
            "ple.layers": [1],
            "ple.layer_multipliers": [23703573157769, 20109073645365, 8052911324071],
            "ple.head_offsets": [0, 20000003] + Array(repeating: 80000200, count: 6)
                + [40000026, 60000107] + Array(repeating: 80000200, count: 6),
            "ple.head_vocab_sizes": vocabularies ?? ([20000003, 20000023]
                + Array(repeating: 1, count: 6) + [20000081, 20000093]
                + Array(repeating: 1, count: 6)),
        ]
        var data = Data("GGUF".utf8)
        data.put(3, 4); data.put(1, 8); data.put(UInt64(1 + integers.count + arrays.count), 8)
        data.text("general.architecture"); data.put(8, 4); data.text("qwen4exp")
        for key in integers.keys.sorted() {
            data.text("qwen4exp." + key); data.put(10, 4); data.put(integers.removeValue(forKey: key)!, 8)
        }
        for key in arrays.keys.sorted() {
            let values = arrays[key]!
            data.text("qwen4exp." + key); data.put(9, 4); data.put(10, 4)
            data.put(UInt64(values.count), 8)
            for value in values { data.put(value, 8) }
        }
        data.text("per_layer_token_embd.weight"); data.put(2, 4)
        data.put(160, 8); data.put(80000201, 8); data.put(2, 4); data.put(0, 8)
        let payload = UInt64((data.count + 31) / 32 * 32)
        return try GGUFIndex.parse(header: data, fileByteCount: payload + 7_200_018_090)
    }

    func testPublishedArtifactExplicitlyUsesDenseAttention() throws {
        let config = try Qwen4ExpConfig(index: index())
        XCTAssertEqual(config.layers, 48)
        XCTAssertEqual(config.recurrentLayers, 36)
        XCTAssertEqual(config.denseAttentionLayers, 12)
        XCTAssertEqual(config.experts, 308)
        XCTAssertEqual(config.expertsPerToken, 10)
        XCTAssertEqual(config.pleLayers, [1])
        XCTAssertFalse(config.usesSparseAttention, "Do not silently repair published metadata")
    }

    func testSparseAttentionIsAnExplicitArtifactChoice() throws {
        var ratios = Array(repeating: UInt64(0), count: 48)
        for i in stride(from: 3, to: 48, by: 4) { ratios[i] = 4 }
        XCTAssertTrue(try Qwen4ExpConfig(index: index(ratios: ratios)).usesSparseAttention)
        ratios[0] = 4
        XCTAssertThrowsError(try Qwen4ExpConfig(index: index(ratios: ratios)))
        XCTAssertThrowsError(try Qwen4ExpConfig(index: index(ratios: [0])))
    }

    func testPLEReferenceHashAndSharedZeroRows() throws {
        let config = try Qwen4ExpConfig(index: index())
        let rows = try Qwen4ExpPLE.rows(current: 42, predecessors: [7, 9], config: config)
        // Independently evaluated uint64 multiply/XOR/modulo formula from
        // llama.cpp qwen4exp, pinned in docs/REAM-IMPLEMENTATION.md.
        XCTAssertEqual(rows, [14120497, 39300209, 80000200, 80000200, 80000200, 80000200,
            80000200, 80000200, 59148504, 64591832, 80000200, 80000200, 80000200,
            80000200, 80000200, 80000200])
    }

    func testPLEEOSCutsPredecessorsAndMissingHistoryMatchesEOS() throws {
        let config = try Qwen4ExpConfig(index: index())
        XCTAssertEqual(
            try Qwen4ExpPLE.rows(current: 42, predecessors: [], config: config),
            try Qwen4ExpPLE.rows(current: 42, predecessors: [config.pleEOS, 99], config: config))
        XCTAssertEqual(
            try Qwen4ExpPLE.rows(current: 42, predecessors: [7], config: config),
            try Qwen4ExpPLE.rows(current: 42, predecessors: [7, config.pleEOS], config: config))
        XCTAssertThrowsError(try Qwen4ExpPLE.rows(current: 42, predecessors: [1, 2, 3], config: config))
        // Force unsigned multiplication wraparound without trapping.
        XCTAssertEqual(try Qwen4ExpPLE.rows(current: UInt64.max, predecessors: [], config: config).count, 16)
    }

    func testInvalidPLETableRangesFailClosed() throws {
        XCTAssertThrowsError(try Qwen4ExpConfig(index: index(vocabularies: Array(repeating: UInt64.max, count: 16))))
        XCTAssertThrowsError(try Qwen4ExpConfig(index: index(vocabularies: Array(repeating: 0, count: 16))))
    }

    func testExactContextStateArithmeticAndCacheRemainder() throws {
        let config = try Qwen4ExpConfig(index: index())
        for (tokens, kv) in [(1024, UInt64(25_165_824)), (8192, 201_326_592),
                              (32768, 805_306_368), (65536, 1_610_612_736)] {
            let plan = try Qwen4ExpMemoryPlan(config: config, contextTokens: tokens,
                budgetBytes: 5_500_000_000, reservedWorkingBytes: 1_000_000_000)
            XCTAssertEqual(plan.kvBytes, kv)
            XCTAssertEqual(plan.recurrentMatrixBytes, 113_246_208)
            XCTAssertEqual(plan.cacheBytes + plan.kvBytes + plan.recurrentMatrixBytes
                + plan.reservedWorkingBytes, plan.budgetBytes)
        }
    }

    func testRejectsContextBudgetAndReserveViolations() throws {
        let config = try Qwen4ExpConfig(index: index())
        for (tokens, budget, reserve) in [(65536, UInt64(1_000_000_000), UInt64(600_000_000)),
            (0, 4_000_000_000, 600_000_000), (262145, 4_000_000_000, 600_000_000),
            (1024, 5_500_000_001, 600_000_000), (1024, 4_000_000_000, 0),
            (1024, 4_000_000_000, UInt64.max)] {
            XCTAssertThrowsError(try Qwen4ExpMemoryPlan(config: config, contextTokens: tokens,
                budgetBytes: budget, reservedWorkingBytes: reserve))
        }
    }
}

private extension Data {
    mutating func put(_ value: UInt64, _ count: Int) {
        for i in 0..<count { append(UInt8(truncatingIfNeeded: value >> (i * 8))) }
    }
    mutating func text(_ value: String) {
        let bytes = Data(value.utf8); put(UInt64(bytes.count), 8); append(bytes)
    }
}
