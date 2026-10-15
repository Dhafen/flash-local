import Foundation
import GGUFCore

/// The GPT-2 byte alphabet, Qwen35 pre-splitter and ranked-pair BPE carried by
/// the pinned GGUF. This is not a sentencepiece/tiktoken vocabulary. In
/// particular, Qwen35 does NOT enable llama.cpp's `ignore_merges` shortcut.
/// Ordinary text never interprets structural markup; the typed chat renderer
/// must insert its verified IDs explicitly. Unicode is not normalized.
public final class Qwen35Vocabulary: @unchecked Sendable {
    public enum Failure: Error, LocalizedError, Equatable {
        case invalid(String), limit(String), reservedText(String)
        public var errorDescription: String? {
            switch self { case .invalid(let s), .limit(let s), .reservedText(let s): return s }
        }
    }

    public static let metadataArrayKeys: Set<String> = [
        "tokenizer.ggml.tokens", "tokenizer.ggml.token_type", "tokenizer.ggml.merges",
    ]
    // Match the pinned reference's QWEN35 branch, including combining marks.
    static let splitPattern = #"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
    private struct Pair: Hashable { let left: String; let right: String }
    private struct Node { var text: String; var previous: Int; var next: Int; var version = 0; var alive = true }
    private struct Candidate { let rank: Int; let left: Int; let right: Int; let lv: Int; let rv: Int }
    private struct Heap {
        var values: [Candidate] = []
        static func before(_ a: Candidate, _ b: Candidate) -> Bool {
            a.rank == b.rank ? a.left < b.left : a.rank < b.rank
        }
        mutating func push(_ value: Candidate) {
            values.append(value); var i = values.count - 1
            while i > 0 {
                let p = (i - 1) / 2
                guard Self.before(values[i], values[p]) else { break }
                values.swapAt(i, p); i = p
            }
        }
        mutating func pop() -> Candidate? {
            guard !values.isEmpty else { return nil }
            if values.count == 1 { return values.removeLast() }
            let result = values[0]; values[0] = values.removeLast(); var i = 0
            while i * 2 + 1 < values.count {
                var c = i * 2 + 1
                if c + 1 < values.count, Self.before(values[c + 1], values[c]) { c += 1 }
                guard Self.before(values[c], values[i]) else { break }
                values.swapAt(i, c); i = c
            }
            return result
        }
    }

    private let tokens: [String]
    private let types: [UInt64]
    private let ids: [String: Int]
    private let ranks: [Pair: Int]
    private let reserved: [String]
    private let byteScalars: [Unicode.Scalar]
    private let scalarBytes: [Unicode.Scalar: UInt8]
    private let splitter: NSRegularExpression
    public let eos: Int
    public let endOfTurn: Int
    public let chatTemplate: String?
    public var vocabularySize: Int { tokens.count }

    /// `index` must come from verified metadata in production. Unit/reference
    /// tests may pass a hash-checked header without downloading tensor payloads.
    public init(index: GGUFIndex) throws {
        let metadata = index.metadata
        guard metadata["tokenizer.ggml.model"]?.text == "gpt2",
              metadata["tokenizer.ggml.pre"]?.text == "qwen35",
              let rawTokens = metadata["tokenizer.ggml.tokens"]?.elements,
              let rawTypes = metadata["tokenizer.ggml.token_type"]?.elements,
              let rawMerges = metadata["tokenizer.ggml.merges"]?.elements,
              rawTokens.count >= 256, rawTokens.count <= 300_000,
              rawTypes.count == rawTokens.count, rawMerges.count <= 300_000 else {
            throw Failure.invalid("Expected bounded Qwen35 GGUF vocabulary, token types and merge tables")
        }
        var tokens: [String] = [], types: [UInt64] = [], ids: [String: Int] = [:]
        tokens.reserveCapacity(rawTokens.count); types.reserveCapacity(rawTypes.count)
        ids.reserveCapacity(rawTokens.count)
        for (id, value) in rawTokens.enumerated() {
            guard let text = value.text, !text.isEmpty, text.utf8.count <= 4096,
                  let type = rawTypes[id].uint, (1...6).contains(type),
                  ids.updateValue(id, forKey: text) == nil else {
                throw Failure.invalid("Invalid/duplicate Qwen35 vocabulary entry at \(id)")
            }
            tokens.append(text); types.append(type)
        }
        let alphabet = Self.alphabet()
        let inverse = Dictionary(uniqueKeysWithValues: alphabet.enumerated().map { ($0.element, UInt8($0.offset)) })
        for scalar in alphabet {
            guard let id = ids[String(scalar)], types[id] == 1 else {
                throw Failure.invalid("Qwen35 vocabulary does not contain every normal byte token")
            }
        }
        for (id, text) in tokens.enumerated() where types[id] == 1 {
            guard text.unicodeScalars.allSatisfy({ inverse[$0] != nil }) else {
                throw Failure.invalid("Qwen35 normal token \(id) is outside the byte alphabet")
            }
        }
        var ranks: [Pair: Int] = [:]; ranks.reserveCapacity(rawMerges.count)
        for (rank, raw) in rawMerges.enumerated() {
            guard let merge = raw.text, let space = merge.firstIndex(of: " "),
                  space != merge.startIndex, merge.index(after: space) != merge.endIndex else {
                throw Failure.invalid("Invalid Qwen35 merge \(rank)")
            }
            let pair = Pair(left: String(merge[..<space]), right: String(merge[merge.index(after: space)...]))
            guard !pair.right.contains(" "), let left = ids[pair.left], let right = ids[pair.right],
                  let output = ids[pair.left + pair.right], types[left] == 1, types[right] == 1,
                  types[output] == 1, ranks.updateValue(rank, forKey: pair) == nil else {
                throw Failure.invalid("Duplicate or inconsistent Qwen35 merge \(rank)")
            }
        }
        func control(_ key: String) throws -> Int {
            guard let n = metadata[key]?.uint, n < UInt64(tokens.count), types[Int(n)] == 3 else {
                throw Failure.invalid("Missing control id \(key)")
            }
            return Int(n)
        }
        self.eos = try control("tokenizer.ggml.eos_token_id")
        // GGUF may omit eot_token_id; the pinned llama.cpp loader recognizes
        // <|im_end|> by spelling while reading Qwen's control vocabulary.
        guard let endOfTurn = ids["<|im_end|>"], types[endOfTurn] == 3 else {
            throw Failure.invalid("Missing Qwen35 end-of-turn control")
        }
        self.endOfTurn = endOfTurn
        if case .boolean(true) = metadata["tokenizer.ggml.add_bos_token"] {
            throw Failure.invalid("Automatic BOS insertion is unsupported for this Qwen35 artifact")
        }
        self.tokens = tokens; self.types = types; self.ids = ids; self.ranks = ranks
        self.chatTemplate = metadata["tokenizer.chat_template"]?.text
        self.reserved = tokens.enumerated().filter { types[$0.offset] != 1 }.map(\.element)
        self.byteScalars = alphabet; self.scalarBytes = inverse
        self.splitter = try NSRegularExpression(pattern: Self.splitPattern)
    }

    public func controlID(_ text: String) throws -> Int {
        guard let id = ids[text], types[id] == 3 || types[id] == 4 else {
            throw Failure.invalid("Qwen35 does not declare control \(text)")
        }
        return id
    }

    /// Bounded complete prompt encoding. No BOS/EOS is added implicitly.
    public func encode(_ text: String, maximumTokens: Int = 131_072) throws -> [Int] {
        guard text.utf8.count <= 262_144, maximumTokens > 0, maximumTokens <= 131_072 else {
            throw Failure.limit("Qwen35 text/token budget exceeded")
        }
        if let marker = reserved.first(where: { text.contains($0) }) {
            throw Failure.reservedText("Message contains reserved tokenizer markup: \(marker)")
        }
        let source = text as NSString
        var result: [Int] = [], cursor = 0
        for match in splitter.matches(in: text, range: NSRange(location: 0, length: source.length)) {
            guard match.range.location == cursor, match.range.length > 0 else {
                throw Failure.invalid("Qwen35 regex left text unmatched")
            }
            let piece = source.substring(with: match.range)
            guard piece.utf8.count <= 16_384 else { throw Failure.limit("Qwen35 pre-token exceeds 16 KiB") }
            let encoded = String(String.UnicodeScalarView(piece.utf8.map { byteScalars[Int($0)] }))
            result.append(contentsOf: try merge(encoded))
            guard result.count <= maximumTokens else { throw Failure.limit("Qwen35 token budget exceeded") }
            cursor += match.range.length
        }
        guard cursor == source.length else { throw Failure.invalid("Qwen35 regex did not cover input") }
        return result
    }

    /// Keep bytes across token boundaries; decoding each token into a String
    /// separately would corrupt split UTF-8 sequences. Control tokens are
    /// returned only when explicitly requested for diagnostics/reference.
    public func bytes(for id: Int, includeControl: Bool = false) throws -> Data {
        guard tokens.indices.contains(id) else { throw Failure.invalid("Token id outside vocabulary") }
        if types[id] == 3 || types[id] == 4 {
            return includeControl ? Data(tokens[id].utf8) : Data()
        }
        guard types[id] == 1 else { throw Failure.invalid("Unsupported output token type \(types[id])") }
        return Data(try tokens[id].unicodeScalars.map { scalar in
            guard let byte = scalarBytes[scalar] else { throw Failure.invalid("Invalid token byte alphabet") }
            return byte
        })
    }

    public func decode(_ ids: [Int], includeControl: Bool = false) throws -> String {
        guard ids.count <= 131_072 else { throw Failure.limit("Qwen35 decode token budget exceeded") }
        var data = Data()
        for id in ids { data.append(try bytes(for: id, includeControl: includeControl)) }
        return String(decoding: data, as: UTF8.self)
    }

    private func merge(_ piece: String) throws -> [Int] {
        var nodes = piece.unicodeScalars.enumerated().map { index, scalar in
            Node(text: String(scalar), previous: index - 1, next: index + 1)
        }
        guard !nodes.isEmpty else { return [] }
        nodes[nodes.count - 1].next = -1
        var heap = Heap()
        func candidate(_ left: Int, _ right: Int) -> Candidate? {
            guard left >= 0, right >= 0, nodes[left].alive, nodes[right].alive,
                  nodes[left].next == right, let rank = ranks[Pair(left: nodes[left].text, right: nodes[right].text)] else { return nil }
            return Candidate(rank: rank, left: left, right: right, lv: nodes[left].version, rv: nodes[right].version)
        }
        for left in 0..<(nodes.count - 1) {
            if let c = candidate(left, left + 1) { heap.push(c) }
        }
        while let c = heap.pop() {
            guard nodes[c.left].alive, nodes[c.right].alive, nodes[c.left].next == c.right,
                  nodes[c.left].version == c.lv, nodes[c.right].version == c.rv else { continue }
            let previous = nodes[c.left].previous, next = nodes[c.right].next
            nodes[c.left].text += nodes[c.right].text; nodes[c.left].next = next
            nodes[c.left].version += 1; nodes[c.right].alive = false
            if next >= 0 { nodes[next].previous = c.left }
            if let item = candidate(previous, c.left) { heap.push(item) }
            if let item = candidate(c.left, next) { heap.push(item) }
        }
        var output: [Int] = [], cursor = 0
        while cursor >= 0 {
            guard let id = ids[nodes[cursor].text], types[id] == 1 else {
                throw Failure.invalid("BPE result is not a normal vocabulary token")
            }
            output.append(id); cursor = nodes[cursor].next
        }
        return output
    }

    private static func alphabet() -> [Unicode.Scalar] {
        let direct = Array(33...126) + Array(161...172) + Array(174...255)
        var result = Array(repeating: Unicode.Scalar(0)!, count: 256), extra = 256
        for byte in 0...255 {
            if direct.contains(byte) { result[byte] = Unicode.Scalar(byte)! }
            else { result[byte] = Unicode.Scalar(extra)!; extra += 1 }
        }
        return result
    }
}
