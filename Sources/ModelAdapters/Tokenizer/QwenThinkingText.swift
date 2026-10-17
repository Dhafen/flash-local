import Foundation

/// Splits a Qwen-generated reasoning channel from its answer. The opening tag
/// can already belong to the canonical prompt. Incomplete control markers stay
/// buffered across output pieces; model text is never replaced with a summary.
public struct QwenThinkingText: Sendable {
    public private(set) var reasoning = ""
    public private(set) var answer = ""
    public private(set) var isThinking: Bool
    private var pending = ""
    private static let markers = ["<think>", "</think>"]

    public init(startsInThinking: Bool = false) { isThinking = startsInThinking }

    public mutating func append(_ text: String) {
        pending += text
        while !pending.isEmpty {
            let next = Self.markers.compactMap { marker -> (String, Range<String.Index>)? in
                pending.range(of: marker).map { (marker, $0) }
            }.min { $0.1.lowerBound < $1.1.lowerBound }
            if let (marker, range) = next {
                emit(String(pending[..<range.lowerBound]))
                pending = String(pending[range.upperBound...])
                isThinking = marker == "<think>"
            } else {
                var held = min(pending.count, 7)
                while held > 0 && !Self.markers.contains(where: { $0.hasPrefix(String(pending.suffix(held))) }) {
                    held -= 1
                }
                let boundary = pending.index(pending.endIndex, offsetBy: -held)
                emit(String(pending[..<boundary])); pending = String(pending[boundary...])
                break
            }
        }
    }

    public mutating func finish() { emit(pending); pending = "" }
    private mutating func emit(_ text: String) {
        if isThinking { reasoning += text } else { answer += text }
    }
}
