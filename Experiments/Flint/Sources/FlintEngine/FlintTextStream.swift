import Foundation

/// UTF-8 pieces may split a scalar or a reasoning delimiter across tokens.
/// Never insert replacement characters into code or tool arguments. At most
/// three UTF-8 bytes and a delimiter prefix are withheld between events.
public struct FlintTextStream: Sendable {
    public struct Delta: Equatable, Sendable {
        public var content = "", reasoning = ""
        public init() {}
    }
    public enum Failure: Error { case invalidUTF8 }
    public private(set) var isThinking = false
    private var utf8: [UInt8] = [], expected = 0
    private var pending = ""
    public init() {}

    public mutating func append(_ bytes: [UInt8]) throws -> Delta {
        var decoded = ""
        for byte in bytes {
            if utf8.isEmpty {
                if byte < 0x80 { decoded.unicodeScalars.append(UnicodeScalar(byte)); continue }
                switch byte {
                case 0xc2...0xdf: expected = 2
                case 0xe0...0xef: expected = 3
                case 0xf0...0xf4: expected = 4
                default: throw Failure.invalidUTF8
                }
            } else if !(0x80...0xbf).contains(byte) { throw Failure.invalidUTF8 }
            utf8.append(byte)
            if utf8.count == expected {
                guard let text = String(bytes: utf8, encoding: .utf8) else { throw Failure.invalidUTF8 }
                decoded += text; utf8.removeAll(keepingCapacity: true); expected = 0
            }
        }
        pending += decoded
        return drain(final: false)
    }
    public mutating func finish() throws -> Delta {
        guard utf8.isEmpty else { throw Failure.invalidUTF8 }
        return drain(final: true)
    }
    private mutating func drain(final: Bool) -> Delta {
        var delta = Delta()
        func emit(_ text: String) {
            if isThinking { delta.reasoning += text } else { delta.content += text }
        }
        while !pending.isEmpty {
            let delimiter = isThinking ? "</think>" : "<think>"
            if let range = pending.range(of: delimiter) {
                emit(String(pending[..<range.lowerBound]))
                pending = String(pending[range.upperBound...]); isThinking.toggle()
                continue
            }
            var hold = 0
            if !final {
                for count in 1..<delimiter.utf8.count where pending.hasSuffix(String(delimiter.prefix(count))) { hold = count }
            }
            emit(hold == 0 ? pending : String(pending.dropLast(hold)))
            pending = hold == 0 ? "" : String(pending.suffix(hold))
            break
        }
        return delta
    }
}
