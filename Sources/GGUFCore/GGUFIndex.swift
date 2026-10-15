import Foundation
import StorageCore

public enum GGUFError: Error, Equatable, CustomStringConvertible, LocalizedError {
    case invalid(String)
    case unsupported(String)
    case truncated
    case limit(String)
    public var errorDescription: String? { description }

    public var description: String {
        switch self {
        case .invalid(let reason): return "Invalid GGUF: \(reason)"
        case .unsupported(let reason): return "Unsupported GGUF: \(reason)"
        case .truncated: return "GGUF header is truncated"
        case .limit(let reason): return "GGUF metadata budget exceeded: \(reason)"
        }
    }
}

/// GGML block encodings. These are not MLX affine encodings.
public enum GGUFQuantization: UInt32, Sendable, CaseIterable {
    case f32 = 0, f16 = 1, q4_0 = 2, q5_0 = 6, q8_0 = 8
    case q4_K = 12, q6_K = 14, bf16 = 30

    public var blockElements: UInt64 {
        switch self {
        case .f32, .f16, .bf16: return 1
        case .q4_0, .q5_0, .q8_0: return 32
        case .q4_K, .q6_K: return 256
        }
    }

    public var blockBytes: UInt64 {
        switch self {
        case .f32: return 4
        case .f16, .bf16: return 2
        case .q4_0: return 18
        case .q5_0: return 22
        case .q8_0: return 34
        case .q4_K: return 144
        case .q6_K: return 210
        }
    }
}

public indirect enum GGUFValue: Sendable, Equatable {
    case unsigned(UInt64), signed(Int64), number(Double), boolean(Bool), string(String)
    case array([GGUFValue])
    /// Vocabulary arrays remain in the bounded header; do not allocate hundreds
    /// of thousands of Swift objects merely to inspect tensor layout.
    case omittedArray(type: UInt32, count: UInt64)

    public var uint: UInt64? {
        switch self {
        case .unsigned(let value): return value
        case .signed(let value) where value >= 0: return UInt64(value)
        default: return nil
        }
    }

    public var text: String? { if case .string(let value) = self { return value }; return nil }
    public var elements: [GGUFValue]? { if case .array(let value) = self { return value }; return nil }
}

public struct GGUFParseLimits: Sendable {
    public var headerBytes = 32 * 1024 * 1024
    public var tensorCount: UInt64 = 100_000
    public var metadataCount: UInt64 = 10_000
    public var arrayCount: UInt64 = 1_000_000
    public var retainedArrayCount: UInt64 = 4096
    /// Only an explicit tokenizer load opts into these larger tables. Layout
    /// inspection continues to skip them. The aggregate bound also includes
    /// ordinary retained arrays, preventing many small tables bypassing it.
    public var largeArrayKeys: Set<String> = []
    public var retainedElementBudget: UInt64 = 1_000_000
    public var stringBytes: UInt64 = 2 * 1024 * 1024
    public init() {}
}

public struct GGUFTensor: Sendable, Equatable {
    public let name: String
    /// GGML order: reduction dimension first, expert dimension last.
    public let dimensions: [UInt64]
    public let quantization: GGUFQuantization
    public let relativeOffset: UInt64
    public let byteCount: UInt64
    public let rowBytes: UInt64

    /// A contiguous run of complete reduction rows. The optional expert index
    /// selects a slice, never the complete expert stack.
    public func region(
        sourceID: UInt64, regionID: UInt64, payloadOffset: UInt64,
        rows: Range<UInt64>, expert: UInt64? = nil
    ) throws -> RegionDescriptor {
        guard dimensions.count >= 2 else { throw GGUFError.invalid("row access needs a matrix") }
        guard rows.lowerBound < rows.upperBound, rows.upperBound <= dimensions[1] else {
            throw GGUFError.invalid("row range for \(name)")
        }
        let slice: UInt64
        if dimensions.count == 3, let expert, expert < dimensions[2] {
            slice = try checkedMultiply(expert, dimensions[1])
        } else if dimensions.count == 2, expert == nil { slice = 0 }
        else { throw GGUFError.invalid("expert selection for \(name)") }
        let offset = try checkedAdd(payloadOffset, relativeOffset)
        let rowOffset = try checkedMultiply(try checkedAdd(slice, rows.lowerBound), rowBytes)
        let length = try checkedMultiply(rows.upperBound - rows.lowerBound, rowBytes)
        guard let count = Int(exactly: length) else { throw GGUFError.limit("read length") }
        return RegionDescriptor(identity: .init(sourceID: sourceID, index: regionID),
                                offset: try checkedAdd(offset, rowOffset), length: count)
    }
}

public struct GGUFIndex: Sendable {
    public let metadata: [String: GGUFValue]
    public let tensors: [String: GGUFTensor]
    public let payloadOffset: UInt64
    public let headerByteCount: Int
    public let fileByteCount: UInt64

    /// Parse a bounded prefix. The caller supplies the size of the actual file,
    /// not the prefix length; no tensor payload is read or materialised here.
    public static func parse(
        header: Data, fileByteCount: UInt64, limits: GGUFParseLimits = .init()
    ) throws -> GGUFIndex {
        guard header.count <= limits.headerBytes else { throw GGUFError.limit("header bytes") }
        var reader = Reader(data: header, limits: limits)
        guard try reader.integer(4) == 0x46554747 else { throw GGUFError.invalid("magic") }
        guard try reader.integer(4) == 3 else { throw GGUFError.unsupported("version (expected 3)") }
        let tensorCount = try reader.integer(8), metadataCount = try reader.integer(8)
        guard tensorCount <= limits.tensorCount, metadataCount <= limits.metadataCount else {
            throw GGUFError.limit("entry count")
        }
        var metadata: [String: GGUFValue] = [:]
        for _ in 0..<metadataCount {
            let key = try reader.string()
            guard metadata[key] == nil else { throw GGUFError.invalid("duplicate key \(key)") }
            let type = UInt32(try reader.integer(4))
            metadata[key] = try reader.value(type: type, largeArray: limits.largeArrayKeys.contains(key))
        }
        let alignment: UInt64
        if let declared = metadata["general.alignment"] {
            guard let value = declared.uint else { throw GGUFError.invalid("alignment type") }
            alignment = value
        } else { alignment = 32 }
        guard alignment > 0, alignment <= 65_536, alignment & (alignment - 1) == 0 else {
            throw GGUFError.invalid("alignment")
        }
        var tensors: [String: GGUFTensor] = [:]
        for _ in 0..<tensorCount {
            let name = try reader.string(), rank = try reader.integer(4)
            guard !name.isEmpty, tensors[name] == nil, (1...4).contains(rank) else {
                throw GGUFError.invalid("tensor name or rank")
            }
            var dimensions: [UInt64] = []
            for _ in 0..<rank {
                let dimension = try reader.integer(8)
                guard dimension > 0 else { throw GGUFError.invalid("zero dimension") }
                dimensions.append(dimension)
            }
            let type = UInt32(try reader.integer(4))
            guard let quantization = GGUFQuantization(rawValue: type) else {
                throw GGUFError.unsupported("GGML tensor type \(type) in \(name)")
            }
            let offset = try reader.integer(8)
            guard offset % alignment == 0 else { throw GGUFError.invalid("tensor alignment") }
            guard dimensions[0] % quantization.blockElements == 0 else {
                throw GGUFError.invalid("quantization row alignment in \(name)")
            }
            let rowBytes = try checkedMultiply(
                dimensions[0] / quantization.blockElements, quantization.blockBytes)
            var bytes = rowBytes
            for dimension in dimensions.dropFirst() { bytes = try checkedMultiply(bytes, dimension) }
            tensors[name] = GGUFTensor(name: name, dimensions: dimensions, quantization: quantization,
                                      relativeOffset: offset, byteCount: bytes, rowBytes: rowBytes)
        }
        let end = try checkedAdd(UInt64(reader.position), alignment - 1)
        let payload = end / alignment * alignment
        guard payload <= fileByteCount else { throw GGUFError.truncated }
        var previousEnd: UInt64 = 0
        for tensor in tensors.values.sorted(by: { $0.relativeOffset < $1.relativeOffset }) {
            guard tensor.relativeOffset >= previousEnd else { throw GGUFError.invalid("overlapping tensors") }
            previousEnd = try checkedAdd(tensor.relativeOffset, tensor.byteCount)
            guard try checkedAdd(payload, previousEnd) <= fileByteCount else { throw GGUFError.truncated }
        }
        return GGUFIndex(metadata: metadata, tensors: tensors, payloadOffset: payload,
                         headerByteCount: reader.position, fileByteCount: fileByteCount)
    }
}

func checkedAdd(_ a: UInt64, _ b: UInt64) throws -> UInt64 {
    let result = a.addingReportingOverflow(b)
    guard !result.overflow else { throw GGUFError.invalid("integer overflow") }
    return result.partialValue
}

func checkedMultiply(_ a: UInt64, _ b: UInt64) throws -> UInt64 {
    let result = a.multipliedReportingOverflow(by: b)
    guard !result.overflow else { throw GGUFError.invalid("integer overflow") }
    return result.partialValue
}

private struct Reader {
    let data: Data
    let limits: GGUFParseLimits
    var position = 0
    var retainedElements: UInt64 = 0

    mutating func integer(_ bytes: Int) throws -> UInt64 {
        guard bytes <= data.count - position else { throw GGUFError.truncated }
        var result: UInt64 = 0
        for i in 0..<bytes { result |= UInt64(data[data.startIndex + position + i]) << (i * 8) }
        position += bytes
        return result
    }

    mutating func string(retain: Bool = true) throws -> String {
        let count = try integer(8)
        guard count <= limits.stringBytes else { throw GGUFError.limit("string length") }
        guard count <= UInt64(data.count - position) else { throw GGUFError.truncated }
        let end = position + Int(count)
        defer { position = end }
        guard retain else { return "" }
        let range = (data.startIndex + position)..<(data.startIndex + end)
        guard let string = String(data: data[range], encoding: .utf8) else {
            throw GGUFError.invalid("UTF-8")
        }
        return string
    }

    mutating func value(type: UInt32, retain: Bool = true, largeArray: Bool = false) throws -> GGUFValue {
        switch type {
        case 0: return .unsigned(try integer(1))
        case 1: return .signed(Int64(Int8(bitPattern: UInt8(try integer(1)))))
        case 2: return .unsigned(try integer(2))
        case 3: return .signed(Int64(Int16(bitPattern: UInt16(try integer(2)))))
        case 4: return .unsigned(try integer(4))
        case 5: return .signed(Int64(Int32(bitPattern: UInt32(try integer(4)))))
        case 6: return .number(Double(Float(bitPattern: UInt32(try integer(4)))))
        case 7:
            let byte = try integer(1)
            guard byte <= 1 else { throw GGUFError.invalid("boolean") }
            return .boolean(byte == 1)
        case 8: return .string(try string(retain: retain))
        case 9:
            let elementType = UInt32(try integer(4)), count = try integer(8)
            guard elementType != 9 else { throw GGUFError.unsupported("nested array") }
            guard count <= limits.arrayCount else { throw GGUFError.limit("array count") }
            // Validate even an empty array's element type.
            guard elementType <= 12 else { throw GGUFError.unsupported("metadata type \(elementType)") }
            let keep = retain && (largeArray || count <= limits.retainedArrayCount)
            if keep {
                guard retainedElements <= limits.retainedElementBudget,
                      count <= limits.retainedElementBudget - retainedElements else {
                    throw GGUFError.limit("aggregate retained array elements")
                }
                retainedElements += count
            }
            var result: [GGUFValue] = []
            for _ in 0..<count {
                let item = try value(type: elementType, retain: keep)
                if keep { result.append(item) }
            }
            return keep ? .array(result) : .omittedArray(type: elementType, count: count)
        case 10: return .unsigned(try integer(8))
        case 11: return .signed(Int64(bitPattern: try integer(8)))
        case 12: return .number(Double(bitPattern: try integer(8)))
        default: throw GGUFError.unsupported("metadata type \(type)")
        }
    }
}
