import Foundation

public struct QPackInstallPlan: Sendable {
    public struct Section: Codable, Sendable {
        public let name: String
        public let dtype: String
        public let shape: [Int]
        public let offset: Int
        public let size: Int
    }
    public struct Layout: Codable, Sendable {
        public let expertCount: Int
        public let layerCount: Int
        public let expertStride: Int
        public let sections: [Section]
        public let linearLayers: [Bool]
    }
    public struct Span: Sendable {
        public enum Target: Sendable {
            case dense(offset: Int)
            case expert(layer: Int, sectionOffset: Int, perExpertBytes: Int)
        }
        public let range: Range<Int>
        public let target: Target
    }
    public struct Piece: Equatable, Sendable {
        public let sourceRange: Range<Int>
        public let file: String
        public let destinationOffset: Int
    }

    public let source: MLXSourceIndex
    public let layout: Layout
    public let layoutData: Data
    public let densePrefix: Data
    public let spansByShard: [[Span]]
    public let fileBytes: [String: Int]
    public let densePayloadBytes: Int
    public let expertPayloadBytes: Int
    public let finalWeightFileBytes: Int

    public init(source: MLXSourceIndex) throws {
        let g = source.geometry
        var sections: [Section] = []
        var sectionByName: [String: Section] = [:]
        var stride = 0
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            for part in ["weight", "scales", "biases"] {
                let suffix = proj + "." + part
                let first = "model.layers.0.mlp.switch_mlp." + suffix
                guard let tensor = source.tensors[first], tensor.shape.first == g.experts,
                      tensor.byteCount % g.experts == 0 else {
                    throw OrnithPerformanceError.unsupportedTensor(first)
                }
                for layer in 0..<g.layers {
                    let name = "model.layers.\(layer).mlp.switch_mlp." + suffix
                    guard let peer = source.tensors[name], peer.shape == tensor.shape,
                          peer.dtype == tensor.dtype, peer.byteCount == tensor.byteCount else {
                        throw OrnithPerformanceError.unsupportedTensor(name + " differs from fixed expert layout")
                    }
                }
                let section = Section(name: suffix, dtype: tensor.dtype,
                    shape: Array(tensor.shape.dropFirst()), offset: stride, size: tensor.byteCount / g.experts)
                sections.append(section)
                sectionByName[suffix] = section
                stride = try checkedAdd(stride, section.size)
            }
        }
        stride = try checkedMultiply(checkedAdd(stride, 16_383) / 16_384, 16_384)
        let layout = Layout(expertCount: g.experts, layerCount: g.layers,
            expertStride: stride, sections: sections, linearLayers: (0..<g.layers).map(g.isLinear))
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let layoutData = try encoder.encode(layout)
        var denseHeader: [String: Any] = [:]
        var offsets: [String: Int] = [:]
        var denseBytes = 0
        var expertBytes = 0
        for (name, t) in source.tensors.sorted(by: { $0.key < $1.key }) {
            if name.contains(".mlp.switch_mlp.") {
                expertBytes = try checkedAdd(expertBytes, t.byteCount)
                continue
            }
            let end = try checkedAdd(denseBytes, t.byteCount)
            denseHeader[name] = ["dtype": t.dtype, "shape": t.shape, "data_offsets": [denseBytes, end]]
            offsets[name] = denseBytes
            denseBytes = end
        }
        let raw = try JSONSerialization.data(withJSONObject: denseHeader, options: [.sortedKeys])
        var prefix = Data()
        var length = UInt64(raw.count).littleEndian
        withUnsafeBytes(of: &length) { prefix.append(contentsOf: $0) }
        prefix.append(raw)
        let denseSize = try checkedAdd(prefix.count, denseBytes)
        var files = ["model.safetensors": denseSize]
        let layerBytes = try checkedMultiply(stride, g.experts)
        for layer in 0..<g.layers { files[Self.layerFile(layer)] = layerBytes }
        var spans = Array(repeating: [Span](), count: source.manifest.shards.count)
        for (name, tensor) in source.tensors {
            let target: Span.Target
            if name.contains(".mlp.switch_mlp.") {
                let parts = name.split(separator: ".")
                guard parts.count == 7, let layer = Int(parts[2]), (0..<g.layers).contains(layer),
                      let section = sectionByName[parts.suffix(2).joined(separator: ".")] else {
                    throw OrnithPerformanceError.unsupportedTensor(name)
                }
                target = .expert(layer: layer, sectionOffset: section.offset, perExpertBytes: section.size)
            } else {
                guard let offset = offsets[name] else { throw OrnithPerformanceError.unsupportedTensor(name) }
                target = .dense(offset: try checkedAdd(prefix.count, offset))
            }
            spans[tensor.sourceShard].append(Span(range: tensor.sourceRange, target: target))
        }
        self.source = source
        self.layout = layout
        self.layoutData = layoutData
        self.densePrefix = prefix
        self.spansByShard = spans.map { $0.sorted { $0.range.lowerBound < $1.range.lowerBound } }
        self.fileBytes = files
        self.densePayloadBytes = denseBytes
        self.expertPayloadBytes = expertBytes
        self.finalWeightFileBytes = try files.values.reduce(0) { try checkedAdd($0, $1) }
    }

    /// Routes one bounded source window, including windows splitting tensor
    /// sections or experts. Source headers are public pinned bytes, not files.
    public func pieces(shard: Int, range: Range<Int>) throws -> [Piece] {
        guard spansByShard.indices.contains(shard), range.lowerBound >= 0,
              range.upperBound <= source.manifest.shards[shard].byteCount else {
            throw OrnithPerformanceError.invalidSource("invalid source window")
        }
        var pieces: [Piece] = []
        for span in spansByShard[shard] {
            let lo = max(range.lowerBound, span.range.lowerBound)
            let hi = min(range.upperBound, span.range.upperBound)
            guard lo < hi else { continue }
            switch span.target {
            case .dense(let offset):
                pieces.append(Piece(sourceRange: lo..<hi, file: "model.safetensors",
                    destinationOffset: try checkedAdd(offset, lo - span.range.lowerBound)))
            case .expert(let layer, let section, let perExpert):
                var cursor = lo
                while cursor < hi {
                    let relative = cursor - span.range.lowerBound
                    let expert = relative / perExpert
                    let within = relative % perExpert
                    let end = min(hi, try checkedAdd(cursor, perExpert - within))
                    let destination = try checkedAdd(checkedMultiply(expert, layout.expertStride),
                                                     checkedAdd(section, within))
                    pieces.append(Piece(sourceRange: cursor..<end, file: Self.layerFile(layer),
                                        destinationOffset: destination))
                    cursor = end
                }
            }
        }
        return pieces
    }

    static func layerFile(_ layer: Int) -> String { String(format: "packed_experts/layer_%02d.bin", layer) }
}
