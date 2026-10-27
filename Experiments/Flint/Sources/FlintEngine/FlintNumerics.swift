import Foundation

public struct FlintRouteDecision: Codable, Equatable, Sendable {
    public let position, layer: Int
    public let experts: [Int32]
}

public struct FlintNumericalCheck: Codable, Sendable {
    public struct Position: Codable, Sendable {
        public let index: Int
        public let maximumAbsoluteError, rmsError, cosineSimilarity: Double
        public let sameTop1: Bool
        public var passed: Bool {
            maximumAbsoluteError <= 0.1 && rmsError <= 0.02 && cosineSimilarity >= 0.9999 && sameTop1
        }
    }
    public let positions: [Position]
    public let routedTokenLayers: Int
    public let exactExpertSets: Bool
    public var passed: Bool { positions.count == 4 && positions.allSatisfy(\.passed) && routedTokenLayers > 0 && exactExpertSets }
    public static func compare(position: Int, reference: [Float], candidate: [Float]) throws -> Position {
        guard !reference.isEmpty, reference.count == candidate.count, reference.count <= 300000,
              reference.allSatisfy(\.isFinite), candidate.allSatisfy(\.isFinite) else { throw ComparisonFailure.invalidLogits }
        var maximum = 0.0, squared = 0.0, cross = 0.0, left = 0.0, right = 0.0, a = 0, b = 0
        for i in reference.indices {
            let x = Double(reference[i]), y = Double(candidate[i]), error = x - y
            maximum = max(maximum, abs(error)); squared += error * error; cross += x * y; left += x * x; right += y * y
            if reference[i] > reference[a] { a = i }; if candidate[i] > candidate[b] { b = i }
        }
        let cosine = left > 0 && right > 0 ? cross / sqrt(left * right) : (reference == candidate ? 1 : 0)
        return .init(index: position, maximumAbsoluteError: maximum, rmsError: sqrt(squared / Double(reference.count)),
            cosineSimilarity: cosine, sameTop1: a == b)
    }
    public enum ComparisonFailure: Error { case invalidLogits }
    public static func sameRoutes(_ reference: [FlintRouteDecision], _ candidate: [FlintRouteDecision]) -> Bool {
        func canonical(_ values: [FlintRouteDecision]) -> [FlintRouteDecision]? {
            var seen = Set<String>()
            for value in values {
                guard value.position >= 0, (2..<49).contains(value.layer), value.experts.count == 6,
                      Set(value.experts).count == 6, value.experts.allSatisfy({ (0..<256).contains($0) }),
                      seen.insert("\(value.position):\(value.layer)").inserted else { return nil }
            }
            return values.map { .init(position: $0.position, layer: $0.layer, experts: $0.experts.sorted()) }
                .sorted { $0.position == $1.position ? $0.layer < $1.layer : $0.position < $1.position }
        }
        guard !reference.isEmpty, let a = canonical(reference), let b = canonical(candidate) else { return false }
        return a == b
    }
}
