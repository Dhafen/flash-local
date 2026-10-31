import Foundation

public struct FlintRouteDecision: Codable, Equatable, Sendable {
    public let position, layer: Int
    public let experts: [Int32]
}

/// Only modes backed by a complete numerical admission may be used to serve.
public enum FlintServingMode: String, Codable, Sendable {
    case standard
    case singleToken
    case singleTokenF32KV
}

public struct FlintNumericalCheck: Codable, Sendable {
    public struct RouteDifference: Codable, Equatable, Sendable {
        public let position, layer: Int
        public let referenceExperts, candidateExperts: [Int32]?
    }
    public struct RouteDiagnostics: Codable, Equatable, Sendable {
        public let referenceCount, candidateCount, mismatchCount: Int
        public let firstMismatch: RouteDifference?
    }
    public struct Position: Codable, Sendable {
        public let index: Int
        public let maximumAbsoluteError, rmsError, cosineSimilarity: Double
        public let sameTop1: Bool
        public var passed: Bool {
            maximumAbsoluteError <= 0.1 && rmsError <= 0.02 && cosineSimilarity >= 0.9999 && sameTop1
        }
    }
    public struct MatchedMetalControl: Codable, Sendable {
        /// Metal with F32 KV storage, Flash Attention off, and CPU-sized batches.
        /// This is diagnostic evidence only and never grants serving admission.
        public let positions: [Position]
        public let exactExpertSets: Bool
        public let routeDiagnostics: RouteDiagnostics?
    }
    public struct Q4F32Reference: Codable, Sendable {
        /// Complete original Metal candidate vs. CPU Q4_1/F32 activation
        /// reference, accumulated in double. No weight or route changes.
        public let positions: [Position]
        public let routedTokenLayers: Int
        public let exactExpertSets: Bool
        public let routeDiagnostics: RouteDiagnostics?
        /// Compare the callback-free serving graph with the traced candidate.
        /// A missing or failed comparison cannot admit the new reference path.
        public let servingPositions: [Position]?
        /// The callback-free graph must also meet the original thresholds
        /// directly against the CPU reference, not just the traced graph.
        public let servingReferencePositions: [Position]?
        init(positions: [Position], routedTokenLayers: Int, exactExpertSets: Bool,
             routeDiagnostics: RouteDiagnostics?, servingPositions: [Position]? = nil,
             servingReferencePositions: [Position]? = nil) {
            self.positions = positions; self.routedTokenLayers = routedTokenLayers
            self.exactExpertSets = exactExpertSets; self.routeDiagnostics = routeDiagnostics
            self.servingPositions = servingPositions
            self.servingReferencePositions = servingReferencePositions
        }
        var referencePassed: Bool {
            guard let routes = routeDiagnostics, routes.referenceCount == routedTokenLayers,
                  routes.candidateCount == routedTokenLayers, routes.mismatchCount == 0,
                  routes.firstMismatch == nil else { return false }
            return FlintNumericalCheck.validControl(positions, routedTokenLayers, exactExpertSets)
        }
        var passed: Bool {
            guard referencePassed, let servingPositions, let servingReferencePositions else { return false }
            return servingPositions.map(\.index) == positions.map(\.index) &&
                servingReferencePositions.map(\.index) == positions.map(\.index) &&
                FlintNumericalCheck.validControl(servingPositions, routedTokenLayers, exactExpertSets) &&
                FlintNumericalCheck.validControl(servingReferencePositions, routedTokenLayers, exactExpertSets)
        }
    }
    public struct MetalFallback: Codable, Sendable {
        public enum Outcome: String, Codable, Sendable { case compared, unavailable }
        public let mode: FlintServingMode
        public let outcome: Outcome
        public let comparison: Q4F32Reference?
        init(mode: FlintServingMode, outcome: Outcome, comparison: Q4F32Reference?) {
            self.mode = mode; self.outcome = outcome; self.comparison = comparison
        }
        var passed: Bool { outcome == .compared && comparison?.passed == true }
    }
    public let positions: [Position]
    public let routedTokenLayers: Int
    public let exactExpertSets: Bool
    /// Absent for older saved admissions or malformed/oversized traces.
    public let routeDiagnostics: RouteDiagnostics?
    public let matchedMetalControl: MatchedMetalControl?
    public let q4F32Reference: Q4F32Reference?
    public let metalFallbacks: [MetalFallback]?
    init(positions: [Position], routedTokenLayers: Int, exactExpertSets: Bool,
         routeDiagnostics: RouteDiagnostics?, matchedMetalControl: MatchedMetalControl? = nil,
         q4F32Reference: Q4F32Reference? = nil, metalFallbacks: [MetalFallback]? = nil) {
        self.positions = positions; self.routedTokenLayers = routedTokenLayers
        self.exactExpertSets = exactExpertSets; self.routeDiagnostics = routeDiagnostics
        self.matchedMetalControl = matchedMetalControl; self.q4F32Reference = q4F32Reference
        self.metalFallbacks = metalFallbacks
    }
    private static func validControl(_ positions: [Position], _ routedTokenLayers: Int, _ exactExpertSets: Bool) -> Bool {
        guard positions.count == 4, let last = positions.last?.index, (7..<63).contains(last) else { return false }
        let tokenCount = last + 1
        return positions.map(\.index) == [0, 3, tokenCount / 2, tokenCount - 1] &&
            positions.allSatisfy(\.passed) && routedTokenLayers == tokenCount * 46 + 4 && exactExpertSets
    }
    public var servingMode: FlintServingMode? {
        if Self.validControl(positions, routedTokenLayers, exactExpertSets) || q4F32Reference?.passed == true {
            return .standard
        }
        let order: [FlintServingMode] = [.singleToken, .singleTokenF32KV]
        guard let candidates = metalFallbacks, !candidates.isEmpty, candidates.count <= order.count,
              candidates.map(\.mode) == Array(order.prefix(candidates.count)) else { return nil }
        return candidates.first(where: \.passed)?.mode
    }
    public var passed: Bool { servingMode != nil }
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
    private static func canonicalRoutes(_ values: [FlintRouteDecision]) -> [FlintRouteDecision]? {
        guard values.count <= 32768 else { return nil }
        var seen = Set<String>()
        for value in values {
            guard value.position >= 0, (2..<49).contains(value.layer), value.experts.count == 6,
                  Set(value.experts).count == 6, value.experts.allSatisfy({ (0..<256).contains($0) }),
                  seen.insert("\(value.position):\(value.layer)").inserted else { return nil }
        }
        return values.map { .init(position: $0.position, layer: $0.layer, experts: $0.experts.sorted()) }
            .sorted { $0.position == $1.position ? $0.layer < $1.layer : $0.position < $1.position }
    }
    public static func sameRoutes(_ reference: [FlintRouteDecision], _ candidate: [FlintRouteDecision]) -> Bool {
        guard !reference.isEmpty, let a = canonicalRoutes(reference), let b = canonicalRoutes(candidate) else { return false }
        return a == b
    }
    public static func diagnoseRoutes(_ reference: [FlintRouteDecision], _ candidate: [FlintRouteDecision]) -> RouteDiagnostics? {
        guard !reference.isEmpty, let a = canonicalRoutes(reference), let b = canonicalRoutes(candidate) else { return nil }
        var i = 0, j = 0, mismatches = 0
        var first: RouteDifference?
        while i < a.count || j < b.count {
            let left = i < a.count ? a[i] : nil, right = j < b.count ? b[j] : nil
            let leftKey = left.map { ($0.position, $0.layer) }, rightKey = right.map { ($0.position, $0.layer) }
            if let l = left, let r = right, leftKey!.0 == rightKey!.0, leftKey!.1 == rightKey!.1 {
                if l.experts != r.experts {
                    mismatches += 1
                    if first == nil { first = .init(position: l.position, layer: l.layer, referenceExperts: l.experts, candidateExperts: r.experts) }
                }
                i += 1; j += 1
            } else if let l = left, right == nil || leftKey! < rightKey! {
                mismatches += 1
                if first == nil { first = .init(position: l.position, layer: l.layer, referenceExperts: l.experts, candidateExperts: nil) }
                i += 1
            } else if let r = right {
                mismatches += 1
                if first == nil { first = .init(position: r.position, layer: r.layer, referenceExperts: nil, candidateExperts: r.experts) }
                j += 1
            }
        }
        return .init(referenceCount: a.count, candidateCount: b.count, mismatchCount: mismatches, firstMismatch: first)
    }
}
