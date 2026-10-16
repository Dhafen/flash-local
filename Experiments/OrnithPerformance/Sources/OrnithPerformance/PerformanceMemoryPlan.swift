import Foundation

/// All values are bytes, not GiB. The upstream cacheBudgetGB API takes GiB;
/// that conversion is performed only at the engine boundary.
public struct PerformanceMemoryPlan: Codable, Equatable, Sendable {
    public struct Limits: Sendable {
        public var processCeilingBytes: UInt64 = 5_500_000_000
        public var preferredExpertCacheBytes: UInt64 = 2_000_000_000
        public var safetyHeadroomBytes: UInt64 = 256 << 20
        public var runtimeReserveBytes: UInt64 = 512 << 20
        public var prefillScratchReserveBytes: UInt64 = 32 << 20
        public init() {}
    }

    public let contextTokens: Int
    public let baselineFootprintBytes: UInt64
    public let effectiveProcessCeilingBytes: UInt64
    public let residentDensePayloadBytes: UInt64
    public let recurrentStateBytes: UInt64
    public let convolutionStateBytes: UInt64
    public let gpuKVCapacityBytes: UInt64
    public let cpuKVMirrorReserveBytes: UInt64
    public let kvGrowthTransientBytes: UInt64
    public let prefillScratchReserveBytes: UInt64
    public let runtimeReserveBytes: UInt64
    public let safetyHeadroomBytes: UInt64
    public let expertCacheBytes: UInt64
    public let minimumExpertCacheBytes: UInt64
    public let reservedTotalBytes: UInt64
    public let retainCPUKVMirror: Bool

    public static func make(
        install: QPackInstallPlan, contextTokens: Int,
        baselineFootprintBytes: UInt64, osAvailableBytes: UInt64?,
        metalRecommendedWorkingSetBytes: UInt64? = nil,
        retainCPUKVMirror: Bool = false, limits: Limits = .init()
    ) throws -> Self {
        let g = install.source.geometry
        guard contextTokens > 0, contextTokens <= g.maximumContext,
              limits.processCeilingBytes > 0 else {
            throw OrnithPerformanceError.invalidSource("invalid context/memory request")
        }
        // Swiftlet currently stores f32 K/V and doubles GPU capacity, with a
        // minimum of 256 rows. Price the allocated capacity, not used rows.
        var rows = 256
        while rows < contextTokens { rows = try checkedMultiply(rows, 2) }
        let rowBytes = try product(g.kvHeads, g.headDimension, 2, 4)
        let gpuKV = try product(g.attentionLayers, rows, rowBytes)
        // Swift arrays can reserve beyond their count; budget up to twice the
        // GPU capacity plus an old per-layer allocation during growth.
        let cpuMirror = retainCPUKVMirror ? try product(gpuKV, 2) : 0
        let growthRows = rows > 256 ? rows / 2 : 0
        let growth = try product(growthRows, rowBytes, retainCPUKVMirror ? 3 : 1)
        let recurrent = try product(g.recurrentLayers, g.linearValueHeads,
                                    g.linearKeyDimension, g.linearValueDimension, 4)
        let convDim = try checkedAdd(checkedMultiply(2 * g.linearKeyHeads, g.linearKeyDimension),
                                     checkedMultiply(g.linearValueHeads, g.linearValueDimension))
        let conv = try product(g.recurrentLayers, g.convolutionWidth - 1, convDim, 4)
        let minimumCache = try product(16, install.layout.expertStride)
        let dense = UInt64(install.densePayloadBytes)
        let processLimit: UInt64
        if let available = osAvailableBytes {
            processLimit = min(limits.processCeilingBytes, try sum(baselineFootprintBytes, available))
        } else {
            processLimit = limits.processCeilingBytes
        }
        let fixed = try sum(baselineFootprintBytes, dense, UInt64(recurrent), UInt64(conv),
            UInt64(gpuKV), UInt64(cpuMirror), UInt64(growth), limits.prefillScratchReserveBytes,
            limits.runtimeReserveBytes, limits.safetyHeadroomBytes)
        let needed = try sum(fixed, UInt64(minimumCache))
        guard needed <= processLimit else {
            throw OrnithPerformanceError.insufficientMemory(required: needed, available: processLimit)
        }
        var cache = min(limits.preferredExpertCacheBytes, processLimit - fixed)
        if let metalLimit = metalRecommendedWorkingSetBytes {
            let gpuFixed = try sum(dense, UInt64(recurrent), UInt64(conv), UInt64(gpuKV),
                                   UInt64(growth), limits.prefillScratchReserveBytes)
            guard gpuFixed <= metalLimit else {
                throw OrnithPerformanceError.insufficientMemory(required: gpuFixed, available: metalLimit)
            }
            cache = min(cache, metalLimit - gpuFixed)
        }
        // The pool's physical allocations can be page-rounded; the expert
        // stride is already 16 KiB-aligned and the upstream pool also checks
        // each MTLBuffer.allocatedSize before admitting it.
        cache = cache / UInt64(install.layout.expertStride) * UInt64(install.layout.expertStride)
        guard cache >= UInt64(minimumCache) else {
            throw OrnithPerformanceError.insufficientMemory(required: UInt64(minimumCache), available: cache)
        }
        return Self(contextTokens: contextTokens, baselineFootprintBytes: baselineFootprintBytes,
            effectiveProcessCeilingBytes: processLimit, residentDensePayloadBytes: dense,
            recurrentStateBytes: UInt64(recurrent), convolutionStateBytes: UInt64(conv),
            gpuKVCapacityBytes: UInt64(gpuKV), cpuKVMirrorReserveBytes: UInt64(cpuMirror),
            kvGrowthTransientBytes: UInt64(growth), prefillScratchReserveBytes: limits.prefillScratchReserveBytes,
            runtimeReserveBytes: limits.runtimeReserveBytes, safetyHeadroomBytes: limits.safetyHeadroomBytes,
            expertCacheBytes: cache, minimumExpertCacheBytes: UInt64(minimumCache),
            reservedTotalBytes: try sum(fixed, cache), retainCPUKVMirror: retainCPUKVMirror)
    }

    /// Additional cache allowance at a quiescent token boundary. Do not
    /// subtract resident weights twice: footprint already includes them.
    /// The caller may shrink but never grows the cache during a generation.
    public func safeCacheLimit(
        currentFootprintBytes: UInt64, currentCacheAllocatedBytes: UInt64,
        osAvailableBytes: UInt64?, futureStateGrowthBytes: UInt64
    ) -> UInt64 {
        let allowance = effectiveProcessCeilingBytes > currentFootprintBytes
            ? effectiveProcessCeilingBytes - currentFootprintBytes : 0
        let headroom = min(allowance, osAvailableBytes ?? allowance)
        let (reserve, overflow) = safetyHeadroomBytes.addingReportingOverflow(futureStateGrowthBytes)
        guard !overflow else { return 0 }
        let recoverable = currentCacheAllocatedBytes.addingReportingOverflow(headroom)
        guard !recoverable.overflow else { return 0 }
        return recoverable.partialValue > reserve
            ? min(expertCacheBytes, recoverable.partialValue - reserve) : 0
    }

    private static func product(_ values: Int...) throws -> Int {
        try values.reduce(1) { try checkedMultiply($0, $1) }
    }

    private static func sum(_ values: UInt64...) throws -> UInt64 {
        try values.reduce(0) {
            let (value, overflow) = $0.addingReportingOverflow($1)
            guard !overflow else { throw OrnithPerformanceError.arithmeticOverflow }
            return value
        }
    }
}
