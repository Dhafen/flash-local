import Darwin
import Foundation

public struct PerformanceDeviceSample: Codable, Equatable, Sendable {
    public let footprintBytes: UInt64
    public let processLifetimePeakBytes: UInt64
    public let availableProcessBytes: UInt64?
    public let thermalState: String
    /// iOS exposes a thermal state, not a public battery/SoC Celsius reading.
    public let temperatureCelsius: Double? = nil
    public let lowPowerMode: Bool

    public static func current() -> Self? {
        var info = task_vm_info_data_t()
        let expectedCount = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        var count = expectedCount
        let status = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard status == KERN_SUCCESS else { return nil }
        let complete = count >= expectedCount
        let footprint = UInt64(info.phys_footprint)
        let peak = complete ? UInt64(info.ledger_phys_footprint_peak) : footprint
        #if os(iOS)
        let available: UInt64? = UInt64(os_proc_available_memory())
        #else
        let available: UInt64? = nil
        #endif
        let thermal: String
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: thermal = "nominal"
        case .fair: thermal = "fair"
        case .serious: thermal = "serious"
        case .critical: thermal = "critical"
        @unknown default: thermal = "unknown"
        }
        return Self(footprintBytes: footprint, processLifetimePeakBytes: max(peak, footprint),
                    availableProcessBytes: available, thermalState: thermal,
                    lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled)
    }
}

public final class PerformanceCancellation: @unchecked Sendable {
    private let lock = NSLock()
    private var stopped = false
    public init() {}
    public func cancel() { lock.lock(); stopped = true; lock.unlock() }
    public var isCancelled: Bool { lock.lock(); defer { lock.unlock() }; return stopped }
}

public struct PerformanceSampledPeak: Codable, Sendable {
    public let maximumObservedFootprintBytes: UInt64
    public let minimumAvailableProcessBytes: UInt64?
    public let samples: Int
    public let thermalStatesObserved: [String]
}

final class PerformanceSampler: @unchecked Sendable {
    private let lock = NSLock()
    private var timer: DispatchSourceTimer?
    private var maximum: UInt64 = 0
    private var minimumAvailable: UInt64?
    private var samples = 0
    private var states = Set<String>()
    private var stopped = false
    init() {
        collect()
        let timer = DispatchSource.makeTimerSource(queue: DispatchQueue(label: "FlashLocal.Ornith.Footprint", qos: .utility))
        timer.schedule(deadline: .now() + .milliseconds(200), repeating: .milliseconds(200))
        timer.setEventHandler { [weak self] in self?.collect() }
        self.timer = timer
        timer.resume()
    }
    deinit { timer?.cancel() }
    private func collect() {
        guard let sample = PerformanceDeviceSample.current() else { return }
        lock.lock(); defer { lock.unlock() }
        guard !stopped else { return }
        samples += 1
        maximum = max(maximum, sample.footprintBytes)
        if let available = sample.availableProcessBytes {
            minimumAvailable = min(minimumAvailable ?? available, available)
        }
        states.insert(sample.thermalState)
    }
    func finish() -> PerformanceSampledPeak {
        timer?.cancel()
        collect()
        lock.lock(); defer { lock.unlock() }
        stopped = true
        return .init(maximumObservedFootprintBytes: maximum, minimumAvailableProcessBytes: minimumAvailable,
                     samples: samples, thermalStatesObserved: states.sorted())
    }
}
