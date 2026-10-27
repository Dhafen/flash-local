import Foundation

/// Prevents competing runtime families from changing omni's process-wide
/// streaming knobs while another family still owns a native context. Multiple
/// contexts within one family retain existing diagnostic/test behavior.
public final class LocalEngineLease: Sendable {
    public enum Family: Sendable { case ornith, flint }
    private let id: UUID
    private static let state = State()
    // Safety: family/owners are accessed only under this lock, including
    // release from deinit. Replace with a checked mutex on a future toolchain.
    private final class State: @unchecked Sendable {
        let lock = NSLock()
        var family: Family?
        var owners: Set<UUID> = []
    }
    private init(id: UUID) { self.id = id }
    public static func acquire(_ family: Family) throws -> LocalEngineLease {
        let s = state; s.lock.lock(); defer { s.lock.unlock() }
        guard s.family == nil || s.family == family else {
            throw LocalAPIError(status: 409, message: "Unload the other local model before changing inference engine")
        }
        let id = UUID(); s.family = family; s.owners.insert(id)
        return LocalEngineLease(id: id)
    }
    deinit {
        let s = Self.state; s.lock.lock(); defer { s.lock.unlock() }
        s.owners.remove(id); if s.owners.isEmpty { s.family = nil }
    }
}
