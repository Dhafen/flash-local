/// Compact sliding-window KV cannot safely rewind to an arbitrary older
/// prefix: the preceding window may already have been overwritten. Reuse only
/// the entire exact cached prefix. A mismatch/shortening requires a reset.
public enum FlintPrefixPolicy {
    public static func reusableCount(cached: [Int32], prompt: [Int32]) -> Int {
        guard cached.count <= prompt.count, prompt.starts(with: cached) else { return 0 }
        return cached.count
    }
    public static func requiresReset(cached: [Int32], prompt: [Int32]) -> Bool {
        !cached.isEmpty && (cached.count > prompt.count || !prompt.starts(with: cached))
    }
}
