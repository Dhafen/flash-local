import Foundation
import Security

/// One device-local credential, separate from model files and preferences.
/// Security.framework serializes access; no key or request enters diagnostics.
public struct LocalAPIKeyStore: Sendable {
    private let service: String
    public init(service: String = "com.dhafen.flashlocal.local-api.v1") { self.service = service }
    private var query: [String: Any] {
        [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service,
         kSecAttrAccount as String: "local-server", kSecAttrSynchronizable as String: false]
    }
    public static func validate(_ key: String) throws {
        guard (32...128).contains(key.utf8.count), key.utf8.allSatisfy({ (33...126).contains($0) }) else {
            throw LocalAPIError.invalid("api_key", "Use 32–128 printable ASCII characters without spaces")
        }
    }
    public func load() throws -> String? {
        var request = query
        request[kSecReturnData as String] = true; request[kSecMatchLimit as String] = kSecMatchLimitOne
        var result: CFTypeRef?
        let status = SecItemCopyMatching(request as CFDictionary, &result)
        if status == errSecItemNotFound { return nil }
        guard status == errSecSuccess, let data = result as? Data,
              let key = String(data: data, encoding: .utf8) else { throw failure() }
        try Self.validate(key)
        return key
    }
    public func save(_ key: String) throws {
        try Self.validate(key)
        let changes: [String: Any] = [kSecValueData as String: Data(key.utf8)]
        let status = SecItemUpdate(query as CFDictionary, changes as CFDictionary)
        if status == errSecSuccess { return }
        guard status == errSecItemNotFound else { throw failure() }
        var item = query
        item[kSecValueData as String] = Data(key.utf8)
        item[kSecAttrAccessible as String] = kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        guard SecItemAdd(item as CFDictionary, nil) == errSecSuccess else { throw failure() }
    }
    public func loadOrCreate() throws -> String {
        if let existing = try load() { return existing }
        let key = try LocalModelAPIServer.makeAccessKey()
        try save(key)
        return key
    }
    public func delete() throws {
        let status = SecItemDelete(query as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else { throw failure() }
    }
    private func failure() -> LocalAPIError {
        .init(status: 500, message: "The device keychain is unavailable; unlock the device and try again")
    }
}
