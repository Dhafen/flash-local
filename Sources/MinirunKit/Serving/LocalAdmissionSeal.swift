import Foundation
import CryptoKit

/// Device-key-authenticated numerical admission, stored only in private app
/// support. Reports from Files are never admissions. The caller binds every
/// identity component and revalidates model/prepared integrity before opening.
public enum LocalAdmissionSeal {
    private struct Record: Codable {
        let version: Int
        let identity: [String: String]
        let payload: Data
    }
    private struct Envelope: Codable { let record: Data; let authentication: Data }
    public static func seal(_ payload: Data, identity: [String: String], key: String) throws -> Data {
        try LocalAPIKeyStore.validate(key)
        guard payload.count <= 1 << 20 else { throw LocalAPIError.invalid("admission", "Admission is too large") }
        let record = try JSONEncoder().encode(Record(version: 1, identity: identity, payload: payload))
        let authentication = Data(HMAC<SHA256>.authenticationCode(for: record, using: SymmetricKey(data: Data(key.utf8))))
        return try JSONEncoder().encode(Envelope(record: record, authentication: authentication))
    }
    public static func open(_ data: Data, identity: [String: String], key: String) -> Data? {
        guard data.count <= 2 << 20, (try? LocalAPIKeyStore.validate(key)) != nil,
              let envelope = try? JSONDecoder().decode(Envelope.self, from: data),
              HMAC<SHA256>.isValidAuthenticationCode(envelope.authentication, authenticating: envelope.record,
                using: SymmetricKey(data: Data(key.utf8))),
              let record = try? JSONDecoder().decode(Record.self, from: envelope.record),
              record.version == 1, record.identity == identity, record.payload.count <= 1 << 20 else { return nil }
        return record.payload
    }
}
