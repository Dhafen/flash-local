import Foundation
import GGUFCore

/// Hugging Face discovery is independent of inference admission. Community files
/// are downloaded with the same bounded, resumable SHA-256 verified GGUF transfer
/// as bundled models, but are NEVER handed to a runtime without its own checks.
public struct CommunityGGUFFile: Sendable, Identifiable, Codable, Equatable {
    public let repository: String
    public let revision: String
    public let path: String
    public let byteCount: UInt64
    public let sha256: String

    public var id: String { sha256 }
    public var name: String { path.split(separator: "/").last.map(String.init) ?? path }
    public var installedName: String { "community-" + sha256 + ".gguf" }

    public func pinnedSource() throws -> PinnedGGUFDownload {
        guard let url = Self.modelURL(repository: repository, revision: revision, path: path) else {
            throw GGUFError.invalid("invalid pinned Hugging Face model URL")
        }
        let manifest = try GGUFModelManifest(fileName: installedName, fileByteCount: byteCount, sha256: sha256)
        return try PinnedGGUFDownload(url: url, manifest: manifest)
    }

    static func modelURL(repository: String, revision: String, path: String) -> URL? {
        guard validRepository(repository), revision.count == 40,
              revision.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }),
              validModelPath(path) else { return nil }
        var url = URLComponents()
        url.scheme = "https"
        url.host = "huggingface.co"
        url.path = "/" + repository + "/resolve/" + revision + "/" + path
        return url.url
    }

    static func validRepository(_ repository: String) -> Bool {
        let parts = repository.split(separator: "/", omittingEmptySubsequences: false)
        return parts.count == 2 && parts.allSatisfy { part in
            (1...96).contains(part.utf8.count) && part != "." && part != ".." &&
            part.utf8.allSatisfy { byte in
                (48...57).contains(byte) || (65...90).contains(byte) || (97...122).contains(byte) ||
                    byte == 45 || byte == 46 || byte == 95
            }
        }
    }

    static func validModelPath(_ path: String) -> Bool {
        guard path.lowercased().hasSuffix(".gguf"), path.utf8.count <= 512 else { return false }
        let parts = path.split(separator: "/", omittingEmptySubsequences: false)
        return !parts.isEmpty && parts.allSatisfy {
            !$0.isEmpty && $0 != "." && $0 != ".." && !$0.contains("\\") &&
            !$0.contains("?") && !$0.contains("#") && !$0.contains("%") &&
            !$0.unicodeScalars.contains(where: { $0.value < 32 })
        }
    }
}

public enum CommunityGGUFCatalog {
    private struct Identity: Decodable { let sha: String }
    private struct TreeFile: Decodable {
        let type: String
        let path: String
        let size: UInt64?
        let lfs: LFS?
        struct LFS: Decodable { let oid: String; let size: UInt64? }
    }
    private static let maximumJSONBytes = 8 << 20

    /// Revision is obtained separately and every download URL pins that
    /// immutable commit; user-typed branch names never become download targets.
    public static func discover(repository: String, session: URLSession = .shared) async throws -> [CommunityGGUFFile] {
        guard CommunityGGUFFile.validRepository(repository) else {
            throw GGUFError.invalid("Ange ett Hugging Face-repo som ägare/modell")
        }
        let identityURL = URL(string: "https://huggingface.co/api/models/" + repository)!
        let identityData = try await checkedJSON(from: identityURL, session: session)
        let identity = try JSONDecoder().decode(Identity.self, from: identityData)
        let rev = identity.sha.lowercased()
        guard rev.utf8.count == 40, rev.utf8.allSatisfy({
            (48...57).contains($0) || (97...102).contains($0)
        }) else { throw GGUFError.invalid("untrusted model revision") }
        let treeURL = URL(string: "https://huggingface.co/api/models/" + repository + "/tree/" + rev + "?recursive=true")!
        let treeData = try await checkedJSON(from: treeURL, session: session)
        return try parseTree(treeData, repository: repository, revision: rev)
    }

    /// Exposed to unit tests: reject entries without LFS SHA-256. The ordinary
    /// Git blob hash is not a content digest for downloadable model payloads.
    public static func parseTree(_ data: Data, repository: String, revision: String) throws -> [CommunityGGUFFile] {
        guard CommunityGGUFFile.validRepository(repository), revision.utf8.count == 40,
              revision.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }),
              data.count <= maximumJSONBytes else { throw GGUFError.invalid("untrusted GGUF catalog") }
        let entries = try JSONDecoder().decode([TreeFile].self, from: data)
        guard entries.count <= 5000 else { throw GGUFError.limit("GGUF catalog exceeds safe entry limit") }
        var result: [CommunityGGUFFile] = []
        var paths = Set<String>()
        for entry in entries {
            guard entry.type == "file", CommunityGGUFFile.validModelPath(entry.path),
                  let lfs = entry.lfs, let bytes = lfs.size ?? entry.size,
                  bytes > 0 && bytes <= 80_000_000_000 else { continue }
            let digest = lfs.oid.lowercased()
            guard digest.utf8.count == 64, digest.utf8.allSatisfy({
                (48...57).contains($0) || (97...102).contains($0)
            }), paths.insert(entry.path).inserted else { continue }
            result.append(.init(repository: repository, revision: revision, path: entry.path,
                                byteCount: bytes, sha256: digest))
        }
        return result.sorted { $0.path.localizedStandardCompare($1.path) == .orderedAscending }
    }

    private static func checkedJSON(from url: URL, session: URLSession) async throws -> Data {
        var request = URLRequest(url: url)
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.timeoutInterval = 40
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200,
              http.url?.scheme == "https", http.url?.host == "huggingface.co",
              data.count <= maximumJSONBytes else {
            throw GGUFError.invalid("Hugging Face catalog unavailable or too large")
        }
        return data
    }

    /// Only successful, verified files appear in the installed library. This
    /// small JSON index does not grant inference or replace hashing on load.
    public static func installed(in root: URL) -> [CommunityGGUFFile] {
        let index = root.appendingPathComponent("community-index.json")
        guard let bytes = try? Data(contentsOf: index), bytes.count <= 1 << 20,
              let entries = try? JSONDecoder().decode([CommunityGGUFFile].self, from: bytes) else { return [] }
        return entries.filter { entry in
            (try? entry.pinnedSource()) != nil &&
            FileManager.default.fileExists(atPath: root.appendingPathComponent(entry.installedName).path)
        }
    }

    public static func recordVerified(_ entry: CommunityGGUFFile, in root: URL) throws {
        _ = try entry.pinnedSource()
        var entries = installed(in: root)
        entries.removeAll { $0.sha256 == entry.sha256 }
        guard entries.count < 128 else { throw GGUFError.limit("too many local community models") }
        entries.append(entry)
        let data = try JSONEncoder().encode(entries)
        try data.write(to: root.appendingPathComponent("community-index.json"), options: .atomic)
    }
}
