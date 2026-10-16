import Foundation

public enum LocalModelStorage {
    public static func prepare(root: URL) throws {
        let manager = FileManager.default
        // Refuse a symlink even if it currently points at a directory. The
        // internal model location must remain on the app's own local volume.
        if let values = try? root.resourceValues(forKeys: [.isSymbolicLinkKey]), values.isSymbolicLink == true {
            throw CocoaError(.fileWriteNoPermission)
        }
        try manager.createDirectory(at: root, withIntermediateDirectories: true)
        let values = try root.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
        guard values.isDirectory == true, values.isSymbolicLink != true else {
            throw CocoaError(.fileWriteNoPermission)
        }
        var url = root
        var resourceValues = URLResourceValues()
        resourceValues.isExcludedFromBackup = true
        try url.setResourceValues(resourceValues)
        #if os(iOS)
        try manager.setAttributes([.protectionKey: FileProtectionType.completeUntilFirstUserAuthentication],
                                  ofItemAtPath: root.path)
        #endif
    }
}

extension StorageManager {
    /// Only iOS opts into an app-owned root. Previews and injected test storage
    /// keep their existing non-persistent policy.
    public static func applicationDefault() -> StorageManager {
        #if os(iOS)
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
        return StorageManager(appOwnedModelRoot: documents?.appendingPathComponent("Models", isDirectory: true))
        #else
        return StorageManager()
        #endif
    }
}
