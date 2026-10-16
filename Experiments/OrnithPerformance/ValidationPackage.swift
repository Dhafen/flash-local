// swift-tools-version: 6.2
import PackageDescription

// CI-only harness over the pinned, unmodified SwiftletCore sources/tests.
// Xcode's aggregate upstream scheme cannot import an executable module from
// SwiftletServerTests. That server is unrelated to the embedded phone engine.
let package = Package(
    name: "SwiftletValidation",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [.library(name: "SwiftletCore", targets: ["SwiftletCore"])],
    dependencies: [.package(url: "https://github.com/huggingface/swift-transformers", exact: "1.3.3")],
    targets: [
        .target(name: "SwiftletCore", dependencies: [
            .product(name: "Tokenizers", package: "swift-transformers"),
        ], resources: [.copy("Kernels.metal.txt")]),
        .testTarget(name: "SwiftletCoreTests", dependencies: ["SwiftletCore"]),
    ]
)
