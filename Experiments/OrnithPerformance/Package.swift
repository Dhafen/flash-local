// swift-tools-version: 6.2
import PackageDescription

// Deliberately separate from the production package and its GGUF runners.
let package = Package(
    name: "OrnithPerformance",
    platforms: [.macOS(.v14), .iOS(.v18)],
    products: [.library(name: "OrnithPerformance", targets: ["OrnithPerformance"])],
    dependencies: [
        .package(url: "https://github.com/Dhafen/flash-local-swiftlet.git",
                 revision: "1e348e7b61289d70ba5bde2c212bcfa7867f4323"),
        .package(url: "https://github.com/huggingface/swift-transformers", exact: "1.3.3"),
    ],
    targets: [
        .target(name: "OrnithPerformance", dependencies: [
            .product(name: "SwiftletCore", package: "flash-local-swiftlet"),
            .product(name: "Tokenizers", package: "swift-transformers"),
        ], resources: [.copy("Resources/ornith-mlx-source.json")]),
        .testTarget(name: "OrnithPerformanceTests", dependencies: ["OrnithPerformance"]),
    ],
    swiftLanguageModes: [.v5]
)
