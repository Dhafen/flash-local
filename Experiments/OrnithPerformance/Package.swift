// swift-tools-version: 6.2
import PackageDescription

// Deliberately separate from the production package and its GGUF runners.
let package = Package(
    name: "OrnithPerformance",
    platforms: [.macOS(.v14), .iOS(.v18)],
    products: [.library(name: "OrnithPerformance", targets: ["OrnithPerformance"])],
    dependencies: [
        .package(url: "https://github.com/Dhafen/flash-local-swiftlet.git",
                 revision: "4a0510ecf678b3a5d4a5715abcd0bbb2aa839845"),
    ],
    targets: [
        .target(name: "OrnithPerformance", dependencies: [
            .product(name: "SwiftletCore", package: "flash-local-swiftlet"),
        ], resources: [.copy("Resources/ornith-mlx-source.json")]),
        .testTarget(name: "OrnithPerformanceTests", dependencies: ["OrnithPerformance"]),
    ],
    swiftLanguageModes: [.v5]
)
