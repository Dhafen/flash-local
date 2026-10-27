// swift-tools-version: 5.9
import PackageDescription
import Foundation

// Only the isolated experiment opts in. Normal production resolution and its
// tokenizer/reference binary remain unchanged.
let enableLES = ProcessInfo.processInfo.environment["FLASH_LOCAL_ENABLE_LES"] == "1"
let enableFlint = ProcessInfo.processInfo.environment["FLASH_LOCAL_ENABLE_FLINT"] == "1"

let package = Package(
    name: "minirun",
    platforms: [
        .macOS(.v14),
        .iOS("18.0"),
    ],
    products: [
        .library(name: "StorageCore", targets: ["StorageCore"]),
        .library(name: "GGUFCore", targets: ["GGUFCore"]),
        .library(name: "MLXBridge", targets: ["MLXBridge"]),
        .library(name: "ModelAdapters", targets: ["ModelAdapters"]),
        .library(name: "iOSBench", targets: ["iOSBench"]),
        .library(name: "BenchScenarios", targets: ["BenchScenarios"]),
        .library(name: "MinirunKit", targets: ["MinirunKit"]),
        .library(name: "MinirunRunners", targets: ["MinirunRunners"]),
    ] + (enableLES ? [.library(name: "OrnithLES", targets: ["OrnithLES"])] : [])
      + (enableFlint ? [.library(name: "FlintEngine", targets: ["FlintEngine"])] : []),
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", exact: "0.31.4"),
    ],
    targets: [
        // Generated reproducibly from the pinned source, before package
        // resolution. See scripts/build_tokenizer_xcframework.py.
        .binaryTarget(name: "LlamaTokenizerRuntime",
                      path: ".work/runtime/LlamaTokenizerRuntime.xcframework"),
        .target(name: "StorageCore", plugins: ["EmbedGitRevision"]),
        .testTarget(name: "StorageCoreTests", dependencies: ["StorageCore"]),
        .target(name: "CGGUFQuants", exclude: ["LICENSE.llama.cpp", "provenance.json"],
                publicHeadersPath: "include"),
        .target(name: "GGUFCore", dependencies: ["StorageCore", "CGGUFQuants"]),
        .testTarget(name: "GGUFCoreTests", dependencies: ["GGUFCore", "StorageCore"]),
        .target(
            name: "MLXBridge",
            dependencies: [
                "StorageCore",
                "GGUFCore",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "MinirunKit",
            dependencies: ["StorageCore", "iOSBench", "GGUFCore"],
            resources: [.copy("Resources/model-catalog.json")]
        ),
        .testTarget(name: "MLXBridgeTests", dependencies: [
            "MLXBridge", "GGUFCore", "StorageCore",
            .product(name: "MLX", package: "mlx-swift"),
        ]),
        .testTarget(
            name: "MinirunKitTests",
            dependencies: ["MinirunKit", "StorageCore"],
            resources: [.copy("Fixtures/hf")]
        ),
        .target(name: "iOSBench", dependencies: ["StorageCore"]),
        .testTarget(
            name: "iOSBenchTests",
            dependencies: ["iOSBench", "StorageCore"]
        ),
        .target(
            name: "ModelAdapters",
            dependencies: ["MLXBridge", "MinirunKit", "StorageCore", "GGUFCore", "LlamaTokenizerRuntime"],
            linkerSettings: [.linkedLibrary("c++")]
        ),
        .target(
            name: "BenchScenarios",
            dependencies: ["ModelAdapters", "iOSBench", "StorageCore"],
            resources: [.copy("Resources/k3-flagship-config.json")]
        ),
        .testTarget(
            name: "ModelAdaptersTests",
            dependencies: ["ModelAdapters", "GGUFCore", "StorageCore", "MinirunKit"]
        ),
        .target(
            name: "MinirunRunners",
            dependencies: [
                "MinirunKit",
                "GGUFCore",
                "BenchScenarios",
                "ModelAdapters",
                "StorageCore",
                .product(name: "MLX", package: "mlx-swift"),
            ] + (enableLES ? [.target(name: "OrnithLES")] : [])
        ),
        .testTarget(
            name: "MinirunRunnersTests",
            dependencies: ["MinirunRunners", "ModelAdapters", "StorageCore", "MinirunKit"]
        ),
        .plugin(name: "EmbedGitRevision", capability: .buildTool()),
    ] + (enableLES ? [
        .binaryTarget(name: "FlashOrnithLES", path: ".work/ornith-les-runtime/FlashOrnithLES.xcframework"),
        .target(name: "OrnithLES", dependencies: ["FlashOrnithLES", "ModelAdapters", "GGUFCore", "MinirunKit", "StorageCore",
            .product(name: "MLX", package: "mlx-swift")], path: "Experiments/OrnithLES/Sources/OrnithLES"),
        .testTarget(name: "OrnithLESTests", dependencies: ["OrnithLES", "GGUFCore", "StorageCore", "ModelAdapters", "MinirunKit"],
            path: "Experiments/OrnithLES/Tests/OrnithLESTests")
    ] : []) + (enableFlint ? [
        .binaryTarget(name: "FlashFlint", path: ".work/flint-runtime/FlashFlint.xcframework"),
        .target(name: "FlintEngine", dependencies: ["FlashFlint", "GGUFCore", "MinirunKit", "StorageCore"],
            path: "Experiments/Flint/Sources/FlintEngine"),
        .testTarget(name: "FlintEngineTests", dependencies: ["FlintEngine"],
            path: "Experiments/Flint/Tests/FlintEngineTests")
    ] : [])
)
