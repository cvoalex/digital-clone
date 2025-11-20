// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "SwiftInference",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(
            name: "swift-infer",
            targets: ["SwiftInference"]
        )
    ],
    dependencies: [],
    targets: [
        .executableTarget(
            name: "SwiftInference",
            dependencies: [],
            path: "Sources"
        )
    ]
)

