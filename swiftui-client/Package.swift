// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "RAGStudio",
    platforms: [
        .macOS(.v15),
        .iOS(.v18)
    ],
    products: [
        .executable(name: "RAGStudio", targets: ["RAGStudio"])
    ],
    targets: [
        .executableTarget(
            name: "RAGStudio",
            path: "Sources"
        )
    ]
)
