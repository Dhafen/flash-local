import SwiftUI
import XCTest
#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif
@testable import MinirunApp

@MainActor
final class ForkBrandRenderTests: XCTestCase {
    func testAboutPageInLightAndDarkAndBundledLicenses() throws {
        XCTAssertEqual(AboutPresentation.productName, "Flash Local")
        let notices = try XCTUnwrap(Bundle.main.url(forResource: "OpenSourceNotices", withExtension: "txt"))
        let text = try String(contentsOf: notices, encoding: .utf8)
        XCTAssertTrue(text.contains("Apache License"))
        XCTAssertTrue(text.contains("The ggml authors"))
        var rendered: [Data] = []
        for scheme in [ColorScheme.light, .dark] {
            let renderer = ImageRenderer(content: AboutPage().frame(width: 430)
                .background(MRColor.abyss).environment(\.colorScheme, scheme))
            renderer.scale = 2
            let png: Data
            let platform: String
            #if canImport(UIKit)
            png = try XCTUnwrap(renderer.uiImage?.pngData())
            platform = "ios-simulator"
            #else
            let tiff = try XCTUnwrap(renderer.nsImage?.tiffRepresentation)
            png = try XCTUnwrap(NSBitmapImageRep(data: tiff)?.representation(using: .png, properties: [:]))
            platform = "macos"
            #endif
            let name = "flash-local-about-\(platform)-\(scheme == .dark ? "dark" : "light").png"
            let attachment = XCTAttachment(data: png, uniformTypeIdentifier: "public.png")
            attachment.name = name
            attachment.lifetime = .keepAlways
            add(attachment)
            rendered.append(png)
            print("FORK_VISUAL_REVIEW \(name) bytes=\(png.count)")
        }
        XCTAssertNotEqual(rendered[0], rendered[1], "The two appearance reviews must actually differ")
    }
}
