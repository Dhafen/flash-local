import SwiftUI
import XCTest
#if canImport(UIKit)
import UIKit
#else
import AppKit
#endif
@testable import MinirunApp

@MainActor
final class REAMChatRenderTests: XCTestCase {
    func testMissingAndPreviewChatDrawInBothAppearances() throws {
        for preview in [false, true] {
            let model = preview ? REAMAppModel.reviewPreview() : REAMAppModel(root: FileManager.default.temporaryDirectory)
            for scheme in [ColorScheme.light, .dark] {
                let renderer = ImageRenderer(content: REAMChatPage(model: model).frame(width: 430, height: 900)
                    .environment(\.colorScheme, scheme))
                renderer.scale = 2
                let data: Data
                #if canImport(UIKit)
                data = try XCTUnwrap(renderer.uiImage?.pngData())
                let platform = "ios"
                #else
                let tiff = try XCTUnwrap(renderer.nsImage?.tiffRepresentation)
                data = try XCTUnwrap(NSBitmapImageRep(data: tiff)?.representation(using: .png, properties: [:]))
                let platform = "macos"
                #endif
                XCTAssertGreaterThan(data.count, 10000)
                let attachment = XCTAttachment(data: data, uniformTypeIdentifier: "public.png")
                attachment.name = "ream-chat-\(platform)-\(preview ? "preview" : "missing")-\(scheme == .dark ? "dark" : "light").png"
                attachment.lifetime = .keepAlways; add(attachment)
            }
        }
    }
}
