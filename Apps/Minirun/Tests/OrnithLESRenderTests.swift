#if canImport(OrnithLES)
import SwiftUI
import XCTest
#if canImport(UIKit)
import UIKit
#else
import AppKit
#endif
@testable import MinirunApp

@MainActor
final class OrnithLESRenderTests: XCTestCase {
    private enum Fixture: String, CaseIterable { case missing, preview, chatThinking = "chat-thinking", chatComplete = "chat-complete" }
    func testMissingAndPreviewChatDrawInBothAppearances() throws {
        var captures: [ColorScheme: [Data]] = [:]
        for fixture in Fixture.allCases {
            let model: OrnithLESAppModel
            switch fixture {
            case .missing: model = OrnithLESAppModel(modelsRoot: FileManager.default.temporaryDirectory, privateRoot: FileManager.default.temporaryDirectory, documents: FileManager.default.temporaryDirectory)
            case .preview: model = .reviewPreview()
            case .chatThinking: model = .reviewChatPreview(thinking: true)
            case .chatComplete: model = .reviewChatPreview(thinking: false)
            }
            for scheme in [ColorScheme.light, .dark] {
                // ImageRenderer omits ScrollView contents and replaces native
                // text fields with yellow placeholders. Host real controls in
                // a real window before capturing, so a PNG proves actual UI.
                let content = Group {
                    if fixture == .chatThinking || fixture == .chatComplete { OrnithLESChatPage(model: model) }
                    else { OrnithLESPage(model: model) }
                }.environment(\.colorScheme, scheme)
                let data: Data
                #if canImport(UIKit)
                let host = UIHostingController(rootView: content)
                let window: UIWindow
                if let scene = UIApplication.shared.connectedScenes.compactMap({ $0 as? UIWindowScene }).first {
                    window = UIWindow(windowScene: scene)
                } else { throw XCTSkip("A simulator window scene is required for real control rendering") }
                window.frame = CGRect(x: 0, y: 0, width: 430, height: 900)
                window.overrideUserInterfaceStyle = scheme == .dark ? .dark : .light
                window.rootViewController = host
                window.makeKeyAndVisible()
                host.view.frame = window.bounds
                host.view.setNeedsLayout(); host.view.layoutIfNeeded()
                RunLoop.main.run(until: Date().addingTimeInterval(0.15))
                let format = UIGraphicsImageRendererFormat(); format.scale = 2
                let image = UIGraphicsImageRenderer(size: window.bounds.size, format: format).image { _ in
                    XCTAssertTrue(window.drawHierarchy(in: window.bounds, afterScreenUpdates: true))
                }
                data = try XCTUnwrap(image.pngData())
                window.isHidden = true; window.rootViewController = nil
                let platform = "ios"
                #else
                let host = NSHostingView(rootView: content)
                let window = NSWindow(contentRect: CGRect(x: 0, y: 0, width: 430, height: 900),
                    styleMask: .borderless, backing: .buffered, defer: false)
                window.isReleasedWhenClosed = false
                window.appearance = NSAppearance(named: scheme == .dark ? .darkAqua : .aqua)
                window.contentView = host; window.orderFront(nil)
                host.frame = window.contentLayoutRect
                host.layoutSubtreeIfNeeded()
                RunLoop.main.run(until: Date().addingTimeInterval(0.15))
                host.displayIfNeeded()
                let bitmap = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
                host.cacheDisplay(in: host.bounds, to: bitmap)
                data = try XCTUnwrap(bitmap.representation(using: .png, properties: [:]))
                window.close()
                let platform = "macos"
                #endif
                XCTAssertGreaterThan(data.count, 10000)
                let attachment = XCTAttachment(data: data, uniformTypeIdentifier: "public.png")
                attachment.name = "ornith-les-\(platform)-\(fixture.rawValue)-\(scheme == .dark ? "dark" : "light").png"
                attachment.lifetime = .keepAlways; add(attachment)
                captures[scheme, default: []].append(data)
            }
        }
        for scheme in [ColorScheme.light, .dark] {
            XCTAssertEqual(captures[scheme]?.count, 4)
            XCTAssertNotEqual(captures[scheme]?[0], captures[scheme]?[1],
                "Missing-model and preview pages must show different visible content, not the same empty viewport")
        }
    }
}

#endif
