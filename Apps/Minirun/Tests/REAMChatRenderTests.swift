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
        var captures: [ColorScheme: [Data]] = [:]
        for preview in [false, true] {
            let model = preview ? REAMAppModel.reviewPreview() : REAMAppModel(root: FileManager.default.temporaryDirectory)
            for scheme in [ColorScheme.light, .dark] {
                // ImageRenderer omits ScrollView contents and replaces native
                // text fields with yellow placeholders. Host real controls in
                // a real window before capturing, so a PNG proves actual UI.
                #if canImport(OrnithLES)
                let content = REAMChatPage(model: model, openLES: {}).environment(\.colorScheme, scheme)
                #else
                let content = REAMChatPage(model: model).environment(\.colorScheme, scheme)
                #endif
                let data = try capture(content, scheme: scheme)
                XCTAssertGreaterThan(data.count, 10000)
                let attachment = XCTAttachment(data: data, uniformTypeIdentifier: "public.png")
                attachment.name = "ream-chat-\(platform)-\(preview ? "preview" : "missing")-\(scheme == .dark ? "dark" : "light").png"
                attachment.lifetime = .keepAlways; add(attachment)
                captures[scheme, default: []].append(data)
            }
        }
        for scheme in [ColorScheme.light, .dark] {
            XCTAssertEqual(captures[scheme]?.count, 2)
            XCTAssertNotEqual(captures[scheme]?[0], captures[scheme]?[1],
                "Missing-model and preview pages must show different visible content, not the same empty viewport")
        }
    }
    func testThinkingAndAnswerRowsDrawInBothAppearances() throws {
        var captures: [ColorScheme: [Data]] = [:]
        for active in [true, false] {
            let message = try XCTUnwrap(REAMAppModel.reviewThinkingPreview(active: active).messages.last)
            for scheme in [ColorScheme.light, .dark] {
                let content = VStack(alignment: .leading, spacing: 24) {
                    Text("Förhandsvisning – ingen modell körs.").font(.caption).foregroundStyle(.secondary)
                    REAMMessageRow(message: message, isGenerating: false)
                    Spacer()
                }.padding(20).frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(scheme == .dark ? Color.black : Color.white)
                    .environment(\.colorScheme, scheme)
                let data = try capture(content, scheme: scheme)
                XCTAssertGreaterThan(data.count, 10000)
                let attachment = XCTAttachment(data: data, uniformTypeIdentifier: "public.png")
                attachment.name = "ornith-thinking-\(platform)-\(active ? "active" : "answer")-\(scheme == .dark ? "dark" : "light").png"
                attachment.lifetime = .keepAlways; add(attachment)
                captures[scheme, default: []].append(data)
            }
        }
        for scheme in [ColorScheme.light, .dark] {
            XCTAssertEqual(captures[scheme]?.count, 2)
            XCTAssertNotEqual(captures[scheme]?[0], captures[scheme]?[1])
        }
    }
    private var platform: String {
        #if canImport(UIKit)
        "ios"
        #else
        "macos"
        #endif
    }
    private func capture<Content: View>(_ content: Content, scheme: ColorScheme) throws -> Data {
        #if canImport(UIKit)
        let host = UIHostingController(rootView: content)
        guard let scene = UIApplication.shared.connectedScenes.compactMap({ $0 as? UIWindowScene }).first else {
            throw XCTSkip("A simulator window scene is required for real control rendering")
        }
        let window = UIWindow(windowScene: scene)
        window.frame = CGRect(x: 0, y: 0, width: 430, height: 900)
        window.overrideUserInterfaceStyle = scheme == .dark ? .dark : .light
        window.rootViewController = host; window.makeKeyAndVisible(); host.view.frame = window.bounds
        host.view.setNeedsLayout(); host.view.layoutIfNeeded()
        RunLoop.main.run(until: Date().addingTimeInterval(0.15))
        let format = UIGraphicsImageRendererFormat(); format.scale = 2
        let image = UIGraphicsImageRenderer(size: window.bounds.size, format: format).image { _ in
            XCTAssertTrue(window.drawHierarchy(in: window.bounds, afterScreenUpdates: true))
        }
        let data = try XCTUnwrap(image.pngData())
        window.isHidden = true; window.rootViewController = nil
        return data
        #else
        let host = NSHostingView(rootView: content)
        let window = NSWindow(contentRect: CGRect(x: 0, y: 0, width: 430, height: 900),
            styleMask: .borderless, backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.appearance = NSAppearance(named: scheme == .dark ? .darkAqua : .aqua)
        window.contentView = host; window.orderFront(nil); host.frame = window.contentLayoutRect
        host.layoutSubtreeIfNeeded(); RunLoop.main.run(until: Date().addingTimeInterval(0.15)); host.displayIfNeeded()
        let bitmap = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
        host.cacheDisplay(in: host.bounds, to: bitmap)
        let data = try XCTUnwrap(bitmap.representation(using: .png, properties: [:]))
        window.close(); return data
        #endif
    }
}
