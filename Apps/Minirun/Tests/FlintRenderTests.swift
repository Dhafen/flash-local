#if canImport(FlintEngine)
import SwiftUI
import XCTest
import MinirunKit
#if canImport(UIKit)
import UIKit
#else
import AppKit
#endif
@testable import MinirunApp

/// Actual native control captures. Empty model storage; no inference or fake metrics.
@MainActor
final class FlintRenderTests: XCTestCase {
    func testUnloadedChatAPIAndMeasurementsInBothAppearances() throws {
        let keys = ["FlashLocal.Flint.System.v1", "FlashLocal.Flint.Context.v1", "FlashLocal.Flint.Options.v1"]
        let defaults = UserDefaults.standard
        let previous = keys.map { defaults.object(forKey: $0) }
        for key in keys { defaults.removeObject(forKey: key) }
        defer {
            for (key, value) in zip(keys, previous) {
                if let value { defaults.set(value, forKey: key) }
                else { defaults.removeObject(forKey: key) }
            }
        }
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        var captures: [ColorScheme: [Data]] = [:]
        for pane in FlintView.Pane.allCases {
            for appearance in [ColorScheme.light, .dark] {
                let model = FlintAppModel(modelsRoot: directory.appendingPathComponent("Models"),
                    privateRoot: directory.appendingPathComponent("State"), documents: directory.appendingPathComponent("Documents"))
                XCTAssertFalse(model.hasModel)
                let content = FlintView(model: model, pane: pane).environment(\.colorScheme, appearance)
                let data: Data
                #if canImport(UIKit)
                let host = UIHostingController(rootView: content)
                guard let scene = UIApplication.shared.connectedScenes.compactMap({ $0 as? UIWindowScene }).first else {
                    throw XCTSkip("A real Simulator window scene is required")
                }
                let window = UIWindow(windowScene: scene)
                window.frame = CGRect(x: 0, y: 0, width: 430, height: 1100)
                window.overrideUserInterfaceStyle = appearance == .dark ? .dark : .light
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
                let window = NSWindow(contentRect: CGRect(x: 0, y: 0, width: 430, height: 1100),
                    styleMask: .borderless, backing: .buffered, defer: false)
                window.isReleasedWhenClosed = false
                window.appearance = NSAppearance(named: appearance == .dark ? .darkAqua : .aqua)
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
                XCTAssertFalse(model.ready || model.isBusy || model.apiListening)
                XCTAssertNil(model.report, "Rendering must never fabricate performance results")
                let attachment = XCTAttachment(data: data, uniformTypeIdentifier: "public.png")
                attachment.name = "flint-\(platform)-unloaded-\(pane.rawValue)-\(appearance == .dark ? "dark" : "light").png"
                attachment.lifetime = .keepAlways
                add(attachment)
                captures[appearance, default: []].append(data)
            }
        }
        for appearance in [ColorScheme.light, .dark] {
            XCTAssertEqual(captures[appearance]?.count, 3)
            XCTAssertNotEqual(captures[appearance]?[0], captures[appearance]?[1])
            XCTAssertNotEqual(captures[appearance]?[1], captures[appearance]?[2])
        }
        XCTAssertNotEqual(captures[.light]?[0], captures[.dark]?[0])
        print("FLINT_RENDER unloadedPanes=3 appearances=2 realControls=true modelLoaded=false trainedInference=false")
    }
}
#endif
