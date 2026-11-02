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
    func testBackgroundSceneDisablesFlintChatAndLANAdmission() {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let model = FlintAppModel(modelsRoot: directory.appendingPathComponent("Models"),
            privateRoot: directory.appendingPathComponent("State"),
            documents: directory.appendingPathComponent("Documents"))
        XCTAssertTrue(model.canStart)
        XCTAssertFalse(model.canSend, "No model has been loaded")
        model.setActive(false)
        XCTAssertFalse(model.canStart, "Backgrounded scenes cannot start Flint operations")
        XCTAssertFalse(model.canSend)
        model.setActive(true)
        XCTAssertTrue(model.canStart, "Foregrounding restores permission to load a model")
        XCTAssertFalse(model.canSend, "Foregrounding must not fabricate a loaded model")
    }
    func testExplicitModelRemovalPreservesPartialDownloadAndChatState() async throws {
        let fm = FileManager.default
        let directory = fm.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? fm.removeItem(at: directory) }
        let root = directory.appendingPathComponent("Models")
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        let installed = root.appendingPathComponent(FlintArtifact.manifest.fileName)
        let partial = root.appendingPathComponent(FlintArtifact.manifest.fileName + ".partial")
        try Data("invalid model bytes".utf8).write(to: installed)
        try Data("resume bytes".utf8).write(to: partial)
        let model = FlintAppModel(modelsRoot: root, privateRoot: directory.appendingPathComponent("State"),
            documents: directory.appendingPathComponent("Documents"))
        XCTAssertTrue(model.hasModel)
        await model.removeInstalledModel()
        XCTAssertFalse(model.hasModel, "An explicitly confirmed recovery must unblock re-download")
        XCTAssertFalse(fm.fileExists(atPath: installed.path))
        XCTAssertEqual(try Data(contentsOf: partial), Data("resume bytes".utf8),
                       "Never delete a resumable partial transfer")
    }
    func testBackgroundCannotRemoveInstalledModel() async throws {
        let fm = FileManager.default
        let directory = fm.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? fm.removeItem(at: directory) }
        let root = directory.appendingPathComponent("Models")
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        let installed = root.appendingPathComponent(FlintArtifact.manifest.fileName)
        try Data("keep".utf8).write(to: installed)
        let model = FlintAppModel(modelsRoot: root, privateRoot: directory.appendingPathComponent("State"),
            documents: directory.appendingPathComponent("Documents"))
        model.setActive(false)
        await model.removeInstalledModel()
        XCTAssertTrue(fm.fileExists(atPath: installed.path))
        XCTAssertTrue(model.hasModel)
    }
    func testLegacyUnsupportedPreferencesDoNotBlockModelLoading() {
        let defaults = UserDefaults.standard
        let contextKey = "FlashLocal.Flint.Context.v1"
        let cacheKey = "FlashLocal.Flint.DecodeWindow.v1"
        let previousContext = defaults.object(forKey: contextKey)
        let previousCache = defaults.object(forKey: cacheKey)
        defer {
            if let previousContext { defaults.set(previousContext, forKey: contextKey) }
            else { defaults.removeObject(forKey: contextKey) }
            if let previousCache { defaults.set(previousCache, forKey: cacheKey) }
            else { defaults.removeObject(forKey: cacheKey) }
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        func model() -> FlintAppModel {
            FlintAppModel(modelsRoot: root.appendingPathComponent("Models"),
                privateRoot: root.appendingPathComponent("State"),
                documents: root.appendingPathComponent("Documents"))
        }
        // An older version could persist these options even though the
        // current UI has no way to select or clear them.
        defaults.set(LocalContextProfile.test65K.rawValue, forKey: contextKey)
        defaults.set(32, forKey: cacheKey)
        let restored = model()
        XCTAssertEqual(restored.contextProfile, .standard)
        XCTAssertEqual(restored.decodeCacheWindow, 0)

        // Valid context choices must still survive an app restart.
        defaults.set(LocalContextProfile.test32K.rawValue, forKey: contextKey)
        XCTAssertEqual(model().contextProfile, .test32K)
    }
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
                // cacheDisplay does not include NSWindow's background. Capture
                // the real controls over an opaque native background so light
                // text/alpha cannot disappear when the PNG is reviewed.
                let host = NSHostingView(rootView: content.background(Color(nsColor: .windowBackgroundColor)))
                let window = NSWindow(contentRect: CGRect(x: 0, y: 0, width: 430, height: 1100),
                    styleMask: .borderless, backing: .buffered, defer: false)
                window.isReleasedWhenClosed = false
                window.appearance = NSAppearance(named: appearance == .dark ? .darkAqua : .aqua)
                window.backgroundColor = .windowBackgroundColor
                host.appearance = window.appearance
                window.contentView = host; window.orderFront(nil)
                host.frame = window.contentLayoutRect
                host.layoutSubtreeIfNeeded()
                RunLoop.main.run(until: Date().addingTimeInterval(0.15))
                host.displayIfNeeded()
                let bitmap = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
                host.cacheDisplay(in: host.bounds, to: bitmap)
                XCTAssertGreaterThan(try XCTUnwrap(bitmap.colorAt(x: 0, y: 0)).alphaComponent, 0.99)
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
