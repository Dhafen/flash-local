#!/usr/bin/env python3
"""Apply a version-pinned, isolated context-window UI overlay to MiniCPM-V-Apps.

This modifies ONLY the checked-out upstream working copy, never flash-local.
Anchors are intentionally strict: fail closed when upstream source changes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

UPSTREAM_SHA = "fa9d837e3d38dfde7a9484ec1cc6bc2e5bc9b420"
CONTEXT_OPTIONS = (4096, 8192, 16384, 32768, 65536)

def replace_once(text: str, before: str, after: str, path: Path) -> str:
    count = text.count(before)
    if count != 1:
        raise RuntimeError(f"{path}: expected one patch anchor; found {count}: {before[:100]!r}")
    return text.replace(before, after, 1)

def update(root: Path, relative: str, changes: list[tuple[str, str]]) -> None:
    path = root / relative
    data = path.read_text(encoding="utf-8")
    for before, after in changes:
        data = replace_once(data, before, after, path)
    path.write_text(data, encoding="utf-8")
    print(f"PATCHED {relative}")

def apply(root: Path) -> None:
    model = "MiniCPM-V-demo/Sources/Settings/ModelConst/MiniCPMModelConst.swift"
    update(root, model, [
        (
            "    static let defaultNCtx = 4096\n    /// 对 minicpmv4_6",
            """    // Opt-in; preserve the original 4096-token default for existing installs.
    static let contextWindowOptions = [4096, 8192, 16384, 32768, 65536]
    private static let contextWindowKey = "minicpm_v47_selected_context_tokens"

    static var defaultNCtx: Int {
        let chosen = UserDefaults.standard.integer(forKey: contextWindowKey)
        return contextWindowOptions.contains(chosen) ? chosen : 4096
    }

    static func setContextWindowTokens(_ tokens: Int) {
        guard contextWindowOptions.contains(tokens) else { return }
        UserDefaults.standard.set(tokens, forKey: contextWindowKey)
    }

    /// KV estimate for ten attention layers of V4.7 (approx. 20 KiB / token).
    static func contextKVMiB(_ tokens: Int) -> Int {
        (tokens * 20_480 + 1_048_575) / 1_048_576
    }

    /// 对 minicpmv4_6"""
        ),
        (
            """        let attempted = cacheWindowC()
        ud.set(false, forKey: cacheWindowPendingKey)
        if attempted > 0 {""",
            """        let attempted = cacheWindowC()
        ud.set(false, forKey: cacheWindowPendingKey)
        // A larger context can exhaust iOS memory during model initialization.
        // Reuse the existing boot-time pending-load recovery to avoid a crash loop.
        if defaultNCtx > 4096 {
            print("[MODEL] previous load unfinished: reverting context \\(defaultNCtx) -> 4096")
            setContextWindowTokens(4096)
        }
        if attempted > 0 {"""
        )
    ])

    keys = "MiniCPM-V-demo/Sources/Localization/L10nKeys.swift"
    update(root, keys, [(
        '        public static let rowLookaheadPrefetchDepth       = "settings.row.lookaheadPrefetchDepth"',
        """        public static let rowContextWindow = "settings.row.contextWindow"
        public static let contextWindowTitle = "settings.contextWindow.title"
        public static let contextWindowMessage = "settings.contextWindow.message"
        public static let rowLookaheadPrefetchDepth       = "settings.row.lookaheadPrefetchDepth" """
        .rstrip()
    )])

    for language, vals in (
        ("en", (
            "Context window",
            "MiniCPM-V 4.7 context window",
            "Changes take effect after reloading the model. The selected size includes input, history and output. Larger settings raise KV memory use; 65,536 tokens is experimental and has NOT been validated on a real iPhone. If loading fails, the next launch falls back to 4,096."
        )),
        ("zh", (
            "上下文窗口",
            "MiniCPM-V 4.7 上下文窗口",
            "更改后需要重新加载模型。所选长度包括输入、历史和输出。更大的窗口会增加 KV 内存占用；65,536 token 尚未在真机上验证。如果加载未完成，下次启动将回退到 4,096。"
        )),
    ):
        p = f"MiniCPM-V-demo/Sources/Localization/L10n+{language}.swift"
        def swift_quote(v: str) -> str:
            return v.replace("\\", "\\\\").replace('"', '\\"')
        rows = (
            f'        L.Settings.rowContextWindow: "{swift_quote(vals[0])}",\n'
            f'        L.Settings.contextWindowTitle: "{swift_quote(vals[1])}",\n'
            f'        L.Settings.contextWindowMessage: "{swift_quote(vals[2])}",\n'
        )
        anchor = '        L.Settings.rowLookaheadPrefetchDepth:'
        path = root / p
        data = path.read_text(encoding="utf-8")
        if data.count(anchor) != 1:
            raise RuntimeError(f"{p}: invalid row anchor count")
        path.write_text(data.replace(anchor, rows + anchor, 1), encoding="utf-8")
        print(f"PATCHED {p}")

    controller = "MiniCPM-V-demo/Sources/Settings/ViewControler/MBSettingsViewController.swift"
    update(root, controller, [
        (
            """        if title == L.Settings.rowLanguage.loc {
            showLanguagePicker()
        } else if title == L.Settings.rowLookaheadPrefetchDepth.loc {""",
            """        if title == L.Settings.rowLanguage.loc {
            showLanguagePicker()
        } else if title == L.Settings.rowContextWindow.loc {
            presentContextWindowPicker()
        } else if title == L.Settings.rowLookaheadPrefetchDepth.loc {"""
        ),
        (
            """    private func handleOtherSettings(model: MBSettingsModel, at indexPath: IndexPath) {""",
            """    /// Context allocation happens when the model is initialized; changes need reload.
    private func presentContextWindowPicker() {
        let alert = UIAlertController(
            title: L.Settings.contextWindowTitle.loc,
            message: L.Settings.contextWindowMessage.loc,
            preferredStyle: .actionSheet
        )
        let current = MiniCPMV47Const.defaultNCtx
        for tokens in MiniCPMV47Const.contextWindowOptions {
            let kv = MiniCPMV47Const.contextKVMiB(tokens)
            let title = "\\(tokens) tokens (KV ~\\(kv) MiB)" + (tokens == current ? " ✓" : "")
            alert.addAction(UIAlertAction(title: title, style: .default) { [weak self] _ in
                MiniCPMV47Const.setContextWindowTokens(tokens)
                self?.loadTableViewData()
            })
        }
        alert.addAction(UIAlertAction(title: L.Common.cancel.loc, style: .cancel))
        if let pop = alert.popoverPresentationController {
            pop.sourceView = self.view
            if let selected = tableView.indexPathForSelectedRow,
               let cell = tableView.cellForRow(at: selected) {
                pop.sourceView = cell
                pop.sourceRect = cell.bounds
            } else {
                pop.sourceRect = CGRect(x: self.view.bounds.midX, y: self.view.bounds.midY, width: 0, height: 0)
                pop.permittedArrowDirections = []
            }
        }
        present(alert, animated: true)
    }

    private func handleOtherSettings(model: MBSettingsModel, at indexPath: IndexPath) {"""
        ),
        (
            """        if isMoEModelActive {
            let lookaheadPrefetchModel = MBSettingsModel()""",
            """        if isMoEModelActive {
            let contextModel = MBSettingsModel()
            contextModel.title = L.Settings.rowContextWindow.loc
            contextModel.icon = UIImage(systemName: "text.alignleft")
            contextModel.accessoryIcon = UIImage(named: "setting_accessory_icon")
            contextModel.detailText = "\\(MiniCPMV47Const.defaultNCtx) tokens"
            section.append(contextModel)

            let lookaheadPrefetchModel = MBSettingsModel()"""
        )
    ])

    # Verify cross-file wiring after applying patches.
    constants = (root / model).read_text(encoding="utf-8")
    load = (root / "MiniCPM-V-demo/Sources/Home/ViewController/MBHomeViewController+LoadModel.swift").read_text(encoding="utf-8")
    assert "nCtx: MiniCPMV47Const.defaultNCtx" in load
    assert "let nCtx = defaultNCtx" in constants
    assert "applyCacheWindowCrashRecoveryIfNeeded" in constants
    assert str(CONTEXT_OPTIONS[-1]) in constants
    print(f"Overlay ready. Choices: {CONTEXT_OPTIONS}. Original default: 4096.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("upstream", type=Path, help="Checked-out MiniCPM-V-Apps repository root")
    args = parser.parse_args()
    apply(args.upstream.resolve())
