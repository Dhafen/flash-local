"""Regenerate with the verified XcodeGen 2.46.0 release on macOS."""
import hashlib
import io
import pathlib
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile

URL = "https://github.com/yonaskolb/XcodeGen/releases/download/2.46.0/xcodegen.zip"
SHA256 = "4d9e34b62172d645eed6457cac13fc222569974098ef4ee9c3368bedf0196806"


def generation_tree(root, temporary):
    """Source-only canonical layout; never copy weights, .git or user state.

    XcodeGen derives a local-package navigator identifier from the checkout
    basename. EAS calls it `build`, GitHub calls it `flash-local`. Generate in
    the latter layout so the existing project and package references reproduce
    exactly on both hosts; all adopted paths remain relative to the real app.
    """
    mirror = temporary / "flash-local"
    app = mirror / "Apps/Minirun"
    app.mkdir(parents=True)
    for name in ("Sources", "Tests", "Resources"):
        shutil.copytree(root / "Apps/Minirun" / name, app / name)
    for name in ("project.yml", "Info-iOS.plist", "Info-macOS.plist",
                 "Minirun-iOS.entitlements", "Minirun-macOS.entitlements"):
        # Apple's plist merge preserves existing whitespace. CRLF from a
        # Windows-made archive otherwise becomes CRCRLF during generation.
        text = (root / "Apps/Minirun" / name).read_text(encoding="utf-8")
        (app / name).write_text(text, encoding="utf-8", newline="\n")
    shutil.copyfile(root / "Package.swift", mirror / "Package.swift")
    experiment = mirror / "Experiments/OrnithPerformance"
    experiment.mkdir(parents=True)
    shutil.copyfile(root / "Experiments/OrnithPerformance/Package.swift", experiment / "Package.swift")
    return app


def generate(binary, root):
    root = root.resolve()
    tools = (root / ".work/tooling").resolve()
    if not tools.is_relative_to(root):
        raise RuntimeError("Generation staging must remain inside the source workspace")
    tools.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="project-generation-", dir=tools) as tmp:
        app = generation_tree(root, pathlib.Path(tmp))
        subprocess.run([str(binary), "generate", "--spec", "project.yml"], cwd=app, check=True)
        project = app / "Minirun.xcodeproj"
        text = (project / "project.pbxproj").read_text(encoding="utf-8")
        if str(app) in text or "relativePath = ../..;" not in text:
            raise RuntimeError("Generated project must retain relative local-package references")
        target = root / "Apps/Minirun"
        shutil.copytree(project, target / "Minirun.xcodeproj", dirs_exist_ok=True)
        for name in ("Info-iOS.plist", "Info-macOS.plist", "Minirun-iOS.entitlements", "Minirun-macOS.entitlements"):
            shutil.copyfile(app / name, target / name)


def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    tools = root / ".work/tooling"
    tools.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=60) as response:
        data = response.read(32 * 1024 * 1024 + 1)
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise RuntimeError("XcodeGen release hash mismatch")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for item in archive.infolist():
            target = (tools / item.filename).resolve()
            if not target.is_relative_to(tools.resolve()):
                raise RuntimeError("Tool archive escaped extraction root")
            if item.is_dir(): target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(item))
    binary = tools / "xcodegen/bin/xcodegen"
    binary.chmod(0o755)
    subprocess.run([str(binary), "--version"], check=True)
    generate(binary, root)


if __name__ == "__main__":
    main()
