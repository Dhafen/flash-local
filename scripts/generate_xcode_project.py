"""Regenerate with the verified XcodeGen 2.46.0 release on macOS."""
import hashlib
import io
import pathlib
import subprocess
import urllib.request
import zipfile

URL = "https://github.com/yonaskolb/XcodeGen/releases/download/2.46.0/xcodegen.zip"
SHA256 = "4d9e34b62172d645eed6457cac13fc222569974098ef4ee9c3368bedf0196806"


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
    subprocess.run([str(binary), "generate", "--spec", "project.yml"],
                   cwd=root / "Apps/Minirun", check=True)


if __name__ == "__main__":
    main()
