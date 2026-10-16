"""Inspect an unsigned device archive before any signing/upload is attempted."""
import argparse
import hashlib
import json
import pathlib
import plistlib
import struct


def privacy_manifest(path):
    try:
        with path.open('rb') as stream:
            data = stream.read(65537)
    except OSError as error:
        raise ValueError("Missing app privacy manifest") from error
    if len(data) > 65536:
        raise ValueError("Unbounded app privacy manifest")
    try:
        manifest = plistlib.loads(data)
    except (ValueError, plistlib.InvalidFileException) as error:
        raise ValueError("Invalid app privacy manifest") from error
    if not isinstance(manifest, dict) or set(manifest) != {
            'NSPrivacyTracking', 'NSPrivacyCollectedDataTypes', 'NSPrivacyAccessedAPITypes'}:
        raise ValueError("Unexpected app privacy manifest keys")
    if manifest['NSPrivacyTracking'] is not False or manifest['NSPrivacyCollectedDataTypes'] != []:
        raise ValueError("Privacy manifest must match the app's no-tracking/no-collection behavior")
    expected = {'NSPrivacyAccessedAPICategoryFileTimestamp': ['C617.1'],
                'NSPrivacyAccessedAPICategorySystemBootTime': ['35F9.1'],
                'NSPrivacyAccessedAPICategoryDiskSpace': ['E174.1'],
                'NSPrivacyAccessedAPICategoryUserDefaults': ['CA92.1']}
    entries = manifest['NSPrivacyAccessedAPITypes']
    if not isinstance(entries, list) or len(entries) != len(expected):
        raise ValueError("Missing required reason API declaration")
    observed = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {'NSPrivacyAccessedAPIType', 'NSPrivacyAccessedAPITypeReasons'}:
            raise ValueError("Invalid required reason API declaration")
        category = entry['NSPrivacyAccessedAPIType']
        if not isinstance(category, str) or category in observed:
            raise ValueError("Invalid or duplicate required reason API category")
        observed[category] = entry['NSPrivacyAccessedAPITypeReasons']
    if observed != expected:
        raise ValueError("Required reason APIs do not match audited app behavior")


def device_executable(path):
    with path.open("rb") as stream:
        header = stream.read(32)
        if len(header) != 32:
            raise ValueError("Missing Mach-O executable header")
        magic, cpu, _, kind, commands, size, _, _ = struct.unpack("<8I", header)
        if (magic, cpu, kind) != (0xFEEDFACF, 0x0100000C, 2):
            raise ValueError("Expected a thin arm64 Mach-O executable")
        if not 0 < commands <= 10000 or not 0 < size <= 16 << 20:
            raise ValueError("Unbounded Mach-O load commands")
        data = stream.read(size)
    if len(data) != size:
        raise ValueError("Truncated Mach-O load commands")
    offset, platforms = 0, []
    for _ in range(commands):
        if offset + 8 > len(data):
            raise ValueError("Truncated Mach-O command")
        command, length = struct.unpack_from("<2I", data, offset)
        if length < 8 or length % 8 or offset + length > len(data):
            raise ValueError("Invalid Mach-O command length")
        if command == 0x32:  # LC_BUILD_VERSION, device iOS is platform 2
            if length < 24:
                raise ValueError("Truncated LC_BUILD_VERSION")
            platforms.append(struct.unpack_from("<I", data, offset + 8)[0])
        offset += length
    if offset != len(data) or platforms != [2]:
        raise ValueError("Executable was not built for device iOS")


def inspect(archive):
    app = archive / "Products/Applications/FlashLocal.app"
    info = plistlib.loads((app / "Info.plist").read_bytes())
    expected = {"CFBundleIdentifier": "com.dhafen.flashlocal", "CFBundleExecutable": "FlashLocal",
                "DTPlatformName": "iphoneos", "UIFileSharingEnabled": True,
                "LSSupportsOpeningDocumentsInPlace": True, "UIDeviceFamily": [1]}
    for key, value in expected.items():
        if info.get(key) != value:
            raise ValueError("Incorrect device archive metadata: " + key)
    device_executable(app / "FlashLocal")
    privacy_manifest(app / "PrivacyInfo.xcprivacy")
    prohibited = {".gguf", ".safetensors", ".p8", ".p12", ".pem", ".key", ".dpapi", ".mobileprovision"}
    total, count = 0, 0
    for item in app.rglob("*"):
        if not item.resolve().is_relative_to(app.resolve()):
            raise ValueError("Archive symlink escapes its app bundle")
        if item.is_file():
            if item.suffix.lower() in prohibited:
                raise ValueError("Model or signing material must not enter the unsigned app bundle")
            total += item.stat().st_size
            count += 1
    if total > 500_000_000:
        raise ValueError("Unexpected app size; model payload must remain external")
    license_digest = hashlib.sha256((app / "QwenModelLicense.txt").read_bytes()).hexdigest()
    if license_digest != "a0dc422560841fd68e06d974907f8b4c709bca44a67daad2b528437bdf676c08":
        raise ValueError("Original model license is missing or altered")
    notices = app / "OpenSourceNotices.txt"
    marker = "Ornith 1.5 publisher license (verbatim)\n\n"
    text = notices.read_text(encoding="utf-8") if notices.is_file() else ""
    if text.count(marker) != 1 or hashlib.sha256(text.split(marker)[-1].encode("utf-8")).hexdigest() != \
            "ba0b4cc44cd824e63b2aa20691ab8707ed7407a91b3772d8ed47896d8c4d9629":
        raise ValueError("Ornith publisher license is missing or altered")
    return {"bundleIdentifier": info["CFBundleIdentifier"], "platform": "iphoneos", "architecture": "arm64",
            "version": info.get("CFBundleShortVersionString"), "build": info.get("CFBundleVersion"),
            "appBytes": total, "files": count, "modelPayloadBundled": False,
            "privacyManifestVerified": True,
            "signed": False, "installedOnIPhone": False, "uploadedToTestFlight": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    result = inspect(args.archive)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
