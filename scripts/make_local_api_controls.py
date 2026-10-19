"""Build source-identical local HTTP controls without MLX, model or tokenizer."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

root = Path(__file__).resolve().parents[1]
destination = root / '.work/local-api-controls'
source = destination / 'Sources/MinirunKit'
tests = destination / 'Tests/MinirunKitTests'
source.mkdir(parents=True, exist_ok=True)
tests.mkdir(parents=True, exist_ok=True)
inputs = sorted((root/'Sources/MinirunKit/Serving').glob('*.swift'))
inputs.append(root/'Tests/MinirunKitTests/LocalModelAPITests.swift')
manifest = {}
for path in inputs:
    target = (tests if path.name.endswith('Tests.swift') else source) / path.name
    shutil.copyfile(path, target)
    manifest[str(path.relative_to(root)).replace('\\', '/')] = hashlib.sha256(path.read_bytes()).hexdigest()
(destination/'Package.swift').write_text('''// swift-tools-version: 5.9
import PackageDescription
let package = Package(name: "LocalAPIControls", platforms: [.macOS(.v14), .iOS("18.0")],
    products: [.library(name: "MinirunKit", targets: ["MinirunKit"])],
    targets: [.target(name: "MinirunKit"), .testTarget(name: "MinirunKitTests", dependencies: ["MinirunKit"])])
''')
(destination/'source-manifest.json').write_text(json.dumps(dict(
    sourceRevision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
    inputs=manifest, modelLoaded=False, phoneReachabilityVerified=False), indent=2)+'\n')
print(json.dumps(dict(files=len(inputs), sourceIdentical=True, modelLoaded=False)))
