"""Render the fork's original geometric vector mark into required app assets.

No upstream image is opened or transformed. Pillow is a build-time dependency
only; geometry is also saved as SVG for review/editing.
"""
import json
import pathlib
from PIL import Image, ImageDraw

ROOT = pathlib.Path(__file__).resolve().parents[1]
ASSETS = ROOT / "Apps/Minirun/Resources/Assets.xcassets"
POLYGON = [(300, 240), (720, 240), (720, 350), (410, 350), (410, 465),
           (650, 465), (650, 575), (410, 575), (410, 784), (300, 784)]


def render(size):
    factor = 4
    canvas = Image.new("RGB", (size * factor, size * factor), "#0B141C")
    draw = ImageDraw.Draw(canvas)
    scale = size * factor / 1024
    draw.polygon([(round(x * scale), round(y * scale)) for x, y in POLYGON], fill="#7EE8B5")
    draw.ellipse(tuple(round(v * scale) for v in [700, 680, 808, 788]), fill="#FFC65A")
    return canvas.resize((size, size), Image.Resampling.LANCZOS)


def main():
    icon = ASSETS / "AppIcon.appiconset"
    contents = json.loads((icon / "Contents.json").read_text())
    for item in contents["images"]:
        if "filename" not in item:
            continue
        size = int(float(item["size"].split("x")[0]) * float(item.get("scale", "1x")[:-1]))
        render(size).save(icon / item["filename"])
    logo = ASSETS / "MinirunLogo.imageset"
    for item in json.loads((logo / "Contents.json").read_text())["images"]:
        render(128 * int(item["scale"][:-1])).save(logo / item["filename"])
    points = " ".join(f"{x},{y}" for x, y in POLYGON)
    (ROOT / "Apps/Minirun/Resources/FlashLocalMark.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024">'
        '<title>Flash Local</title><rect width="1024" height="1024" fill="#0B141C"/>'
        f'<polygon points="{points}" fill="#7EE8B5"/>'
        '<circle cx="754" cy="734" r="54" fill="#FFC65A"/></svg>\n')
    print("Rendered original Flash Local mark; upstream artwork was not used")


if __name__ == "__main__":
    main()
