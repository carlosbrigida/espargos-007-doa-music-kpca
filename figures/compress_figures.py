from pathlib import Path
from PIL import Image

SRC = Path("figures")
DST = Path("figures_overleaf")
DST.mkdir(exist_ok=True)

for file in SRC.glob("*.png"):
    img = Image.open(file).convert("RGB")
    img.thumbnail((1200, 800))

    out = DST / file.name
    img.save(out, optimize=True, quality=85)

    print(f"Saved: {out}")