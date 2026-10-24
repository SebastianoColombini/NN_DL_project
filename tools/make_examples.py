"""Render the example figure of the README: image, human fixation map and the maps of each model.

Usage:
    python tools/make_examples.py --out docs/img/examples.jpg

Overlays use the same rendering as the GUI (jet colormap, map stretched to its own maximum,
alpha following the saliency, 70% opacity). The CC under each map is computed against the
fixation map with the notebook metric, at the model resolution.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from saliency.inference import Predictor  # noqa: E402
from saliency.registry import REGISTRY  # noqa: E402

DEFAULT_SAMPLES = ["i1784306863", "i118680869", "i05june05_static_street_boston_p1010764"]
CELL_W = 320
PAD = 8
HEADER = 34
CAPTION = 26
OPACITY = 0.7


def jet(x):
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def overlay(img: Image.Image, sal: np.ndarray) -> Image.Image:
    """Saliency-weighted overlay, as in app/static/js/colormap.js."""
    s = np.round(sal / sal.max() * 255) / 255 if sal.max() > 0 else sal
    alpha = np.minimum(1.0, np.power(s, 0.8) * 1.6)[..., None] * OPACITY
    base = np.asarray(img.convert("RGB"), dtype=np.float32) / 255
    out = base * (1 - alpha) + jet(s) * alpha
    return Image.fromarray(np.uint8(np.round(out * 255)))


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "docs/img/examples.jpg"))
    ap.add_argument("--samples", nargs="+", default=DEFAULT_SAMPLES, help="sample ids from samples/")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    predictor = Predictor(device=args.device)
    models = list(REGISTRY)
    titles = ["Image", "Human fixations"] + [REGISTRY[m].label for m in models]
    ncols = len(titles)

    rows = []
    for sid in args.samples:
        img = Image.open(ROOT / "samples" / f"{sid}.jpeg").convert("RGB")
        gt = Image.open(ROOT / "samples" / f"{sid}_fixMap.jpg").convert("L")
        cells = [(img, ""), (overlay(img, np.asarray(gt, dtype=np.float32) / 255), "")]
        for m in models:
            pred = predictor.predict(img, m)
            cc = predictor.metrics(pred, gt)["cc"]
            cells.append((overlay(img, pred.map_full), f"CC {cc:.2f}"))
        h = round(CELL_W * img.height / img.width)
        rows.append([(c.resize((CELL_W, h), Image.LANCZOS), cap) for c, cap in cells])

    width = PAD + ncols * (CELL_W + PAD)
    height = HEADER + sum(r[0][0].height + CAPTION for r in rows) + PAD
    fig = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(fig)
    f_head, f_cap = font(17), font(15)
    for i, t in enumerate(titles):
        draw.text((PAD + i * (CELL_W + PAD) + CELL_W / 2, HEADER / 2), t, fill="#222", font=f_head, anchor="mm")
    y = HEADER
    for row in rows:
        for i, (cell, cap) in enumerate(row):
            x = PAD + i * (CELL_W + PAD)
            fig.paste(cell, (x, y))
            if cap:
                draw.text((x + CELL_W / 2, y + cell.height + CAPTION / 2), cap, fill="#444", font=f_cap, anchor="mm")
        y += row[0][0].height + CAPTION

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.save(out, quality=88, optimize=True)
    print(f"wrote {out} ({width}x{height}, {out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
