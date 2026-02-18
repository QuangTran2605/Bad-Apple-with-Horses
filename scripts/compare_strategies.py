"""
compare_strategies.py
Run all 6 mosaic strategies on the test frames and produce side-by-side
comparison grids so you can pick the best strategy visually.

Usage:
    python scripts/compare_strategies.py

Output: data/comparisons/comparison_frame_XXXXX.png  (one per test frame)

Grid layout (4 cols × 2 rows):
    [ original ] [ nearest       ] [ bucket_random  ] [ weighted_random ]
    [ dither_nearest ] [ dither_bucket ] [ dither_weighted ] [ —— ]
"""

import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

STRATEGIES = [
    "nearest",
    "bucket_random",
    "weighted_random",
    "dither_nearest",
    "dither_bucket",
    "dither_weighted",
]
TEST_FRAMES = [1000, 2000, 3000, 4000]

FRAMES_DIR = Path("data/frames")
MOSAIC_BASE_DIR = Path("data/mosaic_frames")
COMPARISONS_DIR = Path("data/comparisons")

# Grid dimensions
COLS, ROWS = 4, 2
PANEL_W, PANEL_H = 480, 360   # each mosaic thumbnail
LABEL_H = 30                   # label bar height below each panel
BG_COLOR = (15, 15, 15)
LABEL_BG = (35, 35, 35)
LABEL_FG = (210, 210, 210)


# ── Helpers ────────────────────────────────────────────────────────────────────

def run_strategy(strategy: str):
    """Call test_build_mosaic.py for this strategy with --test."""
    print(f"\n{'─' * 56}")
    print(f"  Running: {strategy}")
    print(f"{'─' * 56}")
    result = subprocess.run(
        [sys.executable, "scripts/test_build_mosaic.py",
         "--strategy", strategy, "--test"],
    )
    if result.returncode != 0:
        print(f"  WARNING: '{strategy}' exited with code {result.returncode}")


def load_thumbnail(path: Path) -> Image.Image:
    if path.exists():
        with Image.open(path) as img:
            return img.resize((PANEL_W, PANEL_H), Image.LANCZOS).convert("RGB")
    # Grey placeholder with centred text
    ph = Image.new("RGB", (PANEL_W, PANEL_H), (45, 45, 45))
    draw = ImageDraw.Draw(ph)
    draw.text((10, PANEL_H // 2 - 8), "[not generated]", fill=(120, 120, 120))
    return ph


def labeled_panel(img: Image.Image, text: str) -> Image.Image:
    """Return the panel with a label bar underneath."""
    cell = Image.new("RGB", (PANEL_W, PANEL_H + LABEL_H), LABEL_BG)
    cell.paste(img, (0, 0))
    draw = ImageDraw.Draw(cell)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(
        ((PANEL_W - tw) // 2, PANEL_H + (LABEL_H - th) // 2),
        text, fill=LABEL_FG, font=font,
    )
    return cell


def make_comparison(frame_num: int) -> Image.Image:
    cell_h = PANEL_H + LABEL_H
    canvas = Image.new("RGB", (COLS * PANEL_W, ROWS * cell_h), BG_COLOR)

    # Build ordered list: original first, then the 6 strategies
    panels: list[tuple[str, Image.Image]] = []

    orig_path = FRAMES_DIR / f"frame_{frame_num:05d}.png"
    if orig_path.exists():
        with Image.open(orig_path) as img:
            orig = img.resize((PANEL_W, PANEL_H), Image.NEAREST).convert("RGB")
    else:
        orig = Image.new("RGB", (PANEL_W, PANEL_H), (60, 0, 0))
    panels.append(("original", orig))

    for strategy in STRATEGIES:
        path = MOSAIC_BASE_DIR / strategy / f"frame_{frame_num:05d}.jpg"
        panels.append((strategy, load_thumbnail(path)))

    # Paste into 4×2 grid (last slot left blank if 7 panels)
    for i, (label, img) in enumerate(panels):
        row, col = divmod(i, COLS)
        cell = labeled_panel(img, label)
        canvas.paste(cell, (col * PANEL_W, row * cell_h))

    return canvas


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    COMPARISONS_DIR.mkdir(parents=True, exist_ok=True)

    # Run all strategies on test frames
    for strategy in STRATEGIES:
        run_strategy(strategy)

    # Build comparison images
    print(f"\n{'=' * 56}")
    print("Building comparison grids...")
    for frame_num in TEST_FRAMES:
        comp = make_comparison(frame_num)
        out = COMPARISONS_DIR / f"comparison_frame_{frame_num:05d}.png"
        comp.save(out, "PNG")
        print(f"  Saved {out}")

    print(f"\nAll done. Open data/comparisons/ to review.")


if __name__ == "__main__":
    main()
