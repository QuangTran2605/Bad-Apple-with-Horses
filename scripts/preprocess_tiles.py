"""
Stage 2: preprocess_tiles.py
Center-crop, resize all raw horse images to 12×12 tiles, compute average RGB.
"""

import json
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

RAW_DIR = Path("data/raw_horses")
TILE_DIR = Path("data/tiles")
INDEX_FILE = Path("data/tile_index.json")
TILE_SIZE = 12


def center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def avg_rgb(img: Image.Image) -> list[int]:
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    mean = arr.mean(axis=(0, 1))
    return [int(round(v)) for v in mean]


def main():
    TILE_DIR.mkdir(parents=True, exist_ok=True)

    raw_files = list(RAW_DIR.glob("*.jpg")) + list(RAW_DIR.glob("*.png")) + list(RAW_DIR.glob("*.jpeg"))
    if not raw_files:
        print(f"No images found in {RAW_DIR}. Run scrape_horses.py first.")
        return

    # Load existing index to allow resume
    if INDEX_FILE.exists():
        with open(INDEX_FILE) as f:
            existing = json.load(f)
        index = {entry["filename"]: entry for entry in existing}
    else:
        index = {}

    print(f"Found {len(raw_files)} raw images. Already processed: {len(index)}.")

    for raw_path in tqdm(raw_files, desc="Processing tiles"):
        out_name = raw_path.stem + ".jpg"
        if out_name in index:
            continue  # already processed

        try:
            with Image.open(raw_path) as img:
                img = img.convert("RGB")
                img = center_crop_square(img)
                img = img.resize((TILE_SIZE, TILE_SIZE), Image.LANCZOS)
                out_path = TILE_DIR / out_name
                img.save(out_path, "JPEG", quality=95)
                color = avg_rgb(img)
        except Exception as e:
            print(f"\nSkipping {raw_path.name}: {e}")
            continue

        index[out_name] = {"filename": out_name, "avg_rgb": color}

    tile_list = list(index.values())
    with open(INDEX_FILE, "w") as f:
        json.dump(tile_list, f, indent=2)

    print(f"\nDone. {len(tile_list)} tiles saved to {TILE_DIR}, index at {INDEX_FILE}.")


if __name__ == "__main__":
    main()
