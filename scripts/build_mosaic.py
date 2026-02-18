"""
Stage 4: build_mosaic.py
Replace every pixel in each Bad Apple frame with a horse tile chosen by
weighted-random selection from the TOP_K nearest matches.

Performance features:
- scipy cKDTree for O(log n) nearest-neighbor lookup
- All tile images pre-loaded into a single NumPy array
- Per-frame vectorised pixel matching + canvas construction (pure NumPy)
- Per-frame RNG seeded from frame number for reproducible variety
- Multiprocessing across frames
- Checkpoint/resume (skips already-generated output files)
"""

import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from tqdm import tqdm

TILE_DIR = Path("data/tiles")
INDEX_FILE = Path("data/tile_index.json")
FRAMES_DIR = Path("data/frames")
MOSAIC_DIR = Path("data/mosaic_frames")
TILE_SIZE = 12
FRAME_W, FRAME_H = 120, 90
OUT_W, OUT_H = FRAME_W * TILE_SIZE, FRAME_H * TILE_SIZE  # 1440 × 1080
JPEG_QUALITY = 95
NUM_WORKERS = max(1, mp.cpu_count() - 1)
TOP_K = 50    # candidates for weighted-random selection
BASE_SEED = 42


def load_tile_library() -> tuple[np.ndarray, np.ndarray]:
    """Returns (tile_pixels, avg_colors) where:
    - tile_pixels: (N, 12, 12, 3) uint8 array of all tile images
    - avg_colors:  (N, 3) float32 array of average RGB per tile
    """
    with open(INDEX_FILE) as f:
        index = json.load(f)

    if not index:
        raise ValueError(f"Tile index is empty. Run preprocess_tiles.py first.")

    tile_pixels = []
    avg_colors = []
    skipped = 0

    for entry in index:
        path = TILE_DIR / entry["filename"]
        if not path.exists():
            skipped += 1
            continue
        try:
            with Image.open(path) as img:
                arr = np.array(img.convert("RGB"), dtype=np.uint8)
        except Exception:
            skipped += 1
            continue
        if arr.shape != (TILE_SIZE, TILE_SIZE, 3):
            skipped += 1
            continue
        tile_pixels.append(arr)
        avg_colors.append(entry["avg_rgb"])

    if skipped:
        print(f"Warning: skipped {skipped} tiles (missing or corrupt).")
    if not tile_pixels:
        raise ValueError("No valid tiles loaded. Run preprocess_tiles.py first.")

    return np.array(tile_pixels, dtype=np.uint8), np.array(avg_colors, dtype=np.float32)


def build_frame(
    frame_path: Path,
    tile_pixels: np.ndarray,
    tree: cKDTree,
    out_dir: Path,
    top_k: int,
    base_seed: int,
) -> str:
    """Process a single frame using weighted-random tile selection."""
    out_path = out_dir / (frame_path.stem + ".jpg")
    if out_path.exists():
        return f"skip:{frame_path.name}"

    try:
        with Image.open(frame_path) as img:
            frame = np.array(img.convert("RGB"), dtype=np.float32)  # (90, 120, 3)
    except Exception as e:
        return f"error:{frame_path.name}:{e}"

    pixels = frame.reshape(-1, 3)                          # (10800, 3)
    k = min(top_k, len(tree.data))
    distances, indices = tree.query(pixels, k=k, workers=1)  # (10800, k)

    # Weighted-random selection: weight ∝ 1 / (distance + ε)
    eps = 1e-6
    weights = 1.0 / (distances + eps)                     # (10800, k)
    weights /= weights.sum(axis=1, keepdims=True)

    # Vectorised inverse-CDF sampling
    frame_num = int(frame_path.stem.split("_")[1])
    rng = np.random.default_rng(base_seed + frame_num)
    cumw = np.cumsum(weights, axis=1)                      # (10800, k)
    r = rng.random(len(pixels))[:, np.newaxis]             # (10800, 1)
    chosen_pos = np.clip((cumw < r).sum(axis=1), 0, k - 1)
    tile_indices = indices[np.arange(len(pixels)), chosen_pos]  # (10800,)

    # Vectorised canvas construction — no Python pixel loop
    idx_2d = tile_indices.reshape(FRAME_H, FRAME_W)
    canvas = (
        tile_pixels[idx_2d]             # (H, W, T, T, 3)
        .transpose(0, 2, 1, 3, 4)       # (H, T, W, T, 3)
        .reshape(OUT_H, OUT_W, 3)
    )

    Image.fromarray(canvas).save(out_path, "JPEG", quality=JPEG_QUALITY)
    return f"ok:{frame_path.name}"


def worker_init(tile_pixels_shared, avg_colors_shared, out_dir_str, top_k, base_seed):
    """Initializer for each worker process — loads shared data into globals."""
    global _tile_pixels, _tree, _out_dir, _top_k, _base_seed
    _tile_pixels = tile_pixels_shared
    _tree = cKDTree(avg_colors_shared)
    _out_dir = Path(out_dir_str)
    _top_k = top_k
    _base_seed = base_seed


def worker_task(frame_path_str: str) -> str:
    return build_frame(Path(frame_path_str), _tile_pixels, _tree, _out_dir, _top_k, _base_seed)


def main():
    MOSAIC_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading tile library...")
    tile_pixels, avg_colors = load_tile_library()
    print(f"Loaded {len(tile_pixels)} tiles. Strategy: weighted_random (top_k={TOP_K})")

    frame_paths = sorted(FRAMES_DIR.glob("frame_*.png"))
    if not frame_paths:
        print(f"No frames found in {FRAMES_DIR}. Run extract_frames.py first.")
        return

    already_done = len(list(MOSAIC_DIR.glob("frame_*.jpg")))
    print(f"Frames to process: {len(frame_paths)} total, {already_done} already done.")

    frame_strs = [str(p) for p in frame_paths]

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=NUM_WORKERS,
        initializer=worker_init,
        initargs=(tile_pixels, avg_colors, str(MOSAIC_DIR), TOP_K, BASE_SEED),
    ) as pool:
        results = list(
            tqdm(
                pool.imap(worker_task, frame_strs, chunksize=4),
                total=len(frame_strs),
                desc="Building mosaic",
                unit="frame",
            )
        )

    errors = [r for r in results if r.startswith("error:")]
    if errors:
        print(f"\n{len(errors)} errors:")
        for e in errors[:10]:
            print(f"  {e}")

    done = len(list(MOSAIC_DIR.glob("frame_*.jpg")))
    print(f"\nDone. {done} mosaic frames in {MOSAIC_DIR}.")


if __name__ == "__main__":
    main()
