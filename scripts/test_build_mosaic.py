"""
test_build_mosaic.py
Build Bad Apple horse mosaic frames using one of 6 selectable strategies.

Usage:
    python scripts/test_build_mosaic.py --strategy nearest --test-frame 3000
    python scripts/test_build_mosaic.py --strategy dither_bucket --test
    python scripts/test_build_mosaic.py --strategy dither_weighted
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from tqdm import tqdm

# ── Constants ──────────────────────────────────────────────────────────────────

TILE_DIR = Path("data/tiles")
INDEX_FILE = Path("data/tile_index.json")
FRAMES_DIR = Path("data/frames")
MOSAIC_BASE_DIR = Path("data/mosaic_frames")
TILE_SIZE = 12
FRAME_W, FRAME_H = 120, 90
OUT_W, OUT_H = FRAME_W * TILE_SIZE, FRAME_H * TILE_SIZE  # 1440 × 1080
JPEG_QUALITY = 95
TEST_FRAMES = [1000, 2000, 3000, 4000]

BAYER_8x8 = np.array([
    [ 0, 32,  8, 40,  2, 34, 10, 42],
    [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44,  4, 36, 14, 46,  6, 38],
    [60, 28, 52, 20, 62, 30, 54, 22],
    [ 3, 35, 11, 43,  1, 33,  9, 41],
    [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47,  7, 39, 13, 45,  5, 37],
    [63, 31, 55, 23, 61, 29, 53, 21],
], dtype=np.float32) / 63.0 - 0.5  # normalised to [-0.5, 0.5]


# ── Tile library ───────────────────────────────────────────────────────────────

def load_tile_library():
    """Returns (tile_pixels, avg_colors, filenames).

    tile_pixels : (N, 12, 12, 3) uint8
    avg_colors  : (N, 3)         float32
    filenames   : list[str]
    """
    with open(INDEX_FILE) as f:
        index = json.load(f)

    tile_pixels, avg_colors, filenames = [], [], []
    for entry in index:
        path = TILE_DIR / entry["filename"]
        if not path.exists():
            continue
        try:
            with Image.open(path) as img:
                arr = np.array(img.convert("RGB"), dtype=np.uint8)
        except Exception:
            continue
        if arr.shape != (TILE_SIZE, TILE_SIZE, 3):
            continue
        tile_pixels.append(arr)
        avg_colors.append(entry["avg_rgb"])
        filenames.append(entry["filename"])

    if not tile_pixels:
        raise ValueError("No valid tiles loaded. Run preprocess_tiles.py first.")

    return (
        np.array(tile_pixels, dtype=np.uint8),
        np.array(avg_colors, dtype=np.float32),
        filenames,
    )


def build_buckets(avg_colors: np.ndarray, bucket_size: int):
    """Partition tiles into brightness buckets.

    Returns:
        buckets  : list of lists of tile indices (length = bucket_size)
        fallback : int numpy array; fallback[i] = nearest non-empty bucket index,
                   or -1 if no tiles exist at all.
    """
    brightness = (
        0.299 * avg_colors[:, 0]
        + 0.587 * avg_colors[:, 1]
        + 0.114 * avg_colors[:, 2]
    )
    buckets: list[list[int]] = [[] for _ in range(bucket_size)]
    for i, b in enumerate(brightness):
        idx = min(int(b / 255.0 * bucket_size), bucket_size - 1)
        buckets[idx].append(i)

    fallback = np.full(bucket_size, -1, dtype=np.intp)
    for i in range(bucket_size):
        if buckets[i]:
            fallback[i] = i
            continue
        for d in range(1, bucket_size):
            lo, hi = i - d, i + d
            if lo >= 0 and buckets[lo]:
                fallback[i] = lo
                break
            if hi < bucket_size and buckets[hi]:
                fallback[i] = hi
                break

    return buckets, fallback


# ── Canvas builder (vectorised) ────────────────────────────────────────────────

def indices_to_image(indices: np.ndarray, tile_pixels: np.ndarray) -> Image.Image:
    """Convert a (FRAME_H, FRAME_W) tile-index array to a 1440×1080 PIL Image.

    Uses fully vectorised NumPy operations — no Python pixel loop.
    """
    selected = tile_pixels[indices]          # (H, W, T, T, 3)
    canvas = (
        selected
        .transpose(0, 2, 1, 3, 4)           # (H, T, W, T, 3)
        .reshape(OUT_H, OUT_W, 3)
    )
    return Image.fromarray(canvas)


# ── Dithering helpers ──────────────────────────────────────────────────────────

def ordered_dither_frame(frame: np.ndarray, spread: float = 128.0) -> np.ndarray:
    """Return float32 (H, W, 3) frame adjusted by Bayer 8×8 matrix."""
    bayer = np.tile(
        BAYER_8x8,
        ((FRAME_H + 7) // 8, (FRAME_W + 7) // 8),
    )[:FRAME_H, :FRAME_W]                   # (H, W)
    adjusted = frame.astype(np.float32) + bayer[:, :, np.newaxis] * spread
    return np.clip(adjusted, 0.0, 255.0)


def floyd_steinberg_indices(frame: np.ndarray, tree: cKDTree) -> np.ndarray:
    """Floyd-Steinberg dithering → tile index array (H, W).

    Quantises to the mosaic tile palette via error diffusion.
    Processed sequentially (raster order) — parallelism is at frame level.
    """
    buf = frame.astype(np.float32).copy()
    result = np.zeros((FRAME_H, FRAME_W), dtype=np.int32)

    for y in range(FRAME_H):
        for x in range(FRAME_W):
            pixel = np.clip(buf[y, x], 0.0, 255.0)
            _, tile_idx = tree.query(pixel)
            result[y, x] = tile_idx

            error = pixel - tree.data[tile_idx]
            if x + 1 < FRAME_W:
                buf[y, x + 1]     += error * (7 / 16)
            if y + 1 < FRAME_H:
                if x > 0:
                    buf[y + 1, x - 1] += error * (3 / 16)
                buf[y + 1, x]     += error * (5 / 16)
                if x + 1 < FRAME_W:
                    buf[y + 1, x + 1] += error * (1 / 16)

    return result


def floyd_steinberg_adjusted(frame: np.ndarray, tree: cKDTree) -> np.ndarray:
    """Run Floyd-Steinberg and return the per-pixel error-adjusted values
    (pre-quantisation, clamped to [0, 255]) as float32 (H, W, 3).

    Used by dither_bucket and dither_weighted to get a dithered colour image
    which is then fed into bucket/weighted selection instead of nearest.
    """
    buf = frame.astype(np.float32).copy()
    adjusted_out = np.empty((FRAME_H, FRAME_W, 3), dtype=np.float32)

    for y in range(FRAME_H):
        for x in range(FRAME_W):
            pixel = np.clip(buf[y, x], 0.0, 255.0)
            adjusted_out[y, x] = pixel

            _, tile_idx = tree.query(pixel)
            error = pixel - tree.data[tile_idx]

            if x + 1 < FRAME_W:
                buf[y, x + 1]     += error * (7 / 16)
            if y + 1 < FRAME_H:
                if x > 0:
                    buf[y + 1, x - 1] += error * (3 / 16)
                buf[y + 1, x]     += error * (5 / 16)
                if x + 1 < FRAME_W:
                    buf[y + 1, x + 1] += error * (1 / 16)

    return adjusted_out


# ── Selection helpers (vectorised) ─────────────────────────────────────────────

def bucket_select(
    adjusted: np.ndarray,
    buckets: list,
    fallback: np.ndarray,
    bucket_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select tiles by brightness bucket. Returns index array (FRAME_H, FRAME_W)."""
    flat = adjusted.reshape(-1, 3)
    brightness = 0.299 * flat[:, 0] + 0.587 * flat[:, 1] + 0.114 * flat[:, 2]
    bi_arr = np.clip((brightness / 255.0 * bucket_size).astype(int), 0, bucket_size - 1)
    resolved = fallback[bi_arr]  # vectorised fallback lookup

    result = np.zeros(len(flat), dtype=np.int32)
    for bi in np.unique(resolved):
        if bi < 0:
            continue
        mask = resolved == bi
        tiles = np.array(buckets[bi], dtype=np.int32)
        picks = rng.integers(0, len(tiles), size=int(mask.sum()))
        result[mask] = tiles[picks]

    return result.reshape(FRAME_H, FRAME_W)


def weighted_select(
    adjusted: np.ndarray,
    tree: cKDTree,
    top_k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Top-K inverse-distance weighted random selection. Returns (FRAME_H, FRAME_W)."""
    flat = adjusted.reshape(-1, 3)
    k = min(top_k, len(tree.data))
    distances, indices = tree.query(flat, k=k)

    if k == 1:
        return indices.reshape(FRAME_H, FRAME_W)

    eps = 1e-6
    weights = 1.0 / (distances + eps)          # (N, k)
    weights /= weights.sum(axis=1, keepdims=True)

    # Vectorised inverse-CDF sampling
    cumw = np.cumsum(weights, axis=1)           # (N, k)
    r = rng.random(len(flat))[:, np.newaxis]    # (N, 1)
    chosen_pos = np.clip((cumw < r).sum(axis=1), 0, k - 1)
    result = indices[np.arange(len(flat)), chosen_pos]

    return result.reshape(FRAME_H, FRAME_W)


# ── Strategy dispatch ──────────────────────────────────────────────────────────

def run_strategy(
    frame: np.ndarray,
    strategy: str,
    tree: cKDTree,
    buckets: list,
    fallback: np.ndarray,
    bucket_size: int,
    top_k: int,
    dither_method: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return tile index array (FRAME_H, FRAME_W) using the chosen strategy."""

    if strategy == "nearest":
        pixels = frame.reshape(-1, 3).astype(np.float32)
        _, indices = tree.query(pixels)
        return indices.reshape(FRAME_H, FRAME_W)

    elif strategy == "bucket_random":
        return bucket_select(frame.astype(np.float32), buckets, fallback, bucket_size, rng)

    elif strategy == "weighted_random":
        return weighted_select(frame.astype(np.float32), tree, top_k, rng)

    elif strategy == "dither_nearest":
        if dither_method == "ordered":
            adjusted = ordered_dither_frame(frame)
            _, indices = tree.query(adjusted.reshape(-1, 3))
            return indices.reshape(FRAME_H, FRAME_W)
        else:
            return floyd_steinberg_indices(frame, tree)

    elif strategy == "dither_bucket":
        if dither_method == "ordered":
            adjusted = ordered_dither_frame(frame)
        else:
            adjusted = floyd_steinberg_adjusted(frame, tree)
        return bucket_select(adjusted, buckets, fallback, bucket_size, rng)

    elif strategy == "dither_weighted":
        if dither_method == "ordered":
            adjusted = ordered_dither_frame(frame)
        else:
            adjusted = floyd_steinberg_adjusted(frame, tree)
        return weighted_select(adjusted, tree, top_k, rng)

    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")


# ── Frame processing ───────────────────────────────────────────────────────────

def process_frame(
    frame_num: int,
    strategy: str,
    out_dir: Path,
    tile_pixels: np.ndarray,
    tree: cKDTree,
    buckets: list,
    fallback: np.ndarray,
    args: argparse.Namespace,
) -> str:
    frame_path = FRAMES_DIR / f"frame_{frame_num:05d}.png"
    if not frame_path.exists():
        return f"missing:{frame_num}"

    out_path = out_dir / f"frame_{frame_num:05d}.jpg"
    if out_path.exists():
        return f"skip:{frame_num}"

    with Image.open(frame_path) as img:
        frame = np.array(img.convert("RGB"), dtype=np.float32)

    rng = np.random.default_rng(args.seed + frame_num)

    indices = run_strategy(
        frame, strategy, tree,
        buckets, fallback, args.bucket_size,
        args.top_k, args.dither_method, rng,
    )

    mosaic = indices_to_image(indices, tile_pixels)
    mosaic.save(out_path, "JPEG", quality=JPEG_QUALITY)
    return f"ok:{frame_num}"


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Bad Apple horse mosaic frames.")
    p.add_argument(
        "--strategy", required=True,
        choices=["nearest", "bucket_random", "weighted_random",
                 "dither_nearest", "dither_bucket", "dither_weighted"],
    )
    p.add_argument(
        "--test", action="store_true",
        help=f"Process only test frames: {TEST_FRAMES}",
    )
    p.add_argument(
        "--test-frame", type=int, metavar="N",
        help="Process only this single frame number",
    )
    p.add_argument("--bucket-size", type=int, default=32,
                   help="Number of brightness buckets (default: 32)")
    p.add_argument("--top-k", type=int, default=10,
                   help="Candidates for weighted random selection (default: 10)")
    p.add_argument("--dither-method", choices=["floyd_steinberg", "ordered"],
                   default="floyd_steinberg")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    out_dir = MOSAIC_BASE_DIR / args.strategy
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Strategy : {args.strategy}")
    if args.strategy in ("dither_nearest", "dither_bucket", "dither_weighted"):
        print(f"Dithering: {args.dither_method}")
    print("Loading tile library...")

    tile_pixels, avg_colors, _ = load_tile_library()
    tree = cKDTree(avg_colors)
    buckets, fallback = build_buckets(avg_colors, args.bucket_size)
    print(f"Loaded {len(tile_pixels)} tiles into {args.bucket_size} brightness buckets.")

    # Determine frames to process
    if args.test_frame is not None:
        frame_nums = [args.test_frame]
    elif args.test:
        frame_nums = TEST_FRAMES
    else:
        all_frames = sorted(FRAMES_DIR.glob("frame_*.png"))
        if not all_frames:
            print(f"No frames found in {FRAMES_DIR}. Run extract_frames.py first.")
            sys.exit(1)
        frame_nums = [int(p.stem.split("_")[1]) for p in all_frames]

    print(f"Processing {len(frame_nums)} frame(s) → {out_dir}\n")

    ok = skipped = errors = 0
    for fn in tqdm(frame_nums, desc=args.strategy, unit="frame"):
        result = process_frame(
            fn, args.strategy, out_dir, tile_pixels, tree, buckets, fallback, args
        )
        if result.startswith("ok"):
            ok += 1
        elif result.startswith("skip"):
            skipped += 1
        else:
            errors += 1
            tqdm.write(f"  {result}")

    print(f"\nDone. {ok} built, {skipped} skipped, {errors} errors.")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
