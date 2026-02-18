"""
filter_tiles.py
Curate the tile library by removing tiles that are too close to pure black or
pure white and selecting 1000 tiles with even brightness distribution from
the remainder.

What this does:
  1. Load data/tile_index.json
  2. Compute perceptual brightness for every tile
  3. Drop tiles whose brightness < DARK_THRESHOLD or > BRIGHT_THRESHOLD
  4. From the remaining tiles, pick up to TARGET_COUNT (1000) with even
     spread across the brightness range (bucket sampling)
  5. Delete the discarded tiles from data/tiles/ and data/raw_horses/
  6. Rewrite data/tile_index.json with only the kept tiles

Run:
    python scripts/filter_tiles.py [--dry-run]
"""

import argparse
import json
from pathlib import Path

import numpy as np

TILE_DIR = Path("data/tiles")
RAW_DIR = Path("data/raw_horses")
INDEX_FILE = Path("data/tile_index.json")

# Brightness thresholds (perceptual, 0–255)
# Tiles outside this range are considered too close to black/white.
DARK_THRESHOLD = 30     # below → near-black, excluded
BRIGHT_THRESHOLD = 225  # above → near-white, excluded

TARGET_COUNT = 1000
NUM_BUCKETS = 50        # evenly-spaced brightness buckets for fair sampling


def perceptual_brightness(avg_rgb: list[int]) -> float:
    r, g, b = avg_rgb
    return 0.299 * r + 0.587 * g + 0.114 * b


def bucket_sample(tiles: list[dict], brightnesses: list[float], n: int, num_buckets: int) -> list[dict]:
    """Select up to n tiles spread evenly across brightness buckets."""
    # Assign each tile to a bucket
    buckets: list[list[int]] = [[] for _ in range(num_buckets)]
    for i, b in enumerate(brightnesses):
        idx = min(int(b / 255.0 * num_buckets), num_buckets - 1)
        buckets[idx].append(i)

    # Fill buckets with order: we spread the quota evenly but give overflow to buckets
    # that have more tiles.
    quota = max(1, n // num_buckets)
    selected_indices: list[int] = []

    # First pass: take up to quota from each bucket (preserving brightness order within)
    remainder: list[tuple[int, list[int]]] = []
    for bi, bucket in enumerate(buckets):
        take = min(quota, len(bucket))
        selected_indices.extend(bucket[:take])
        leftover = len(bucket) - take
        if leftover > 0:
            remainder.append((bi, bucket[take:]))

    # Second pass: fill up to n from buckets that still have tiles, largest first
    still_needed = n - len(selected_indices)
    if still_needed > 0:
        all_extras = [(idx, bi) for bi, bucket in remainder for idx in bucket]
        # Sort extras by brightness for determinism
        all_extras.sort(key=lambda t: brightnesses[t[0]])
        step = max(1, len(all_extras) // still_needed)
        for i in range(0, len(all_extras), step):
            if len(selected_indices) >= n:
                break
            selected_indices.append(all_extras[i][0])

    return [tiles[i] for i in sorted(set(selected_indices))]


def delete_file(path: Path, dry_run: bool):
    if dry_run:
        print(f"  [dry-run] would delete: {path}")
    else:
        if path.exists():
            path.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be deleted without deleting anything")
    args = parser.parse_args()

    if not INDEX_FILE.exists():
        print(f"ERROR: {INDEX_FILE} not found. Run preprocess_tiles.py first.")
        return

    with open(INDEX_FILE) as f:
        all_tiles: list[dict] = json.load(f)

    print(f"Loaded {len(all_tiles)} tiles from {INDEX_FILE}")

    # Compute brightness for every tile
    brightnesses = [perceptual_brightness(t["avg_rgb"]) for t in all_tiles]
    b_arr = np.array(brightnesses)

    # ── Step 1: Filter by brightness thresholds ──────────────────────────────
    mid_mask = (b_arr >= DARK_THRESHOLD) & (b_arr <= BRIGHT_THRESHOLD)
    mid_tiles = [t for t, m in zip(all_tiles, mid_mask) if m]
    mid_bright = [b for b, m in zip(brightnesses, mid_mask) if m]

    excluded_by_threshold = [t for t, m in zip(all_tiles, mid_mask) if not m]
    print(f"\nBrightness filter  ({DARK_THRESHOLD}–{BRIGHT_THRESHOLD}):")
    print(f"  Kept   : {len(mid_tiles)}")
    print(f"  Removed: {len(excluded_by_threshold)} (too dark or too bright)")

    # ── Step 2: Bucket-sample down to TARGET_COUNT ────────────────────────────
    if len(mid_tiles) > TARGET_COUNT:
        kept_tiles = bucket_sample(mid_tiles, mid_bright, TARGET_COUNT, NUM_BUCKETS)
        excluded_by_sample = [t for t in mid_tiles if t not in kept_tiles]
    else:
        kept_tiles = mid_tiles
        excluded_by_sample = []

    print(f"\nBucket sampling to {TARGET_COUNT}:")
    print(f"  Kept   : {len(kept_tiles)}")
    print(f"  Removed: {len(excluded_by_sample)} (excess tiles)")

    # ── Step 3: Brightness distribution of kept tiles ─────────────────────────
    kept_b = np.array([perceptual_brightness(t["avg_rgb"]) for t in kept_tiles])
    print(f"\nKept tile brightness stats:")
    print(f"  Min   : {kept_b.min():.1f}")
    print(f"  Max   : {kept_b.max():.1f}")
    print(f"  Mean  : {kept_b.mean():.1f}")
    print(f"  Median: {np.median(kept_b):.1f}")

    # ── Step 4: Delete discarded files ────────────────────────────────────────
    to_delete = excluded_by_threshold + excluded_by_sample
    kept_names = {t["filename"] for t in kept_tiles}

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Deleting {len(to_delete)} discarded tiles...")
    deleted_tiles = 0
    deleted_raw = 0
    missing_raw = 0

    for tile in to_delete:
        fname = tile["filename"]
        stem = Path(fname).stem

        tile_path = TILE_DIR / fname
        delete_file(tile_path, args.dry_run)
        deleted_tiles += 1

        # Raw image may have .jpg, .jpeg, or .png extension
        deleted_this_raw = False
        for ext in (".jpg", ".jpeg", ".png"):
            raw_path = RAW_DIR / (stem + ext)
            if raw_path.exists():
                delete_file(raw_path, args.dry_run)
                deleted_raw += 1
                deleted_this_raw = True
                break
        if not deleted_this_raw:
            missing_raw += 1

    print(f"  Tile files deleted : {deleted_tiles}")
    print(f"  Raw files deleted  : {deleted_raw}")
    if missing_raw:
        print(f"  Raw files not found: {missing_raw} (already missing, OK)")

    # ── Step 5: Rewrite tile_index.json ───────────────────────────────────────
    if args.dry_run:
        print(f"\n[dry-run] Would rewrite {INDEX_FILE} with {len(kept_tiles)} entries.")
    else:
        with open(INDEX_FILE, "w") as f:
            json.dump(kept_tiles, f, indent=2)
        print(f"\nRewritten {INDEX_FILE} with {len(kept_tiles)} tiles.")

    print("\nDone.")
    if args.dry_run:
        print("Re-run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
