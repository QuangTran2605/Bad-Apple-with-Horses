## Context

Read the project `README.md` first for full project context. This file provides instructions for implementing **multiple mosaic building strategies** so we can visually compare them and pick the best one.

## The Problem

Bad Apple is essentially a black-and-white video. A naive "pick the closest color match" algorithm will repeatedly select the same 1-2 horse images for black and white pixels, producing a boring, repetitive mosaic. We need strategies that introduce **visual variety** while preserving the overall recognizability of each frame.

## What to Build

Create a file `scripts/test_build_mosaic.py` that supports **6 different strategies**, selectable via a command-line argument. The script should also support a **test mode** that processes only a small number of frames for quick visual comparison.

### Usage

```bash
# Process a single test frame with a specific strategy
python scripts/test_build_mosaic.py --strategy nearest --test-frame 3000

# Process a small batch of test frames (e.g., 5 frames spread across the video)
python scripts/test_build_mosaic.py --strategy dither_bucket --test

# Process all frames for final output
python scripts/test_build_mosaic.py --strategy dither_bucket
```

### Command-Line Arguments

| Argument | Description |
|---|---|
| `--strategy` | **Required.** One of: `nearest`, `bucket_random`, `weighted_random`, `dither_nearest`, `dither_bucket`, `dither_weighted` |
| `--test` | Process only 5 representative frames (frame 500, 1500, 3000, 4500, 6000) for quick comparison |
| `--test-frame N` | Process only frame number N |
| `--bucket-size` | Number of brightness/color buckets (default: 32). Applies to bucket strategies |
| `--top-k` | Number of top candidates for weighted random (default: 10). Applies to weighted strategies |
| `--dither-method` | Dithering algorithm: `floyd_steinberg` or `ordered` (default: `floyd_steinberg`). Applies to dither strategies |
| `--seed` | Random seed for reproducibility (default: 42) |

### Output Directory Structure

Each strategy writes to its own subfolder for easy comparison:

```
data/mosaic_frames/nearest/frame_00001.jpg
data/mosaic_frames/bucket_random/frame_00001.jpg
data/mosaic_frames/dither_nearest/frame_00001.jpg
...
```

---

## The 6 Strategies

All strategies share the same setup:
1. Load tile library from `data/tile_index.json` and all tile images from `data/tiles/`
2. Load the source frame (120×90) from `data/frames/`
3. For each pixel, select a tile using the strategy-specific method
4. Place the selected tile (12×12px) onto the output canvas (1440×1080)
5. Save the result as JPEG (quality 95)

Below are the 6 strategies. Each one describes how to go from a pixel's color to a selected tile.

---

### Strategy 1: `nearest` (Baseline — Closest Match)

The simplest approach. For each pixel, find the tile whose average RGB is closest by Euclidean distance.

**Algorithm:**
1. Build a KD-Tree (scipy.spatial.cKDTree) from all tile average RGB values
2. For each pixel, query the tree for the 1 nearest neighbor
3. Place that tile

**Expected result:** Very repetitive — mostly 1-2 tiles for the black/white regions. This is the control/baseline for comparison.

---

### Strategy 2: `bucket_random` (Bucket + Random Selection)

Group tiles into buckets by brightness, then randomly pick from the matching bucket.

**Algorithm:**
1. Compute brightness for each tile: `brightness = 0.299*R + 0.587*G + 0.114*B`
2. Divide the brightness range [0, 255] into N equal buckets (N = `--bucket-size`, default 32)
3. Assign each tile to a bucket based on its brightness
4. For each pixel in the source frame:
   a. Compute the pixel's brightness
   b. Find the corresponding bucket
   c. Randomly select a tile from that bucket
5. If a bucket is empty, fall back to the nearest non-empty bucket

**Expected result:** More variety than nearest, but may look noisy since tiles are chosen purely randomly within each bucket. Color accuracy within each bucket is not guaranteed.

---

### Strategy 3: `weighted_random` (Top-K Weighted Random)

Pick randomly from the K closest matches, weighted by inverse distance.

**Algorithm:**
1. Build a KD-Tree from all tile average RGB values
2. For each pixel, query the tree for the K nearest neighbors (K = `--top-k`, default 10)
3. Compute weights as `1 / (distance + epsilon)` where epsilon = 1e-6 to avoid division by zero
4. Normalize weights to probabilities
5. Randomly select one tile according to these probabilities

**Expected result:** Good balance of accuracy and variety. Closer matches are more likely but not guaranteed. Tiles that are very far off in color are excluded entirely.

---

### Strategy 4: `dither_nearest` (Dither + Closest Match)

Apply dithering to the source frame first to convert binary B&W into a range of gray/color values, then use nearest-neighbor matching.

**Algorithm:**
1. Load the source frame at 120×90
2. Apply dithering to create intermediate tones:
   - **Floyd-Steinberg** (default): Error-diffusion dithering. Process pixels left-to-right, top-to-bottom. For each pixel, find the nearest tile color, compute the error (difference between original pixel and chosen tile's average color), and distribute the error to neighboring pixels:
     - right pixel: +7/16 of error
     - bottom-left: +3/16
     - bottom: +5/16
     - bottom-right: +1/16
   
   **Important:** The dithering here is NOT converting to B&W. It's quantizing to the palette of available tile colors. The "nearest color" at each step comes from the tile library, and the error diffusion spreads the quantization error to neighbors. This naturally forces neighboring pixels to compensate, pulling in a variety of different tiles.
   
   - **Ordered dithering** (if `--dither-method ordered`): Apply a Bayer matrix threshold to shift pixel values before matching. Use an 8×8 Bayer matrix, tiled across the frame. For each pixel:
     1. Get the Bayer threshold value `t` at position `(x % 8, y % 8)`, normalized to [-0.5, 0.5]
     2. Adjust: `adjusted_pixel = original_pixel + t * spread` where `spread` controls dithering intensity (use 128 as default)
     3. Clamp to [0, 255]
     4. Find nearest tile to the adjusted pixel value

3. Build a KD-Tree from tile colors and match each (possibly error-adjusted) pixel to its nearest tile

**Expected result:** Much better variety than plain nearest, because the error diffusion forces neighboring pixels to use different tiles. The overall frame still reads correctly from a distance. Floyd-Steinberg tends to look more organic; ordered dithering has a more structured/patterned look.

---

### Strategy 5: `dither_bucket` (Dither + Bucket Random)

Apply dithering first, then use bucket randomization for additional variety.

**Algorithm:**
1. Apply dithering to the source frame (same as Strategy 4)
2. For each dithered pixel value, find the matching brightness bucket
3. Randomly select a tile from that bucket

**Expected result:** Maximum variety. Dithering creates tonal spread, and bucket randomization adds randomness on top. May be slightly noisier but very visually interesting.

---

### Strategy 6: `dither_weighted` (Dither + Top-K Weighted Random)

Apply dithering first, then use weighted random selection. This is expected to be the best balance of variety and accuracy.

**Algorithm:**
1. Apply dithering to the source frame (same as Strategy 4)
2. For each dithered pixel value, query the KD-Tree for K nearest tiles
3. Select randomly with inverse-distance weighting (same as Strategy 3)

**Expected result:** Best of both worlds — dithering creates natural tonal variety, and weighted randomization adds controlled diversity while keeping color accuracy high. This is the expected winner.

---

## Comparison Helper

Also create a small helper script `scripts/compare_strategies.py` that:

1. Runs `test_build_mosaic.py --test` for ALL 6 strategies automatically
2. For each test frame, creates a **side-by-side comparison image** showing all 6 strategy results in a 2×3 grid, with the strategy name labeled on each
3. Also includes the original Bad Apple frame (scaled up to match) as a 7th panel for reference
4. Saves comparison images to `data/comparisons/comparison_frame_XXXXX.png`

This lets us visually evaluate all strategies at a glance.

## Test frame choice: 
Select frames 01000, 02000, 03000, and 04000 for testing purpose. 

---

## Performance Requirements

- Use `scipy.spatial.cKDTree` for all nearest-neighbor lookups
- Pre-load all tile images into a NumPy array at startup
- Add `tqdm` progress bars for frame processing
- **Checkpoint/resume:** Skip frames that already exist in the output directory (check file existence before processing)
- For full runs (not test mode), consider using `concurrent.futures.ProcessPoolExecutor` for parallel frame processing. Note: dithering strategies process pixels sequentially within a frame (due to error diffusion), so parallelism is at the frame level, not pixel level
- Set random seed (`--seed`) at the start for reproducible results

## Dependencies

All dependencies are already in `requirements.txt` — no new packages needed. Use:
- `numpy` for array operations
- `scipy.spatial.cKDTree` for nearest-neighbor
- `Pillow` for image I/O
- `tqdm` for progress bars
- `argparse` for CLI arguments
