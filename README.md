# 🐴 Bad Apple!! Horse Mosaic Video

## Project Overview

Recreate the iconic **Bad Apple!! shadow art music video** as a photo mosaic, where every pixel in each frame is a tiny horse image. This celebrates the **Year of the Horse (2026)** by sourcing horse photographs of varying colors and brightness levels to reconstruct the video frame by frame.

**Inspiration:** [Bad Apple!! but every frame is 2,135 Yu-Gi-Oh cards](https://youtu.be/jTSdRIgrzZM) and similar pixel-art mosaic recreations of the Bad Apple music video.

---

## Technical Specifications

| Parameter | Value |
|---|---|
| Frame resolution | 120 × 90 pixels (10,800 horse images per frame) |
| Aspect ratio | 4:3 (matches original Bad Apple) |
| Total frames | ~6,570 frames |
| Frame rate | 30 FPS |
| Duration | ~3 minutes 39 seconds |
| Tile scale factor | 12× (each horse image rendered at 12×12 pixels) |
| Output resolution | 1440 × 1080 pixels |
| Color matching | RGB Euclidean distance (full color, not grayscale) |
| Tile repeats | Allowed freely within and across frames |

---

## Pipeline Architecture

The project is a 5-stage pipeline. Each script is standalone and can be re-run independently.

```
scrape_horses.py → preprocess_tiles.py → extract_frames.py → build_mosaic.py → encode_video.py
```

### Stage 1: `scrape_horses.py` — Scrape Horse Images from Pexels

**Purpose:** Download a diverse set of horse photographs spanning the full brightness and color spectrum.

**API:** Pexels API v1 (`https://api.pexels.com/v1/search`)
- Auth: API key passed via `Authorization` header
- Rate limit: 200 requests/hour, 20,000 requests/month
- Max 80 results per page
- The API key should be read from a `.env` file (variable: `PEXELS_API_KEY`)

**Search Strategy — Brightness/Color Coverage:**

We need horse images ranging from very bright (white) to very dark (black) to accurately reproduce Bad Apple's high-contrast shadow art. Use these search queries to get good coverage:

*Bright/White:*
- `"white horse snow"`
- `"white horse bright"`
- `"white horse beach"`
- `"white horse field"`
- `"pale horse light"`
- `"gray horse"`
- `"horse fog mist"`

*Dark/Black:*
- `"black horse"`
- `"black horse dark"`
- `"dark horse mountain"`
- `"horse silhouette"`
- `"horse night"`
- `"black stallion"`
- `"horse shadow"`

**Download Details:**
- Use the `small` image size from Pexels (130px height, aspect ratio preserved) — we're shrinking these to 12×12px tiles anyway, so we don't need large files
- Save images to `data/raw_horses/`
- Save attribution metadata to `data/attribution.json` (photographer name, Pexels URL, photo ID) — Pexels requires attribution
- Skip duplicates (check by photo ID)
- Target: **500–1000+ unique images** (more = better color coverage)
- For each query, paginate through at least 5 pages (80 results per page = 400 per query)

**Output:**
- `data/raw_horses/*.jpg` — downloaded horse images
- `data/attribution.json` — list of `{id, photographer, photographer_url, photo_url}`

---

### Stage 2: `preprocess_tiles.py` — Prepare Tile Library

**Purpose:** Resize all horse images to uniform square tiles and compute their average RGB color for fast matching.

**Steps:**
1. Load each image from `data/raw_horses/`
2. Center-crop to square (crop the longer dimension to match the shorter)
3. Resize to **12×12 pixels** (the tile size for the final mosaic)
4. Compute the **average RGB color** of the tile (mean of all pixels across R, G, B channels)
5. Save the processed tile to `data/tiles/`
6. Save the tile library index to `data/tile_index.json`:
   ```json
   [
     {
       "filename": "12345.jpg",
       "avg_rgb": [182, 165, 140]
     },
     ...
   ]
   ```

**Output:**
- `data/tiles/*.jpg` — 12×12 pixel square horse tiles
- `data/tile_index.json` — index mapping each tile to its average RGB

---

### Stage 3: `extract_frames.py` — Extract Bad Apple Frames

**Purpose:** Extract individual frames from the Bad Apple source video.

**Prerequisites:**
- FFmpeg must be installed and available on PATH
- A Bad Apple source video file placed at `data/bad_apple.mp4`
  - The user needs to download this themselves (e.g., from YouTube using yt-dlp)
  - Any resolution is fine; we resize during extraction

**Steps:**
1. Use FFmpeg to extract all frames at 30 FPS
2. Resize each frame to **120 × 90 pixels** during extraction
3. Save as PNG files to `data/frames/`

**FFmpeg command:**
```bash
ffmpeg -i data/bad_apple.mp4 -vf "scale=120:90" -r 30 data/frames/frame_%05d.png
```

**Output:**
- `data/frames/frame_00001.png` through `frame_XXXXX.png` — 120×90 pixel frames

---

### Stage 4: `build_mosaic.py` — Build Mosaic Frames

**Purpose:** For each Bad Apple frame, replace every pixel with the closest-matching horse tile to create the mosaic frame.

**Algorithm:**
1. Load the tile library index (`data/tile_index.json`) and all tile images into memory
2. Build a NumPy array of all tile average RGB values for vectorized matching
3. For each frame in `data/frames/`:
   a. Load the 120×90 frame
   b. For each pixel at position (x, y):
      - Get the pixel's RGB value
      - Find the an appropriate tile to replace the pixel (see more details about tile choices in the MOSAIC_STRATEGIES file)
      - Place that tile's 12×12 image at position (x*12, y*12) in the output canvas
   c. Save the completed 1440×1080 mosaic frame to `data/mosaic_frames/`

**Performance Considerations:**
- Pre-load all tiles into a NumPy array for fast distance computation
- Use vectorized operations (e.g., `scipy.spatial.cKDTree` or `sklearn.neighbors.BallTree`) for nearest-neighbor lookup — avoids O(n) linear scan per pixel
- Use `numpy` broadcasting to compute distances for all pixels in a frame at once if possible
- Consider multiprocessing (e.g., `concurrent.futures.ProcessPoolExecutor`) to process multiple frames in parallel
- Add progress bar with `tqdm`
- Save output as JPEG (quality 95) to save disk space — ~6,570 frames at 1440×1080 will use significant disk space
- **Checkpoint/resume support:** Track which frames have been processed so the script can resume after interruption (check if output file already exists, skip if so)

**Output:**
- `data/mosaic_frames/frame_00001.jpg` through `frame_XXXXX.jpg` — 1440×1080 mosaic frames

---

### Stage 5: `encode_video.py` — Encode Final Video

**Purpose:** Stitch all mosaic frames into a video and add the Bad Apple audio track.

**Steps:**
1. Extract audio from the original Bad Apple video:
   ```bash
   ffmpeg -i data/bad_apple.mp4 -vn -acodec copy data/bad_apple_audio.aac
   ```
2. Encode the mosaic frames into a video:
   ```bash
   ffmpeg -framerate 30 -i data/mosaic_frames/frame_%05d.jpg -i data/bad_apple_audio.aac -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p -c:a aac -shortest output/bad_apple_horse_mosaic.mp4
   ```
3. The `-crf 18` gives high quality, `-preset slow` gives better compression

**Output:**
- `output/bad_apple_horse_mosaic.mp4` — the final video!

---

## Project Directory Structure

```
bad-apple-horse-mosaic/
├── README.md                  # This file
├── .env                       # PEXELS_API_KEY=your_key_here
├── .gitignore                 # Ignore data/, output/, .env
├── requirements.txt           # Python dependencies
│
├── scripts/
│   ├── scrape_horses.py       # Stage 1: Scrape from Pexels
│   ├── preprocess_tiles.py    # Stage 2: Resize + compute avg RGB
│   ├── extract_frames.py      # Stage 3: Extract Bad Apple frames
│   ├── build_mosaic.py        # Stage 4: Build mosaic frames
│   └── encode_video.py        # Stage 5: Encode final video
│
├── data/                      # All intermediate data (gitignored)
│   ├── raw_horses/            # Downloaded horse images
│   ├── tiles/                 # Processed 12×12 tiles
│   ├── frames/                # Extracted Bad Apple frames (120×90)
│   ├── mosaic_frames/         # Generated mosaic frames (1440×1080)
│   ├── tile_index.json        # Tile RGB index
│   └── attribution.json       # Pexels photographer credits
│
└── output/                    # Final video output
    └── bad_apple_horse_mosaic.mp4
```

---

## Dependencies

```
# requirements.txt
requests>=2.31.0      # HTTP requests for Pexels API
Pillow>=10.0.0        # Image processing
numpy>=1.24.0         # Array operations
scipy>=1.11.0         # cKDTree for fast nearest-neighbor
tqdm>=4.65.0          # Progress bars
python-dotenv>=1.0.0  # Load .env file
```

**System dependencies:**
- Python 3.10+
- FFmpeg (install via `winget install ffmpeg` or download from https://ffmpeg.org)

---

## Setup Instructions

1. Clone/create the project directory
2. Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Get a free Pexels API key at https://www.pexels.com/api/ and add to `.env`:
   ```
   PEXELS_API_KEY=your_api_key_here
   ```
5. Download the Bad Apple video and place it at `data/bad_apple.mp4`
   ```bash
   # Using yt-dlp (install: pip install yt-dlp)
   yt-dlp -o "data/bad_apple.mp4" "https://www.nicovideo.jp/watch/sm8628149"
   ```
6. Run the pipeline in order:
   ```bash
   python scripts/scrape_horses.py
   python scripts/preprocess_tiles.py
   python scripts/extract_frames.py
   python scripts/build_mosaic.py
   python scripts/encode_video.py
   ```

---

## Notes & Considerations

- **Pexels Attribution:** Pexels requires credit to photographers. The scraper saves attribution data. Consider adding a credits section at the end of the video or in the video description.
- **Disk Space:** ~6,570 mosaic frames at 1440×1080 JPEG (quality 95) will use roughly 10-20 GB. Ensure sufficient disk space.
- **Processing Time:** The mosaic building step (Stage 4) is the bottleneck. With vectorized matching and multiprocessing, expect several hours for all frames. The checkpoint/resume feature is critical.
- **Color Matching Quality:** The quality of the final video depends heavily on having horse images that cover the full brightness/color range. If the mosaic looks washed out or lacks contrast, scrape more images targeting the underrepresented brightness ranges.
- **Bad Apple Source:** The original shadow art video from Nico Nico Douga (sm8628149) is the canonical source. YouTube mirrors work fine too. Any version with the standard shadow art animation will work.
