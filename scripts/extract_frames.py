"""
Stage 3: extract_frames.py
Extract Bad Apple frames at 120×90px using FFmpeg.
Looks for the source video at data/bad_apple.mp4 or any .mp4 in data/.
"""

import subprocess
import sys
from pathlib import Path

FRAMES_DIR = Path("data/frames")
# Try canonical name first, then fall back to any .mp4 in data/
CANONICAL = Path("data/bad_apple.mp4")


def find_source_video() -> Path:
    if CANONICAL.exists():
        return CANONICAL
    mp4s = list(Path("data").glob("*.mp4"))
    if not mp4s:
        print("ERROR: No .mp4 file found in data/.")
        print("Place your Bad Apple source video at data/bad_apple.mp4")
        sys.exit(1)
    if len(mp4s) > 1:
        print(f"Multiple .mp4 files found: {[p.name for p in mp4s]}")
        print(f"Using: {mp4s[0].name}")
    return mp4s[0]


def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("ERROR: ffmpeg not found on PATH.")
        print("Install via: winget install ffmpeg")
        sys.exit(1)


def main():
    check_ffmpeg()
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    source = find_source_video()
    print(f"Source video: {source}")

    existing = list(FRAMES_DIR.glob("frame_*.png"))
    if existing:
        print(f"Found {len(existing)} existing frames — skipping extraction.")
        print("Delete data/frames/ to re-extract.")
        return

    output_pattern = str(FRAMES_DIR / "frame_%05d.png")
    cmd = [
        "ffmpeg",
        "-i", str(source),
        "-vf", "scale=120:90",
        "-r", "30",
        output_pattern,
    ]

    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("\nFFmpeg failed. Check the command above for details.")
        sys.exit(1)

    frames = list(FRAMES_DIR.glob("frame_*.png"))
    print(f"\nDone. Extracted {len(frames)} frames to {FRAMES_DIR}.")


if __name__ == "__main__":
    main()
