"""
Stage 5: encode_video.py
Extract audio from the source video, then stitch mosaic frames + audio into the final MP4.
"""

import subprocess
import sys
from pathlib import Path

MOSAIC_DIR = Path("data/mosaic_frames")
OUTPUT_DIR = Path("output")
AUDIO_FILE = Path("data/bad_apple_audio.aac")
FINAL_VIDEO = OUTPUT_DIR / "bad_apple_horse_mosaic.mp4"
CANONICAL_SOURCE = Path("data/bad_apple.mp4")


def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("ERROR: ffmpeg not found on PATH.")
        print("Install via: winget install ffmpeg")
        sys.exit(1)


def find_source_video() -> Path:
    if CANONICAL_SOURCE.exists():
        return CANONICAL_SOURCE
    mp4s = [p for p in Path("data").glob("*.mp4") if p != FINAL_VIDEO]
    if not mp4s:
        print("ERROR: No source .mp4 found in data/")
        sys.exit(1)
    return mp4s[0]


def extract_audio(source: Path):
    if AUDIO_FILE.exists():
        print(f"Audio already extracted: {AUDIO_FILE}")
        return
    print(f"Extracting audio from {source}...")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(source),
        "-vn",
        "-acodec", "copy",
        str(AUDIO_FILE),
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        # Try AAC encoding if stream copy fails
        print("Stream copy failed — trying AAC encode...")
        cmd[-2] = "-acodec"
        cmd[-1] = "aac"
        result = subprocess.run(cmd)
    if result.returncode != 0:
        print("WARNING: Could not extract audio. Final video will be silent.")


def encode_video():
    frames = sorted(MOSAIC_DIR.glob("frame_*.jpg"))
    if not frames:
        print(f"No mosaic frames found in {MOSAIC_DIR}. Run build_mosaic.py first.")
        sys.exit(1)
    print(f"Found {len(frames)} mosaic frames.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frame_pattern = str(MOSAIC_DIR / "frame_%05d.jpg")

    if AUDIO_FILE.exists():
        cmd = [
            "ffmpeg", "-y",
            "-framerate", "30",
            "-i", frame_pattern,
            "-i", str(AUDIO_FILE),
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-shortest",
            str(FINAL_VIDEO),
        ]
    else:
        print("No audio file — encoding video-only.")
        cmd = [
            "ffmpeg", "-y",
            "-framerate", "30",
            "-i", frame_pattern,
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            str(FINAL_VIDEO),
        ]

    print(f"\nEncoding final video...")
    print(f"Command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("\nFFmpeg encoding failed.")
        sys.exit(1)

    size_mb = FINAL_VIDEO.stat().st_size / 1_048_576
    print(f"\nDone! Output: {FINAL_VIDEO} ({size_mb:.1f} MB)")


def main():
    check_ffmpeg()
    source = find_source_video()
    extract_audio(source)
    encode_video()


if __name__ == "__main__":
    main()
