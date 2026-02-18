"""
Stage 1: scrape_horses.py
Download horse images from Pexels API covering the full brightness spectrum.
"""

import os
import json
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("PEXELS_API_KEY")
if not API_KEY:
    raise ValueError("PEXELS_API_KEY not set in .env")

BASE_URL = "https://api.pexels.com/v1/search"
RAW_DIR = Path("data/raw_horses")
ATTRIBUTION_FILE = Path("data/attribution.json")
PAGES_PER_QUERY = 5
PER_PAGE = 80

QUERIES = [
    # Bright/White
    "white horse snow",
    "white horse bright",
    "white horse beach",
    "white horse field",
    "pale horse light",
    "gray horse",
    "horse fog mist",
    # Dark/Black
    "black horse",
    "black horse dark",
    "dark horse mountain",
    "horse silhouette",
    "horse night",
    "black stallion",
    "horse shadow",
]


def load_attribution():
    if ATTRIBUTION_FILE.exists():
        with open(ATTRIBUTION_FILE) as f:
            records = json.load(f)
        return {r["id"]: r for r in records}
    return {}


def save_attribution(records: dict):
    with open(ATTRIBUTION_FILE, "w") as f:
        json.dump(list(records.values()), f, indent=2)


def fetch_page(query: str, page: int, session: requests.Session):
    resp = session.get(
        BASE_URL,
        params={"query": query, "per_page": PER_PAGE, "page": page},
        headers={"Authorization": API_KEY},
        timeout=30,
    )
    if resp.status_code == 429:
        print("  Rate limited — waiting 60s...")
        time.sleep(60)
        return fetch_page(query, page, session)
    resp.raise_for_status()
    return resp.json()


def download_image(photo: dict, session: requests.Session, attribution: dict) -> bool:
    photo_id = str(photo["id"])
    if photo_id in attribution:
        return False  # already downloaded

    url = photo["src"]["small"]
    dest = RAW_DIR / f"{photo_id}.jpg"

    try:
        r = session.get(url, timeout=30)
        r.raise_for_status()
        dest.write_bytes(r.content)
    except Exception as e:
        print(f"  Failed to download {photo_id}: {e}")
        return False

    attribution[photo_id] = {
        "id": photo_id,
        "photographer": photo.get("photographer", ""),
        "photographer_url": photo.get("photographer_url", ""),
        "photo_url": photo.get("url", ""),
    }
    return True


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ATTRIBUTION_FILE.parent.mkdir(parents=True, exist_ok=True)

    attribution = load_attribution()
    print(f"Starting with {len(attribution)} already-downloaded images.\n")

    session = requests.Session()
    new_count = 0

    for query in QUERIES:
        print(f"Query: '{query}'")
        for page in range(1, PAGES_PER_QUERY + 1):
            try:
                data = fetch_page(query, page, session)
            except Exception as e:
                print(f"  Page {page} error: {e}")
                break

            photos = data.get("photos", [])
            if not photos:
                break

            for photo in photos:
                if download_image(photo, session, attribution):
                    new_count += 1

            print(f"  Page {page}: {len(photos)} results, total unique: {len(attribution)}")
            time.sleep(0.5)  # be polite to the API

        save_attribution(attribution)

    print(f"\nDone. Downloaded {new_count} new images. Total: {len(attribution)} unique horse images.")
    save_attribution(attribution)


if __name__ == "__main__":
    main()
