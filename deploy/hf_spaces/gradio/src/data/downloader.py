from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

HAM10000_URLS = {
    "metadata": "https://dataverse.harvard.edu/api/access/datafile/3172582",
    "part1": "https://dataverse.harvard.edu/api/access/datafile/3172585",
    "part2": "https://dataverse.harvard.edu/api/access/datafile/3172584",
}


def download_file(url: str, destination: Path, chunk_size: int = 8192) -> None:
    """Download a file from URL with progress logging."""
    logger.info("Downloading %s -> %s", url, destination)
    response = requests.get(url, stream=True, timeout=600)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    downloaded = 0

    with destination.open("wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded / total * 100
                if downloaded % (chunk_size * 128) == 0:
                    logger.info("  %.1f%% (%d / %d)", pct, downloaded, total)

    logger.info("Download complete: %s (%.1f MB)", destination.name, destination.stat().st_size / 1e6)


def download_ham10000(data_dir: str = "data/raw") -> dict[str, Path]:
    """Download the HAM10000 dataset.

    Returns dict with paths to downloaded files.
    """
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)

    # Download metadata
    metadata_path = root / "HAM10000_metadata.csv"
    if not metadata_path.exists():
        download_file(HAM10000_URLS["metadata"], metadata_path)

    # Download and extract image archives
    for part_name, url_key in [("part1", "part1"), ("part2", "part2")]:
        archive_path = root / f"HAM10000_images_{part_name}.zip"
        extract_dir = root / f"HAM10000_images_{part_name}"

        if not extract_dir.exists() or not any(extract_dir.iterdir()):
            if not archive_path.exists():
                download_file(HAM10000_URLS[url_key], archive_path)

            logger.info("Extracting %s ...", archive_path.name)
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(extract_dir)
            logger.info("Extraction complete: %s", extract_dir)

    # Verify
    total_images = sum(
        1 for p in root.rglob("*.jpg")
    )
    logger.info("Total images found: %d (expected ~10015)", total_images)
    if total_images < 10000:
        logger.warning(
            "Expected ~10015 images, found %d. Download may be incomplete.", total_images
        )

    return {
        "metadata": metadata_path,
        "image_count": total_images,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_ham10000()
