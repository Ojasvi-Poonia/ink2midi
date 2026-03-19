"""Dataset download automation for MUSCIMA++ and DeepScores V2."""

import os
import shutil
import tarfile
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from omr.utils.logging import get_logger

logger = get_logger("data.download")

# MUSCIMA++ v2 URLs
MUSCIMA_PP_URL = (
    "https://github.com/OMR-Research/muscima-pp/releases/download/v2.0/MUSCIMA-pp_v2.0.zip"
)

# CVC-MUSCIMA staff-removed images — multiple mirrors to try
CVC_MUSCIMA_URLS = [
    "https://datasets.cvc.uab.cat/muscima/CVCMUSCIMA_SR.zip",
    "http://datasets.cvc.uab.cat/muscima/CVCMUSCIMA_SR.zip",
]

# DeepScores V2 dense subset (Zenodo) — file is tar.gz, not zip
DEEPSCORES_V2_DENSE_URL = (
    "https://zenodo.org/records/4012193/files/ds2_dense.tar.gz?download=1"
)

# Alternative mirror from OMR-Datasets GitHub
DEEPSCORES_V2_MIRROR_URL = (
    "https://github.com/apacha/OMR-Datasets/releases/download/datasets/deep-scores-v2-dense.tar.gz"
)


def download_file(
    url: str,
    dest_path: Path,
    chunk_size: int = 8192,
    verify_ssl: bool = True,
    timeout: int = 600,
) -> Path:
    """Download a file with progress bar.

    Args:
        url: URL to download from.
        dest_path: Local destination path.
        chunk_size: Download chunk size.
        verify_ssl: Whether to verify SSL certificates.
        timeout: Request timeout in seconds.

    Returns:
        The destination path.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        logger.info(f"File already exists: {dest_path}")
        return dest_path

    logger.info(f"Downloading {url} -> {dest_path}")

    try:
        response = requests.get(
            url, stream=True, timeout=timeout, verify=verify_ssl
        )
        response.raise_for_status()
    except requests.exceptions.SSLError:
        if verify_ssl:
            logger.warning(
                f"SSL verification failed for {url}. "
                "Retrying without SSL verification..."
            )
            return download_file(
                url, dest_path, chunk_size, verify_ssl=False, timeout=timeout
            )
        raise

    total_size = int(response.headers.get("content-length", 0))

    with open(dest_path, "wb") as f, tqdm(
        total=total_size, unit="B", unit_scale=True, desc=dest_path.name
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            pbar.update(len(chunk))

    # Validate the download
    actual_size = dest_path.stat().st_size
    if total_size > 0 and actual_size < total_size:
        logger.warning(
            f"Download may be incomplete: expected {total_size} bytes, "
            f"got {actual_size} bytes"
        )
        dest_path.unlink()
        raise RuntimeError(f"Incomplete download: {actual_size}/{total_size} bytes")

    return dest_path


def extract_zip(zip_path: Path, extract_dir: Path) -> Path:
    """Extract a zip file."""
    logger.info(f"Extracting {zip_path} -> {extract_dir}")
    extract_dir.mkdir(parents=True, exist_ok=True)

    # Validate before extracting
    if not zipfile.is_zipfile(zip_path):
        raise zipfile.BadZipFile(
            f"{zip_path} is not a valid zip file (possibly truncated download). "
            f"Delete it and re-run the download."
        )

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    return extract_dir


def extract_tar(tar_path: Path, extract_dir: Path) -> Path:
    """Extract a tar.gz file."""
    logger.info(f"Extracting {tar_path} -> {extract_dir}")
    extract_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(extract_dir)

    return extract_dir


def extract_archive(archive_path: Path, extract_dir: Path) -> Path:
    """Extract a zip or tar.gz archive based on file extension."""
    name = archive_path.name.lower()
    if name.endswith(".tar.gz") or name.endswith(".tgz"):
        return extract_tar(archive_path, extract_dir)
    elif name.endswith(".zip"):
        return extract_zip(archive_path, extract_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")


def download_muscima_pp(target_dir: str | Path) -> Path:
    """Download MUSCIMA++ v2 annotations and CVC-MUSCIMA images.

    Downloads:
    - MUSCIMA++ v2.0 annotations (XML files with symbol bounding boxes and relationships)
    - CVC-MUSCIMA staff-removed images (the score images annotations refer to)
    """
    target_dir = Path(target_dir) / "muscima_pp_v2"
    target_dir.mkdir(parents=True, exist_ok=True)

    # Download annotations
    ann_zip = target_dir / "muscima_pp_v2.zip"
    download_file(MUSCIMA_PP_URL, ann_zip)
    extract_zip(ann_zip, target_dir)

    # Download CVC-MUSCIMA images (staff-removed version)
    # The server at datasets.cvc.uab.cat has known SSL certificate issues,
    # so we disable SSL verification for this download.
    img_zip = target_dir / "cvc_muscima_sr.zip"

    # Validate existing zip — if corrupt/truncated, delete and re-download
    if img_zip.exists():
        if zipfile.is_zipfile(img_zip):
            logger.info(f"CVC-MUSCIMA zip already exists and is valid: {img_zip}")
        else:
            logger.warning(
                f"Existing CVC-MUSCIMA zip is invalid/truncated "
                f"({img_zip.stat().st_size / 1e6:.1f}MB). Deleting..."
            )
            img_zip.unlink()

    if not img_zip.exists():
        downloaded = False
        for url in CVC_MUSCIMA_URLS:
            try:
                logger.info(f"Trying CVC-MUSCIMA URL: {url}")
                download_file(url, img_zip, verify_ssl=False, timeout=600)
                # Validate the zip after download
                if not zipfile.is_zipfile(img_zip):
                    size_mb = img_zip.stat().st_size / 1e6
                    logger.warning(
                        f"Downloaded file is not a valid zip ({size_mb:.1f}MB), removing..."
                    )
                    img_zip.unlink()
                    continue
                downloaded = True
                break
            except Exception as e:
                logger.warning(f"Failed to download from {url}: {e}")
                if img_zip.exists():
                    img_zip.unlink()
                continue

        if not downloaded:
            logger.error(
                "Could not download CVC-MUSCIMA images from any URL.\n"
                "Please download manually from:\n"
                "  https://datasets.cvc.uab.cat/muscima/CVCMUSCIMA_SR.zip\n"
                f"and place the zip file at: {img_zip}\n"
                "Then re-run this script."
            )

    # Extract images if valid zip exists
    if img_zip.exists() and zipfile.is_zipfile(img_zip):
        images_dir = target_dir / "images"
        extract_zip(img_zip, images_dir)
        _catalog_cvc_images(images_dir)

    logger.info(f"MUSCIMA++ v2 ready at {target_dir}")
    return target_dir


def _catalog_cvc_images(images_dir: Path) -> None:
    """Log how many CVC-MUSCIMA images were extracted."""
    all_images = list(images_dir.rglob("*.png"))
    if not all_images:
        all_images = list(images_dir.rglob("*.tif"))
    if not all_images:
        all_images = list(images_dir.rglob("*.jpg"))

    if all_images:
        logger.info(f"Found {len(all_images)} CVC-MUSCIMA image files")
    else:
        logger.warning("No image files found in CVC-MUSCIMA extraction")


def download_deepscores_v2(target_dir: str | Path) -> Path:
    """Download DeepScores V2 dense subset.

    The dense subset contains 1,714 digitally rendered pages with
    annotations for 135 symbol classes. The file is distributed as tar.gz.
    """
    target_dir = Path(target_dir) / "deepscores_v2"
    target_dir.mkdir(parents=True, exist_ok=True)

    ds_archive = target_dir / "ds2_dense.tar.gz"

    if not ds_archive.exists():
        # Try primary Zenodo URL, then GitHub mirror
        urls = [DEEPSCORES_V2_DENSE_URL, DEEPSCORES_V2_MIRROR_URL]
        downloaded = False
        for url in urls:
            try:
                logger.info(f"Trying DeepScores V2 URL: {url}")
                download_file(url, ds_archive, timeout=1200)
                downloaded = True
                break
            except Exception as e:
                logger.warning(f"Failed to download from {url}: {e}")
                if ds_archive.exists():
                    ds_archive.unlink()
                continue

        if not downloaded:
            raise RuntimeError(
                "Could not download DeepScores V2 from any URL.\n"
                "Download manually from: https://zenodo.org/records/4012193\n"
                f"Place ds2_dense.tar.gz at: {ds_archive}"
            )

    extract_tar(ds_archive, target_dir)

    logger.info(f"DeepScores V2 dense ready at {target_dir}")
    return target_dir


def download_all(raw_dir: str | Path) -> dict:
    """Download all datasets.

    Each dataset download is independent — a failure in one does not
    prevent the others from being downloaded.
    """
    raw_dir = Path(raw_dir)
    paths = {}

    # MUSCIMA++ v2
    try:
        logger.info("=" * 60)
        logger.info("Downloading MUSCIMA++ v2...")
        logger.info("=" * 60)
        paths["muscima_pp"] = download_muscima_pp(raw_dir)
    except Exception as e:
        logger.error(f"Failed to download MUSCIMA++ v2: {e}")
        logger.info("You can retry later with: python scripts/download_data.py")

    # DeepScores V2
    try:
        logger.info("=" * 60)
        logger.info("Downloading DeepScores V2 dense...")
        logger.info("=" * 60)
        paths["deepscores_v2"] = download_deepscores_v2(raw_dir)
    except Exception as e:
        logger.error(f"Failed to download DeepScores V2: {e}")
        logger.info("You can retry later with: python scripts/download_data.py")

    # Summary
    logger.info("=" * 60)
    logger.info("Download summary:")
    for name, path in paths.items():
        logger.info(f"  {name}: {path}")
    if not paths:
        logger.warning("No datasets were downloaded successfully.")
    logger.info("=" * 60)

    return paths
