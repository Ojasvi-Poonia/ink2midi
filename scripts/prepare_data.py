#!/usr/bin/env python3
"""Parse and convert datasets to YOLO/COCO format with train/val/test splits."""

import argparse
import json
import re
from pathlib import Path

from omr.data.coco_converter import convert_to_coco
from omr.data.muscima_parser import MUSCIMAParser
from omr.data.deepscores_parser import DeepScoresParser
from omr.data.yolo_converter import convert_to_yolo, verify_yolo_dataset
from omr.utils.logging import setup_logging, get_logger

logger = get_logger("scripts.prepare_data")


def _find_annotations_dir(muscima_dir: Path) -> Path | None:
    """Find the MUSCIMA++ annotations directory.

    The extracted zip creates a nested structure like:
        muscima_pp_v2/v2.0/data/annotations/CVC-MUSCIMA_W-XX_N-YY_D-ideal.xml

    But we need to handle variations in extraction structure.
    """
    # Known path from MUSCIMA++ v2.0 zip extraction
    candidates = [
        muscima_dir / "v2.0" / "data" / "annotations",
        muscima_dir / "data" / "annotations",
        muscima_dir / "annotations",
        muscima_dir / "MUSCIMA-pp_v2.0" / "v2.0" / "data" / "annotations",
    ]

    for candidate in candidates:
        if candidate.exists() and list(candidate.glob("CVC-MUSCIMA_*.xml")):
            logger.info(f"Found annotations at: {candidate}")
            return candidate

    # Fallback: search recursively for MUSCIMA++ annotation XMLs
    logger.info("Standard paths not found, searching recursively...")
    for xml_file in muscima_dir.rglob("CVC-MUSCIMA_*.xml"):
        ann_dir = xml_file.parent
        # Verify it has multiple annotation files (not just one stray file)
        xml_count = len(list(ann_dir.glob("CVC-MUSCIMA_*.xml")))
        if xml_count > 10:
            logger.info(f"Found {xml_count} annotations at: {ann_dir}")
            return ann_dir

    return None


def _find_images_dir(muscima_dir: Path) -> Path | None:
    """Find the CVC-MUSCIMA images directory.

    CVC-MUSCIMA images may be in various nested structures after extraction:
        muscima_pp_v2/images/CVCMUSCIMA_SR/CVC-MUSCIMA_SR/
        muscima_pp_v2/images/

    We search recursively for directories containing .png files.
    """
    images_base = muscima_dir / "images"

    if not images_base.exists():
        return None

    # Check if images are directly in images_base
    if list(images_base.glob("*.png")):
        return images_base

    # CVC-MUSCIMA has writer-based subdirectories like w-01, w-02, etc.
    # The actual PNGs are at: images/CVCMUSCIMA_SR/CVC-MUSCIMA_SR/w-XX/image/pXX.png
    # or similar nested paths. The parser does recursive search, so we just
    # need to return the root images directory.
    png_files = list(images_base.rglob("*.png"))
    if png_files:
        logger.info(f"Found {len(png_files)} PNG images under {images_base}")
        return images_base

    return None


def prepare_muscima(raw_dir: Path, processed_dir: Path, splits_dir: Path):
    """Parse MUSCIMA++ and convert to YOLO format with writer-disjoint splits."""
    logger.info("Preparing MUSCIMA++ v2...")

    muscima_dir = raw_dir / "muscima_pp_v2"

    if not muscima_dir.exists():
        logger.error(
            f"MUSCIMA++ directory not found at {muscima_dir}\n"
            "Run 'python scripts/download_data.py' first"
        )
        return

    # Find annotations
    annotations_dir = _find_annotations_dir(muscima_dir)
    if annotations_dir is None:
        logger.error(
            f"MUSCIMA++ annotations not found under {muscima_dir}\n"
            "Run 'python scripts/download_data.py' first"
        )
        return

    # Find images
    images_dir = _find_images_dir(muscima_dir)
    if images_dir is None:
        logger.warning(
            "CVC-MUSCIMA images not found. Annotations will be converted "
            "but images will not be copied to the YOLO dataset.\n"
            "You can download images manually from:\n"
            "  https://datasets.cvc.uab.cat/muscima/CVCMUSCIMA_SR.zip\n"
            f"Extract to: {muscima_dir / 'images'}"
        )
        # Use a dummy path — parser will handle missing images gracefully
        images_dir = muscima_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Annotations dir: {annotations_dir}")
    logger.info(f"Images dir: {images_dir}")

    parser = MUSCIMAParser()
    documents = parser.parse_directory(annotations_dir, images_dir)

    if not documents:
        logger.error("No documents parsed from MUSCIMA++")
        return

    # Writer-disjoint splits
    # MUSCIMA++ files are named like: CVC-MUSCIMA_W-XX_N-YY_D-ideal.xml
    # where XX is writer number
    train_docs = []
    val_docs = []
    test_docs = []

    for doc in documents:
        writer_num = _extract_writer_number(doc.image_id)
        if writer_num <= 35:
            train_docs.append(doc)
        elif writer_num <= 42:
            val_docs.append(doc)
        else:
            test_docs.append(doc)

    logger.info(
        f"MUSCIMA++ splits: train={len(train_docs)}, "
        f"val={len(val_docs)}, test={len(test_docs)}"
    )

    # Convert to YOLO format
    output_dir = processed_dir / "muscima_yolo"
    for split_name, split_docs in [
        ("train", train_docs),
        ("val", val_docs),
        ("test", test_docs),
    ]:
        convert_to_yolo(split_docs, output_dir, split=split_name)

    # Also convert to COCO for evaluation
    coco_dir = processed_dir / "muscima_coco"
    convert_to_coco(train_docs, coco_dir / "train.json")
    convert_to_coco(val_docs, coco_dir / "val.json")
    convert_to_coco(test_docs, coco_dir / "test.json")

    # Verify
    verification = verify_yolo_dataset(output_dir)
    logger.info(f"YOLO dataset verification: {verification}")

    # Report image status
    for split_name, split_result in verification.get("splits", {}).items():
        n_labels = split_result.get("num_labels", 0)
        n_images = split_result.get("num_images", 0)
        n_missing = split_result.get("missing_images", 0)
        if n_missing > 0:
            logger.warning(
                f"  {split_name}: {n_labels} labels, {n_images} images, "
                f"{n_missing} missing images (download CVC-MUSCIMA images)"
            )
        else:
            logger.info(
                f"  {split_name}: {n_labels} labels, {n_images} images ✓"
            )

    # Save split info
    splits_dir.mkdir(parents=True, exist_ok=True)
    split_info = {
        "train": [d.image_id for d in train_docs],
        "val": [d.image_id for d in val_docs],
        "test": [d.image_id for d in test_docs],
    }
    with open(splits_dir / "muscima_splits.json", "w") as f:
        json.dump(split_info, f, indent=2)

    logger.info(f"Split info saved to {splits_dir / 'muscima_splits.json'}")


def prepare_deepscores(raw_dir: Path, processed_dir: Path, splits_dir: Path):
    """Parse DeepScores V2 and convert to YOLO format."""
    logger.info("Preparing DeepScores V2...")

    ds_dir = raw_dir / "deepscores_v2"

    if not ds_dir.exists():
        logger.warning(
            f"DeepScores V2 directory not found at {ds_dir}\n"
            "Skipping. Run 'python scripts/download_data.py' first."
        )
        return

    # Find annotation JSON
    ann_path = None
    for candidate in ds_dir.rglob("*.json"):
        if "annotation" in candidate.name.lower() or "deepscores" in candidate.name.lower():
            ann_path = candidate
            break

    if ann_path is None:
        # Try common names
        for name in ["ds2_dense.json", "annotations.json", "deepscores_train.json"]:
            candidate = ds_dir / name
            if candidate.exists():
                ann_path = candidate
                break

    if ann_path is None:
        logger.error(f"DeepScores V2 annotations not found in {ds_dir}")
        logger.info("Run 'python scripts/download_data.py' first")
        return

    images_dir = ann_path.parent / "images"
    if not images_dir.exists():
        images_dir = ds_dir / "images"

    parser = DeepScoresParser()
    documents = parser.parse_annotations(ann_path, images_dir)

    if not documents:
        logger.error("No documents parsed from DeepScores V2")
        return

    # Random 70/15/15 split
    import random
    random.seed(42)
    random.shuffle(documents)

    n = len(documents)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    train_docs = documents[:n_train]
    val_docs = documents[n_train : n_train + n_val]
    test_docs = documents[n_train + n_val :]

    logger.info(
        f"DeepScores V2 splits: train={len(train_docs)}, "
        f"val={len(val_docs)}, test={len(test_docs)}"
    )

    # Convert to YOLO format
    output_dir = processed_dir / "deepscores_yolo"
    for split_name, split_docs in [
        ("train", train_docs),
        ("val", val_docs),
        ("test", test_docs),
    ]:
        convert_to_yolo(split_docs, output_dir, split=split_name)

    # Save split info
    splits_dir.mkdir(parents=True, exist_ok=True)
    split_info = {
        "train": [d.image_id for d in train_docs],
        "val": [d.image_id for d in val_docs],
        "test": [d.image_id for d in test_docs],
    }
    with open(splits_dir / "deepscores_splits.json", "w") as f:
        json.dump(split_info, f, indent=2)


def _extract_writer_number(image_id: str) -> int:
    """Extract writer number from MUSCIMA++ image ID."""
    # Pattern: CVC-MUSCIMA_W-XX_N-YY_D-ideal or similar
    match = re.search(r"W-(\d+)", image_id)
    if match:
        return int(match.group(1))
    # Fallback: hash-based assignment
    return hash(image_id) % 50 + 1


def main():
    parser = argparse.ArgumentParser(description="Prepare OMR datasets")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument(
        "--dataset",
        choices=["muscima", "deepscores", "all"],
        default="all",
    )
    args = parser.parse_args()

    setup_logging("INFO")

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    splits_dir = Path(args.splits_dir)

    if args.dataset in ("muscima", "all"):
        prepare_muscima(raw_dir, processed_dir, splits_dir)

    if args.dataset in ("deepscores", "all"):
        prepare_deepscores(raw_dir, processed_dir, splits_dir)


if __name__ == "__main__":
    main()
