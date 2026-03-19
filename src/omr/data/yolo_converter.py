"""Convert unified annotations to YOLO format for YOLOv8 training."""

import shutil
from pathlib import Path

from omr.data.muscima_parser import DocumentAnnotation
from omr.utils.logging import get_logger

logger = get_logger("data.yolo_converter")


def convert_to_yolo(
    documents: list[DocumentAnnotation],
    output_dir: str | Path,
    split: str = "train",
    copy_images: bool = True,
) -> None:
    """Convert unified annotations to YOLO format.

    Creates the directory structure expected by Ultralytics YOLOv8:
        output_dir/
            images/
                train/
                    image1.png
                    image2.png
            labels/
                train/
                    image1.txt
                    image2.txt

    Each label file contains one line per symbol:
        class_id x_center y_center width height
    All coordinates are normalized to [0, 1].

    Args:
        documents: List of DocumentAnnotation objects.
        output_dir: Root output directory.
        split: Split name (train, val, test).
        copy_images: Whether to copy source images to output dir.
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images" / split
    labels_dir = output_dir / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    num_files = 0
    num_symbols = 0

    for doc in documents:
        if not doc.symbols:
            continue

        # Determine output filename from image_id
        filename = doc.image_id
        label_path = labels_dir / f"{filename}.txt"

        # Write YOLO label file
        lines = []
        for sym in doc.symbols:
            x_c, y_c, w, h = sym.bbox
            # Clamp to [0, 1]
            x_c = max(0.0, min(1.0, x_c))
            y_c = max(0.0, min(1.0, y_c))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))
            lines.append(f"{sym.class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

        with open(label_path, "w") as f:
            f.write("\n".join(lines))

        # Copy image if requested and source exists
        if copy_images and doc.image_path:
            src_img = Path(doc.image_path)
            if src_img.exists():
                dst_img = images_dir / f"{filename}{src_img.suffix}"
                if not dst_img.exists():
                    shutil.copy2(src_img, dst_img)

        num_files += 1
        num_symbols += len(doc.symbols)

    logger.info(
        f"Wrote {num_files} YOLO label files ({num_symbols} symbols) "
        f"to {labels_dir}"
    )


def verify_yolo_dataset(dataset_dir: str | Path) -> dict:
    """Verify a YOLO-format dataset for completeness.

    Checks:
    - Every label file has a corresponding image
    - Every image has a corresponding label file
    - Label format is valid (5 floats per line)

    Returns:
        Dict with verification results.
    """
    dataset_dir = Path(dataset_dir)
    results = {
        "valid": True,
        "errors": [],
        "splits": {},
    }

    for split in ["train", "val", "test"]:
        images_dir = dataset_dir / "images" / split
        labels_dir = dataset_dir / "labels" / split

        if not images_dir.exists():
            continue

        image_stems = {p.stem for p in images_dir.iterdir() if p.is_file()}
        label_stems = {p.stem for p in labels_dir.iterdir() if p.suffix == ".txt"}

        missing_labels = image_stems - label_stems
        missing_images = label_stems - image_stems

        # Validate label format
        format_errors = []
        total_symbols = 0
        for label_path in labels_dir.glob("*.txt"):
            with open(label_path) as f:
                for line_num, line in enumerate(f, 1):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        format_errors.append(f"{label_path.name}:{line_num}: expected 5 values")
                        continue
                    try:
                        class_id = int(parts[0])
                        coords = [float(x) for x in parts[1:]]
                        if not all(0 <= c <= 1 for c in coords):
                            format_errors.append(
                                f"{label_path.name}:{line_num}: coords out of [0,1]"
                            )
                        total_symbols += 1
                    except ValueError:
                        format_errors.append(f"{label_path.name}:{line_num}: invalid number")

        split_result = {
            "num_images": len(image_stems),
            "num_labels": len(label_stems),
            "total_symbols": total_symbols,
            "missing_labels": len(missing_labels),
            "missing_images": len(missing_images),
            "format_errors": len(format_errors),
        }
        results["splits"][split] = split_result

        if missing_labels or missing_images or format_errors:
            results["valid"] = False
            results["errors"].extend(format_errors[:10])

    return results
