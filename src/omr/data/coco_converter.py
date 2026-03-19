"""Convert unified annotations to COCO JSON format."""

import json
from pathlib import Path

from omr.data.muscima_parser import CLASS_NAME_TO_ID, DocumentAnnotation
from omr.utils.logging import get_logger

logger = get_logger("data.coco_converter")


def convert_to_coco(
    documents: list[DocumentAnnotation],
    output_path: str | Path,
) -> dict:
    """Convert unified annotations to COCO detection format.

    Creates a single JSON file with:
    - images: [{id, file_name, width, height}, ...]
    - annotations: [{id, image_id, category_id, bbox, area, iscrowd}, ...]
    - categories: [{id, name, supercategory}, ...]

    COCO bbox format: [x_min, y_min, width, height] in absolute pixels.

    Args:
        documents: List of DocumentAnnotation objects.
        output_path: Path for the output JSON file.

    Returns:
        The COCO dict.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build categories
    categories = []
    for name, cid in sorted(CLASS_NAME_TO_ID.items(), key=lambda x: x[1]):
        supercategory = _get_supercategory(name)
        categories.append({
            "id": cid,
            "name": name,
            "supercategory": supercategory,
        })

    images = []
    annotations = []
    ann_id = 0

    for img_idx, doc in enumerate(documents):
        images.append({
            "id": img_idx,
            "file_name": Path(doc.image_path).name if doc.image_path else f"{doc.image_id}.png",
            "width": doc.image_width,
            "height": doc.image_height,
        })

        for sym in doc.symbols:
            # Convert from (top, left, w, h) to COCO [x_min, y_min, w, h]
            top, left, w, h = sym.bbox_abs
            bbox_coco = [left, top, w, h]
            area = w * h

            annotations.append({
                "id": ann_id,
                "image_id": img_idx,
                "category_id": sym.class_id,
                "bbox": bbox_coco,
                "area": area,
                "iscrowd": 0,
            })
            ann_id += 1

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    with open(output_path, "w") as f:
        json.dump(coco_dict, f, indent=2)

    logger.info(
        f"Wrote COCO JSON: {len(images)} images, {len(annotations)} annotations -> {output_path}"
    )
    return coco_dict


def _get_supercategory(class_name: str) -> str:
    """Map class name to a supercategory."""
    if "notehead" in class_name:
        return "notehead"
    if class_name == "stem":
        return "stem"
    if class_name == "beam":
        return "beam"
    if "flag" in class_name:
        return "flag"
    if "rest" in class_name:
        return "rest"
    if class_name in ("sharp", "flat", "natural", "double_sharp"):
        return "accidental"
    if "clef" in class_name:
        return "clef"
    if "time_sig" in class_name:
        return "time_signature"
    if "bar_line" in class_name:
        return "barline"
    if class_name == "dot":
        return "dot"
    if class_name == "tie_slur":
        return "articulation"
    return "other"
