"""Parse DeepScores V2 JSON annotations into unified format."""

import json
from pathlib import Path

from omr.data.muscima_parser import (
    CLASS_NAME_TO_ID,
    DocumentAnnotation,
    SymbolAnnotation,
)
from omr.utils.logging import get_logger

logger = get_logger("data.deepscores_parser")

# Map DeepScores V2 category names to our 26 unified classes
# DeepScores has 135 classes; we map the relevant ones
DEEPSCORES_CLASS_MAP = {
    # Noteheads
    "noteheadBlack": "notehead_filled",
    "noteheadHalf": "notehead_half",
    "noteheadWhole": "notehead_whole",
    "noteheadBlackSmall": "notehead_filled",
    # Stems
    "stem": "stem",
    # Beams
    "beam": "beam",
    # Flags
    "flag8thUp": "flag_eighth_up",
    "flag8thDown": "flag_eighth_down",
    "flag16thUp": "flag_sixteenth_up",
    "flag16thDown": "flag_sixteenth_down",
    # Rests
    "restWhole": "rest_whole",
    "restHalf": "rest_half",
    "restQuarter": "rest_quarter",
    "rest8th": "rest_eighth",
    "rest16th": "rest_sixteenth",
    # Accidentals
    "accidentalSharp": "sharp",
    "accidentalFlat": "flat",
    "accidentalNatural": "natural",
    "accidentalDoubleSharp": "double_sharp",
    # Clefs
    "gClef": "treble_clef",
    "fClef": "bass_clef",
    "cClef": "alto_clef",
    # Time signatures (mapped to generic digit)
    "timeSig0": "time_sig_digit",
    "timeSig1": "time_sig_digit",
    "timeSig2": "time_sig_digit",
    "timeSig3": "time_sig_digit",
    "timeSig4": "time_sig_digit",
    "timeSig5": "time_sig_digit",
    "timeSig6": "time_sig_digit",
    "timeSig7": "time_sig_digit",
    "timeSig8": "time_sig_digit",
    "timeSig9": "time_sig_digit",
    "timeSigCommon": "time_sig_digit",
    "timeSigCutCommon": "time_sig_digit",
    # Barlines
    "barline": "bar_line",
    "barlineDouble": "double_bar_line",
    "barlineFinal": "double_bar_line",
    "barlineReverseFinal": "double_bar_line",
    # Dots
    "augmentationDot": "dot",
    # Ties and slurs
    "tie": "tie_slur",
    "slur": "tie_slur",
}


class DeepScoresParser:
    """Parse DeepScores V2 annotation files.

    DeepScores V2 uses a custom JSON format (not standard COCO):
    - categories: dict mapping string ID -> {name, annotation_set, color}
    - annotations: dict mapping string ID -> {a_bbox, o_bbox, cat_id, area, img_id, comments}
    - images: list of {id, filename, width, height, ann_ids}
    - annotation_sets: ['deepscores', 'muscima++']

    Each annotation has cat_id as a list of string IDs (one per annotation set).
    We use the 'deepscores' annotation set (first element).
    """

    def __init__(self):
        self.class_map = DEEPSCORES_CLASS_MAP
        self.class_to_id = CLASS_NAME_TO_ID

    def parse_annotations(
        self,
        json_path: str | Path,
        images_dir: str | Path,
    ) -> list[DocumentAnnotation]:
        """Parse DeepScores V2 annotation JSON.

        Args:
            json_path: Path to the annotation JSON file.
            images_dir: Directory containing the score images.

        Returns:
            List of DocumentAnnotation objects.
        """
        json_path = Path(json_path)
        images_dir = Path(images_dir)

        logger.info(f"Loading DeepScores V2 annotations from {json_path}")
        with open(json_path) as f:
            data = json.load(f)

        # Build category ID -> name mapping
        # categories is a dict: {str_id: {name, annotation_set, color}}
        cat_id_to_name = {}
        categories = data.get("categories", {})
        if isinstance(categories, dict):
            for cat_id_str, cat_info in categories.items():
                # Only use 'deepscores' annotation set categories
                if cat_info.get("annotation_set") == "deepscores":
                    cat_id_to_name[cat_id_str] = cat_info["name"]
        elif isinstance(categories, list):
            # Fallback for standard COCO format
            for cat in categories:
                cat_id_to_name[str(cat["id"])] = cat["name"]

        logger.info(f"Found {len(cat_id_to_name)} DeepScores category mappings")

        # Build image ID -> info mapping
        # images is a list of {id, filename, width, height, ann_ids}
        image_info = {}
        for img in data.get("images", []):
            img_id = str(img["id"])
            # DeepScores uses 'filename' not 'file_name'
            fname = img.get("filename") or img.get("file_name", "")
            image_info[img_id] = {
                "filename": fname,
                "width": img["width"],
                "height": img["height"],
            }

        # Parse annotations
        # annotations is a dict: {str_id: {a_bbox, o_bbox, cat_id, area, img_id, comments}}
        annotations_raw = data.get("annotations", {})
        if isinstance(annotations_raw, dict):
            ann_items = list(annotations_raw.values())
        elif isinstance(annotations_raw, list):
            ann_items = annotations_raw
        else:
            ann_items = []

        # Group annotations by image
        anns_by_image: dict[str, list[dict]] = {}
        for ann in ann_items:
            # img_id can be string or int depending on format
            img_id = str(ann.get("img_id") or ann.get("image_id", ""))
            if img_id not in anns_by_image:
                anns_by_image[img_id] = []
            anns_by_image[img_id].append(ann)

        # Parse each image's annotations
        documents = []
        for img_id, img_info in image_info.items():
            img_w = img_info["width"]
            img_h = img_info["height"]
            image_path = images_dir / img_info["filename"]

            doc = DocumentAnnotation(
                image_id=img_id,
                image_path=str(image_path),
                image_width=img_w,
                image_height=img_h,
            )

            for i, ann in enumerate(anns_by_image.get(img_id, [])):
                # cat_id is a list of string IDs (one per annotation set)
                # Use the first element (deepscores annotation set)
                cat_id_raw = ann.get("cat_id")
                if isinstance(cat_id_raw, list):
                    cat_id_str = str(cat_id_raw[0]) if cat_id_raw else None
                else:
                    cat_id_str = str(cat_id_raw) if cat_id_raw is not None else None

                if cat_id_str is None:
                    continue

                cat_name = cat_id_to_name.get(cat_id_str)
                if cat_name is None:
                    continue

                unified_name = self.class_map.get(cat_name)
                if unified_name is None:
                    continue

                class_id = self.class_to_id[unified_name]

                # Parse aligned bounding box: [x_min, y_min, x_max, y_max]
                a_bbox = ann.get("a_bbox", [0, 0, 0, 0])
                if len(a_bbox) < 4:
                    continue
                x_min, y_min, x_max, y_max = a_bbox[:4]

                width = x_max - x_min
                height = y_max - y_min

                if width <= 0 or height <= 0:
                    continue

                # Absolute bbox (top, left, width, height)
                bbox_abs = (int(y_min), int(x_min), int(width), int(height))

                # Normalized bbox (x_center, y_center, w, h)
                x_center = (x_min + width / 2) / img_w
                y_center = (y_min + height / 2) / img_h
                w_norm = width / img_w
                h_norm = height / img_h
                bbox_norm = (x_center, y_center, w_norm, h_norm)

                symbol = SymbolAnnotation(
                    symbol_id=i,
                    class_name=unified_name,
                    class_id=class_id,
                    bbox=bbox_norm,
                    bbox_abs=bbox_abs,
                    image_id=img_id,
                )
                doc.symbols.append(symbol)

            documents.append(doc)

        total_symbols = sum(len(d.symbols) for d in documents)
        logger.info(
            f"Parsed {len(documents)} images with {total_symbols} mapped symbols "
            f"(from {len(ann_items)} total annotations)"
        )
        return documents
