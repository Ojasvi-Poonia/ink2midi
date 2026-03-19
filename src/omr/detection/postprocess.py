"""Post-processing for music symbol detections."""

from omr.data.graph_builder import Detection
from omr.utils.logging import get_logger

logger = get_logger("detection.postprocess")


def postprocess_detections(
    detections: list[Detection],
    score_threshold: float = 0.3,
    min_area: int = 16,
    merge_noteheads_iou: float = 0.7,
    validate_stems: bool = True,
    stem_min_aspect: float = 2.0,
) -> list[Detection]:
    """Post-process detections with music-aware filtering.

    Args:
        detections: Raw detections from the model.
        score_threshold: Minimum confidence score.
        min_area: Minimum bounding box area in pixels.
        merge_noteheads_iou: IoU threshold for merging duplicate noteheads.
        validate_stems: If True, filter stems that aren't elongated.
        stem_min_aspect: Minimum height/width ratio for stems.

    Returns:
        Filtered and cleaned detections.
    """
    filtered = []

    for det in detections:
        # Confidence filter
        if det.confidence < score_threshold:
            continue

        # Minimum area filter
        if det.area < min_area:
            continue

        # Stem shape validation: stems should be tall and narrow
        if validate_stems and det.class_name == "stem":
            if det.height == 0 or det.height / max(det.width, 1) < stem_min_aspect:
                continue

        filtered.append(det)

    # Merge overlapping noteheads (common duplicate detection)
    if merge_noteheads_iou < 1.0:
        filtered = _merge_duplicate_noteheads(filtered, merge_noteheads_iou)

    logger.debug(
        f"Post-processing: {len(detections)} -> {len(filtered)} detections"
    )
    return filtered


def _merge_duplicate_noteheads(
    detections: list[Detection], iou_threshold: float
) -> list[Detection]:
    """Merge overlapping notehead detections, keeping the highest confidence."""
    noteheads = [d for d in detections if "notehead" in d.class_name]
    others = [d for d in detections if "notehead" not in d.class_name]

    if len(noteheads) <= 1:
        return detections

    # Sort by confidence (highest first)
    noteheads.sort(key=lambda d: d.confidence, reverse=True)

    keep = []
    suppressed = set()

    for i, det_i in enumerate(noteheads):
        if i in suppressed:
            continue
        keep.append(det_i)

        for j in range(i + 1, len(noteheads)):
            if j in suppressed:
                continue
            if _compute_iou(det_i.bbox, noteheads[j].bbox) > iou_threshold:
                suppressed.add(j)

    return keep + others


def _compute_iou(
    bbox1: tuple[float, ...], bbox2: tuple[float, ...]
) -> float:
    """Compute IoU between two (x1, y1, x2, y2) bounding boxes."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0
    return intersection / union
