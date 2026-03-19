"""Detection evaluation metrics."""

import numpy as np

from omr.data.graph_builder import Detection
from omr.data.muscima_parser import CLASS_NAME_TO_ID
from omr.utils.logging import get_logger

logger = get_logger("evaluation.detection_metrics")

ID_TO_CLASS_NAME = {v: k for k, v in CLASS_NAME_TO_ID.items()}


def compute_detection_metrics(
    pred_detections: list[Detection],
    gt_detections: list[Detection],
    iou_threshold: float = 0.5,
) -> dict:
    """Compute detection metrics: precision, recall, F1, per-class AP.

    Args:
        pred_detections: Predicted detections.
        gt_detections: Ground truth detections.
        iou_threshold: IoU threshold for a match.

    Returns:
        Dict with overall and per-class metrics.
    """
    from omr.detection.postprocess import _compute_iou

    # Match predictions to ground truth
    matched_gt = set()
    tp = 0
    fp = 0

    # Sort predictions by confidence (highest first)
    preds_sorted = sorted(pred_detections, key=lambda d: d.confidence, reverse=True)

    for pred in preds_sorted:
        best_iou = 0.0
        best_gt_idx = -1

        for i, gt in enumerate(gt_detections):
            if i in matched_gt:
                continue
            if pred.class_id != gt.class_id:
                continue

            iou = _compute_iou(pred.bbox, gt.bbox)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(gt_detections) - len(matched_gt)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    # Per-class metrics
    per_class = {}
    all_class_ids = set(d.class_id for d in gt_detections) | set(
        d.class_id for d in pred_detections
    )

    for cid in sorted(all_class_ids):
        class_preds = [d for d in pred_detections if d.class_id == cid]
        class_gt = [d for d in gt_detections if d.class_id == cid]

        c_matched = set()
        c_tp = 0
        c_fp = 0

        class_preds_sorted = sorted(class_preds, key=lambda d: d.confidence, reverse=True)
        for pred in class_preds_sorted:
            best_iou = 0.0
            best_idx = -1
            for j, gt in enumerate(class_gt):
                if j in c_matched:
                    continue
                iou = _compute_iou(pred.bbox, gt.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j
            if best_iou >= iou_threshold and best_idx >= 0:
                c_tp += 1
                c_matched.add(best_idx)
            else:
                c_fp += 1

        c_fn = len(class_gt) - len(c_matched)
        c_prec = c_tp / max(c_tp + c_fp, 1)
        c_rec = c_tp / max(c_tp + c_fn, 1)
        c_f1 = 2 * c_prec * c_rec / max(c_prec + c_rec, 1e-8)

        class_name = ID_TO_CLASS_NAME.get(cid, f"class_{cid}")
        per_class[class_name] = {
            "precision": c_prec,
            "recall": c_rec,
            "f1": c_f1,
            "tp": c_tp,
            "fp": c_fp,
            "fn": c_fn,
        }

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "per_class": per_class,
    }
