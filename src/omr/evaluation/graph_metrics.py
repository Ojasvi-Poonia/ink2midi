"""Graph relationship evaluation metrics."""

from omr.utils.logging import get_logger

logger = get_logger("evaluation.graph_metrics")


def compute_edge_metrics(
    pred_edges: list[tuple[int, int, float]],
    gt_edges: list[tuple[int, int]],
) -> dict:
    """Compute edge prediction metrics.

    Args:
        pred_edges: Predicted edges as (src, tgt, probability).
        gt_edges: Ground truth edges as (src, tgt).

    Returns:
        Dict with precision, recall, F1.
    """
    pred_set = {(src, tgt) for src, tgt, _ in pred_edges}
    gt_set = set(gt_edges)

    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "num_predicted": len(pred_set),
        "num_gt": len(gt_set),
    }


def compute_per_type_metrics(
    pred_edges: list[tuple[int, int, float]],
    gt_edges: list[tuple[int, int]],
    edge_types: dict[tuple[int, int], str],
) -> dict[str, dict]:
    """Compute metrics broken down by relationship type.

    Args:
        pred_edges: Predicted edges.
        gt_edges: Ground truth edges.
        edge_types: Mapping from (src, tgt) to relationship type
                    (e.g., "notehead-stem", "stem-beam").

    Returns:
        Dict mapping relationship type to metrics.
    """
    pred_set = {(src, tgt) for src, tgt, _ in pred_edges}
    gt_set = set(gt_edges)

    # Group GT edges by type
    type_gt: dict[str, set] = {}
    for edge in gt_set:
        etype = edge_types.get(edge, "unknown")
        if etype not in type_gt:
            type_gt[etype] = set()
        type_gt[etype].add(edge)

    results = {}
    for etype, gt_typed in type_gt.items():
        tp = len(pred_set & gt_typed)
        fn = len(gt_typed - pred_set)
        # FP for this type: predicted edges that hit a GT edge of this type's nodes
        # but aren't in GT
        fp_candidates = {e for e in pred_set if e not in gt_set}
        fp = len(fp_candidates)  # Simplified; type-specific FP needs node info

        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)

        results[etype] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tp": tp,
            "fn": fn,
        }

    return results
