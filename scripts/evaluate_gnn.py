#!/usr/bin/env python3
"""Evaluate a trained GNN edge classifier on the writer-disjoint TEST split.

Reports edge-level Precision, Recall, F1, AUROC at the default 0.5 threshold
plus a threshold sweep, and supports head-to-head GAT vs GCN comparison.

Usage:
    # Single model
    python scripts/evaluate_gnn.py \\
        --weights checkpoints/gnn/gat_best.pt \\
        --model-type gat

    # Side-by-side GAT vs GCN
    python scripts/evaluate_gnn.py \\
        --weights checkpoints/gnn/gat_best.pt \\
        --weights-baseline checkpoints/gnn_gcn/gcn_best.pt \\
        --model-type gat --baseline-type gcn \\
        --output evaluation/gnn_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from omr.data.graph_builder import NotationGraphBuilder
from omr.data.muscima_parser import MUSCIMAParser
from omr.relationship.gat_model import NotationGAT
from omr.relationship.gcn_model import NotationGCN
from omr.relationship.graph_dataset import NotationGraphDataset
from omr.utils.device import get_device
from omr.utils.logging import get_logger, setup_logging
from omr.utils.reproducibility import set_seed

logger = get_logger("scripts.evaluate_gnn")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _find_annotations_dir(base_dir: Path) -> Path:
    """Locate the MUSCIMA++ XML annotations folder (mirrors train_gnn.py)."""
    for candidate in (
        base_dir / "v2.0" / "data" / "annotations",
        base_dir / "data" / "annotations",
        base_dir / "annotations",
        base_dir,
    ):
        if candidate.exists() and any(candidate.glob("CVC-MUSCIMA_*.xml")):
            return candidate
    for xml in base_dir.rglob("CVC-MUSCIMA_*.xml"):
        if len(list(xml.parent.glob("CVC-MUSCIMA_*.xml"))) > 10:
            return xml.parent
    raise FileNotFoundError(f"No MUSCIMA++ annotation XMLs under {base_dir}")


def _load_test_documents(args) -> list:
    """Parse MUSCIMA++ and return only the writer-disjoint TEST documents."""
    splits_file = Path(args.splits_file)
    if not splits_file.exists():
        raise FileNotFoundError(
            f"Splits file not found: {splits_file}. "
            f"Run scripts/prepare_data.py first to generate writer-disjoint splits."
        )
    with open(splits_file) as f:
        splits = json.load(f)
    test_ids = set(splits.get("test", []))
    if not test_ids:
        raise ValueError("Splits file has no 'test' key. Re-run prepare_data.py.")

    ann_dir = _find_annotations_dir(Path(args.annotations_dir))
    docs = MUSCIMAParser().parse_directory(ann_dir, Path(args.images_dir))
    test_docs = [d for d in docs if d.image_id in test_ids]
    logger.info(
        f"Test documents: {len(test_docs)} "
        f"(of {len(docs)} parsed; {len(test_ids)} ids in test split)"
    )
    if not test_docs:
        raise RuntimeError(
            "No test documents matched. Check that splits['test'] image_ids "
            "match parsed document.image_id values."
        )
    return test_docs


def _build_model(model_type: str, hidden_dim: int, num_heads: int,
                 num_layers: int, dropout: float) -> torch.nn.Module:
    if model_type == "gat":
        return NotationGAT(
            hidden_dim=hidden_dim, num_heads=num_heads,
            num_layers=num_layers, dropout=dropout,
        )
    if model_type == "gcn":
        return NotationGCN(
            hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout,
        )
    raise ValueError(f"Unknown model type: {model_type}")


def _load_checkpoint(model: torch.nn.Module, weights: Path,
                     device: torch.device) -> torch.nn.Module:
    state = torch.load(weights, map_location=device)
    # Trainer may save the raw state_dict or a dict with "model_state_dict"
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device).eval()
    return model


# ----------------------------------------------------------------------------
# Core evaluation
# ----------------------------------------------------------------------------
@torch.no_grad()
def collect_predictions(model: torch.nn.Module,
                        loader: DataLoader,
                        device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Return (probs, labels) flattened across the entire test loader."""
    all_probs, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch).squeeze(-1)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        labels = batch.edge_label.detach().cpu().numpy().astype(np.int64)
        all_probs.append(probs)
        all_labels.append(labels)
    return np.concatenate(all_probs), np.concatenate(all_labels)


def metrics_at_threshold(probs: np.ndarray, labels: np.ndarray,
                         threshold: float) -> dict:
    preds = (probs >= threshold).astype(np.int64)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(1, len(labels))
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def auroc(probs: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC. Uses sklearn if available, otherwise a self-contained
    rank-based computation."""
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, probs))
    except Exception:
        # Mann-Whitney U formulation (no sklearn dependency)
        order = np.argsort(probs)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(probs) + 1)
        n_pos = int(labels.sum())
        n_neg = len(labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        sum_pos = ranks[labels == 1].sum()
        return float((sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def evaluate_one(model: torch.nn.Module, loader: DataLoader,
                 device: torch.device,
                 thresholds: list[float]) -> dict:
    probs, labels = collect_predictions(model, loader, device)
    sweep = [metrics_at_threshold(probs, labels, t) for t in thresholds]
    best_f1 = max(sweep, key=lambda m: m["f1"])
    return {
        "n_edges": int(len(labels)),
        "positive_rate": float(labels.mean()),
        "auroc": auroc(probs, labels),
        "at_0.5": metrics_at_threshold(probs, labels, 0.5),
        "best_f1": best_f1,
        "sweep": sweep,
    }


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained GNN on the test split")
    # Primary model
    parser.add_argument("--weights", required=True,
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--model-type", choices=["gat", "gcn"], default="gat")

    # Optional baseline for side-by-side comparison
    parser.add_argument("--weights-baseline", default=None,
                        help="Optional second checkpoint for ablation row")
    parser.add_argument("--baseline-type", choices=["gat", "gcn"], default="gcn")

    # Architecture (must match training)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--k-neighbors", type=int, default=8)
    parser.add_argument("--max-distance", type=float, default=200.0)

    # Data
    parser.add_argument("--annotations-dir", default="data/raw/muscima_pp_v2")
    parser.add_argument("--images-dir", default="data/raw/muscima_pp_v2/images")
    parser.add_argument("--splits-file", default="data/splits/muscima_splits.json")

    # Eval
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="evaluation/gnn_results.json")

    args = parser.parse_args()
    setup_logging("INFO")
    set_seed(args.seed, fast=False)
    device = get_device(args.device)

    # 1. Build the test loader once and reuse for both models
    test_docs = _load_test_documents(args)
    builder = NotationGraphBuilder(
        k_neighbors=args.k_neighbors, max_distance_px=args.max_distance,
    )
    test_set = NotationGraphDataset(test_docs, builder, use_gt_detections=True)
    if len(test_set) == 0:
        raise RuntimeError("No test graphs could be built.")
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, pin_memory=(device.type == "cuda"))

    thresholds = [round(0.05 * i, 2) for i in range(2, 19)]  # 0.10 ... 0.90

    # 2. Primary model
    logger.info(f"Loading primary model: {args.model_type.upper()} @ {args.weights}")
    primary = _load_checkpoint(
        _build_model(args.model_type, args.hidden_dim, args.num_heads,
                     args.num_layers, args.dropout),
        Path(args.weights), device,
    )
    primary_metrics = evaluate_one(primary, loader, device, thresholds)
    logger.info(
        f"[{args.model_type.upper()}] AUROC={primary_metrics['auroc']:.4f}  "
        f"@0.5: P={primary_metrics['at_0.5']['precision']:.4f} "
        f"R={primary_metrics['at_0.5']['recall']:.4f} "
        f"F1={primary_metrics['at_0.5']['f1']:.4f}  "
        f"BestF1={primary_metrics['best_f1']['f1']:.4f} "
        f"@thr={primary_metrics['best_f1']['threshold']}"
    )

    results = {
        "n_test_graphs": len(test_set),
        "primary": {"model_type": args.model_type, "weights": args.weights,
                    **primary_metrics},
    }

    # 3. Baseline model (optional)
    if args.weights_baseline:
        logger.info(
            f"Loading baseline model: {args.baseline_type.upper()} "
            f"@ {args.weights_baseline}"
        )
        baseline = _load_checkpoint(
            _build_model(args.baseline_type, args.hidden_dim, args.num_heads,
                         args.num_layers, args.dropout),
            Path(args.weights_baseline), device,
        )
        baseline_metrics = evaluate_one(baseline, loader, device, thresholds)
        logger.info(
            f"[{args.baseline_type.upper()}] AUROC={baseline_metrics['auroc']:.4f}  "
            f"@0.5: P={baseline_metrics['at_0.5']['precision']:.4f} "
            f"R={baseline_metrics['at_0.5']['recall']:.4f} "
            f"F1={baseline_metrics['at_0.5']['f1']:.4f}  "
            f"BestF1={baseline_metrics['best_f1']['f1']:.4f} "
            f"@thr={baseline_metrics['best_f1']['threshold']}"
        )
        results["baseline"] = {
            "model_type": args.baseline_type,
            "weights": args.weights_baseline,
            **baseline_metrics,
        }
        delta_f1 = (primary_metrics["at_0.5"]["f1"]
                    - baseline_metrics["at_0.5"]["f1"])
        logger.info(
            f"Delta F1 ({args.model_type.upper()} - {args.baseline_type.upper()}) "
            f"@0.5 = {delta_f1:+.4f}"
        )
        results["delta_f1_at_0.5"] = float(delta_f1)

    # 4. Persist
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    logger.info(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
