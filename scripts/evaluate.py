#!/usr/bin/env python3
"""Run full evaluation suite across all modules."""

import argparse
import json
from pathlib import Path

from omr.utils.logging import setup_logging, get_logger

logger = get_logger("scripts.evaluate")


def evaluate_detection(args):
    """Evaluate symbol detection performance."""
    from omr.detection.yolo_trainer import SymbolDetector

    logger.info("=== Symbol Detection Evaluation ===")

    detector = SymbolDetector(args.detector_weights)
    metrics = detector.evaluate(args.muscima_yaml, device=args.device)

    logger.info(f"mAP@0.5: {metrics['mAP50']:.4f}")
    logger.info(f"mAP@0.5:0.95: {metrics['mAP50_95']:.4f}")

    return {"detection": metrics}


def evaluate_gnn(args):
    """Evaluate GNN relationship parsing performance."""
    logger.info("=== GNN Relationship Parsing Evaluation ===")

    gnn_path = Path(args.gnn_weights)
    if not gnn_path.exists():
        logger.warning(f"GNN weights not found at {gnn_path}, skipping")
        return {}

    # Load test graphs and run evaluation
    # (Requires MUSCIMA++ test set graphs)
    logger.info("GNN evaluation requires pre-built test graphs")
    logger.info("Run: python scripts/train_gnn.py first")

    return {}


def evaluate_midi(args):
    """Evaluate end-to-end MIDI generation fidelity."""
    logger.info("=== MIDI Fidelity Evaluation ===")

    gt_midi_dir = Path(args.gt_midi_dir) if args.gt_midi_dir else None
    if gt_midi_dir is None or not gt_midi_dir.exists():
        logger.warning("Ground truth MIDI directory not specified or not found")
        return {}

    from omr.evaluation.midi_metrics import compute_midi_metrics

    pred_midi_dir = Path(args.pred_midi_dir)
    results = []

    for pred_midi in pred_midi_dir.glob("*.mid"):
        gt_midi = gt_midi_dir / pred_midi.name
        if gt_midi.exists():
            metrics = compute_midi_metrics(pred_midi, gt_midi)
            results.append(metrics)
            logger.info(
                f"{pred_midi.name}: P={metrics['note_precision']:.3f}, "
                f"R={metrics['note_recall']:.3f}, F1={metrics['note_f1']:.3f}"
            )

    if results:
        avg_f1 = sum(m["note_f1"] for m in results) / len(results)
        avg_pitch = sum(m["pitch_accuracy"] for m in results) / len(results)
        logger.info(f"Average Note F1: {avg_f1:.4f}")
        logger.info(f"Average Pitch Accuracy: {avg_pitch:.4f}")

    return {"midi": results}


def main():
    parser = argparse.ArgumentParser(description="Evaluate OMR system")
    parser.add_argument(
        "--detector-weights",
        default="checkpoints/detection/yolov8s_muscima_finetune/weights/best.pt",
    )
    parser.add_argument("--gnn-weights", default="checkpoints/gnn/gat_best.pt")
    parser.add_argument("--muscima-yaml", default="configs/detection/yolov8_muscima.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gt-midi-dir", default=None)
    parser.add_argument("--pred-midi-dir", default="outputs/midi")
    parser.add_argument("--output", default="outputs/evaluation_results.json")
    args = parser.parse_args()

    setup_logging("INFO")

    all_results = {}

    # Detection evaluation
    if Path(args.detector_weights).exists():
        all_results.update(evaluate_detection(args))

    # GNN evaluation
    all_results.update(evaluate_gnn(args))

    # MIDI evaluation
    all_results.update(evaluate_midi(args))

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"Evaluation results saved to {output_path}")


if __name__ == "__main__":
    main()
