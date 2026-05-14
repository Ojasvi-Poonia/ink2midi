#!/usr/bin/env python3
"""End-to-end MIDI evaluation using mir_eval (industry standard).

Pairs predicted MIDI files (output of run_inference.py) with ground-truth MIDI
and reports note-level Precision / Recall / F1 in two settings:

  1. Pitch + onset only      (lenient — duration ignored)
  2. Pitch + onset + offset  (strict — full note transcription)

Falls back to the in-repo MIDI matcher (omr.evaluation.midi_metrics) if
mir_eval is not installed.

Optional: auto-build ground-truth MIDI from MUSCIMA++ MusicXML using music21.

Usage:
    # Just evaluate (predicted + groundtruth dirs already exist)
    python scripts/evaluate_midi.py \\
        --pred-dir evaluation/predictions \\
        --gt-dir   evaluation/groundtruth \\
        --output   evaluation/midi_results.json

    # Auto-generate ground truth from MUSCIMA++ MusicXML
    python scripts/evaluate_midi.py \\
        --pred-dir evaluation/predictions \\
        --gt-dir   evaluation/groundtruth \\
        --musicxml-dir data/raw/muscima_pp_v2/v2.0/data/musicxml \\
        --build-gt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from omr.utils.logging import get_logger, setup_logging

logger = get_logger("scripts.evaluate_midi")


# ---------------------------------------------------------------------------
# Optional: build GT MIDI from MusicXML
# ---------------------------------------------------------------------------
def build_groundtruth_from_musicxml(musicxml_dir: Path, gt_dir: Path,
                                    only_stems: set[str] | None = None) -> int:
    """Convert MUSCIMA++ MusicXML files to MIDI.

    Returns number of files written. Requires `music21`.
    """
    from music21 import converter

    gt_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    xmls = sorted(p for p in musicxml_dir.rglob("*.xml")
                  if "CVC-MUSCIMA" in p.name)
    if not xmls:
        xmls = sorted(p for p in musicxml_dir.rglob("*.xml"))
    for xml in xmls:
        if only_stems and xml.stem not in only_stems:
            continue
        out = gt_dir / f"{xml.stem}.mid"
        if out.exists():
            written += 1
            continue
        try:
            score = converter.parse(xml)
            score.write("midi", fp=out)
            written += 1
        except Exception as e:
            logger.warning(f"Skipped {xml.name}: {e}")
    logger.info(f"Wrote {written} ground-truth MIDI files to {gt_dir}")
    return written


# ---------------------------------------------------------------------------
# mir_eval-based scoring (preferred)
# ---------------------------------------------------------------------------
def _midi_to_arrays(path: Path):
    """Return (intervals, pitches) numpy arrays for a MIDI file."""
    import numpy as np
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(path))
    notes = [n for inst in pm.instruments for n in inst.notes]
    if not notes:
        return np.zeros((0, 2)), np.zeros((0,))
    intervals = np.array([[n.start, n.end] for n in notes], dtype=np.float64)
    pitches = np.array([float(n.pitch) for n in notes], dtype=np.float64)
    # Convert MIDI pitch to Hz for mir_eval (it expects Hz, not MIDI numbers)
    hz = 440.0 * (2.0 ** ((pitches - 69.0) / 12.0))
    return intervals, hz


def _evaluate_with_mir_eval(pred_path: Path, gt_path: Path,
                            onset_tolerance: float) -> dict:
    import mir_eval

    p_int, p_hz = _midi_to_arrays(pred_path)
    g_int, g_hz = _midi_to_arrays(gt_path)
    if len(p_int) == 0 or len(g_int) == 0:
        return {
            "n_pred": int(len(p_int)),
            "n_gt": int(len(g_int)),
            "pitch_onset": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "pitch_onset_offset": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        }

    p1, r1, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        g_int, g_hz, p_int, p_hz,
        onset_tolerance=onset_tolerance, offset_ratio=None,
    )
    p2, r2, f2, _ = mir_eval.transcription.precision_recall_f1_overlap(
        g_int, g_hz, p_int, p_hz,
        onset_tolerance=onset_tolerance, offset_ratio=0.2,
    )
    return {
        "n_pred": int(len(p_int)),
        "n_gt": int(len(g_int)),
        "pitch_onset": {"precision": float(p1), "recall": float(r1), "f1": float(f1)},
        "pitch_onset_offset": {"precision": float(p2), "recall": float(r2), "f1": float(f2)},
    }


# ---------------------------------------------------------------------------
# In-repo fallback (uses omr.evaluation.midi_metrics)
# ---------------------------------------------------------------------------
def _evaluate_with_repo(pred_path: Path, gt_path: Path) -> dict:
    from omr.evaluation.midi_metrics import compute_midi_metrics

    m = compute_midi_metrics(pred_path, gt_path)
    return {
        "n_pred": int(m.get("num_predicted", 0)),
        "n_gt": int(m.get("num_gt", 0)),
        "pitch_onset": {
            "precision": float(m.get("note_precision", 0.0)),
            "recall": float(m.get("note_recall", 0.0)),
            "f1": float(m.get("note_f1", 0.0)),
        },
        "pitch_onset_offset": None,  # not provided by the repo metric
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predicted MIDI vs ground-truth MIDI")
    parser.add_argument("--pred-dir", required=True,
                        help="Directory of predicted .mid files (run_inference output)")
    parser.add_argument("--gt-dir", required=True,
                        help="Directory of ground-truth .mid files")
    parser.add_argument("--musicxml-dir", default=None,
                        help="If set together with --build-gt, generate GT MIDI from MusicXML")
    parser.add_argument("--build-gt", action="store_true",
                        help="Build GT MIDI from --musicxml-dir using music21 before evaluating")
    parser.add_argument("--onset-tolerance", type=float, default=0.05,
                        help="Onset matching tolerance (seconds) for mir_eval")
    parser.add_argument("--output", default="evaluation/midi_results.json")
    args = parser.parse_args()

    setup_logging("INFO")
    pred_dir, gt_dir = Path(args.pred_dir), Path(args.gt_dir)
    if not pred_dir.exists():
        raise FileNotFoundError(f"Predictions directory not found: {pred_dir}")

    # Optional: build GT first
    if args.build_gt:
        if not args.musicxml_dir:
            raise ValueError("--build-gt requires --musicxml-dir")
        # Only build for files we have predictions for (saves time)
        wanted = {p.stem for p in pred_dir.glob("*.mid")}
        build_groundtruth_from_musicxml(Path(args.musicxml_dir), gt_dir, only_stems=wanted)

    # Pair files by stem
    preds = sorted(pred_dir.glob("*.mid"))
    gts = {p.stem: p for p in gt_dir.glob("*.mid")} if gt_dir.exists() else {}
    pairs = [(p, gts[p.stem]) for p in preds if p.stem in gts]
    if not pairs:
        logger.warning(
            f"No paired files found between {pred_dir} and {gt_dir}. "
            f"({len(preds)} predictions, {len(gts)} ground-truth files)"
        )

    # Choose backend
    use_mir_eval = True
    try:
        import mir_eval  # noqa: F401
        import pretty_midi  # noqa: F401
    except ImportError:
        use_mir_eval = False
        logger.warning(
            "mir_eval / pretty_midi not installed. "
            "Falling back to omr.evaluation.midi_metrics. "
            "For publication-grade numbers run: pip install mir_eval pretty_midi"
        )

    # Per-file evaluation
    per_file = []
    for pred_path, gt_path in pairs:
        if use_mir_eval:
            res = _evaluate_with_mir_eval(pred_path, gt_path, args.onset_tolerance)
        else:
            res = _evaluate_with_repo(pred_path, gt_path)
        res["file"] = pred_path.stem
        per_file.append(res)
        po = res["pitch_onset"]
        line = (f"{pred_path.stem:50s}  "
                f"P+O F1={po['f1']:.3f}  P={po['precision']:.3f}  R={po['recall']:.3f}")
        if res.get("pitch_onset_offset"):
            poo = res["pitch_onset_offset"]
            line += f"   strict F1={poo['f1']:.3f}"
        logger.info(line)

    # Aggregate
    summary = {
        "n_paired_files": len(per_file),
        "backend": "mir_eval" if use_mir_eval else "omr.evaluation.midi_metrics",
        "onset_tolerance_sec": float(args.onset_tolerance),
    }
    if per_file:
        po_f1 = [r["pitch_onset"]["f1"] for r in per_file]
        po_p = [r["pitch_onset"]["precision"] for r in per_file]
        po_r = [r["pitch_onset"]["recall"] for r in per_file]
        summary["pitch_onset"] = {
            "precision_mean": float(mean(po_p)),
            "recall_mean": float(mean(po_r)),
            "f1_mean": float(mean(po_f1)),
        }
        if any(r.get("pitch_onset_offset") for r in per_file):
            poo_f1 = [r["pitch_onset_offset"]["f1"] for r in per_file
                      if r.get("pitch_onset_offset")]
            poo_p = [r["pitch_onset_offset"]["precision"] for r in per_file
                     if r.get("pitch_onset_offset")]
            poo_r = [r["pitch_onset_offset"]["recall"] for r in per_file
                     if r.get("pitch_onset_offset")]
            summary["pitch_onset_offset"] = {
                "precision_mean": float(mean(poo_p)),
                "recall_mean": float(mean(poo_r)),
                "f1_mean": float(mean(poo_f1)),
            }

        logger.info("=" * 60)
        logger.info("AVERAGES")
        logger.info(
            f"  Pitch+Onset      : "
            f"P={summary['pitch_onset']['precision_mean']:.3f}  "
            f"R={summary['pitch_onset']['recall_mean']:.3f}  "
            f"F1={summary['pitch_onset']['f1_mean']:.3f}"
        )
        if "pitch_onset_offset" in summary:
            logger.info(
                f"  Pitch+Onset+Offset: "
                f"P={summary['pitch_onset_offset']['precision_mean']:.3f}  "
                f"R={summary['pitch_onset_offset']['recall_mean']:.3f}  "
                f"F1={summary['pitch_onset_offset']['f1_mean']:.3f}"
            )

    # Persist
    out = {"summary": summary, "per_file": per_file}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    logger.info(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
