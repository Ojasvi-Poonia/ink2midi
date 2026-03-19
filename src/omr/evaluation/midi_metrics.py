"""MIDI playback fidelity evaluation metrics."""

from pathlib import Path

from omr.sequencer.midi_writer import MIDIWriter
from omr.utils.logging import get_logger

logger = get_logger("evaluation.midi_metrics")


def compute_midi_metrics(
    predicted_midi: str | Path,
    ground_truth_midi: str | Path,
    onset_tolerance_sec: float = 0.05,
    offset_tolerance_ratio: float = 0.2,
) -> dict:
    """Compare two MIDI files note-by-note.

    Metrics:
    - Note precision: fraction of predicted notes matching a GT note
    - Note recall: fraction of GT notes matched by a prediction
    - Note F1: harmonic mean
    - Pitch accuracy: Of correctly-onset notes, how many have correct pitch
    - Duration accuracy: Of matched notes, duration correctness

    Matching criteria:
    - Pitch: exact MIDI number match
    - Onset: within onset_tolerance_sec
    - Offset: within offset_tolerance_ratio * note_duration or onset_tolerance_sec

    Args:
        predicted_midi: Path to predicted MIDI file.
        ground_truth_midi: Path to ground truth MIDI file.
        onset_tolerance_sec: Maximum onset time difference (seconds).
        offset_tolerance_ratio: Maximum offset tolerance as fraction of duration.

    Returns:
        Dict with evaluation metrics.
    """
    writer = MIDIWriter()
    pred_notes = writer.midi_to_note_list(predicted_midi)
    gt_notes = writer.midi_to_note_list(ground_truth_midi)

    if not gt_notes:
        return {
            "note_precision": 0.0,
            "note_recall": 0.0,
            "note_f1": 0.0,
            "pitch_accuracy": 0.0,
            "num_predicted": len(pred_notes),
            "num_gt": 0,
        }

    # Match predicted notes to GT notes
    matched_gt = set()
    matches = []

    for pred in pred_notes:
        best_match = None
        best_onset_diff = float("inf")

        for j, gt in enumerate(gt_notes):
            if j in matched_gt:
                continue

            # Check pitch match
            if pred["pitch"] != gt["pitch"]:
                continue

            # Check onset tolerance
            onset_diff = abs(pred["onset_seconds"] - gt["onset_seconds"])
            if onset_diff > onset_tolerance_sec:
                continue

            # Check offset tolerance
            offset_tol = max(
                onset_tolerance_sec,
                offset_tolerance_ratio * gt["duration_seconds"],
            )
            pred_offset = pred["onset_seconds"] + pred["duration_seconds"]
            gt_offset = gt["onset_seconds"] + gt["duration_seconds"]
            if abs(pred_offset - gt_offset) > offset_tol:
                continue

            if onset_diff < best_onset_diff:
                best_onset_diff = onset_diff
                best_match = j

        if best_match is not None:
            matched_gt.add(best_match)
            matches.append((pred, gt_notes[best_match]))

    tp = len(matches)
    fp = len(pred_notes) - tp
    fn = len(gt_notes) - tp

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    # Pitch accuracy (for onset-matched notes regardless of pitch)
    onset_matches = 0
    pitch_correct = 0
    for pred in pred_notes:
        for j, gt in enumerate(gt_notes):
            onset_diff = abs(pred["onset_seconds"] - gt["onset_seconds"])
            if onset_diff <= onset_tolerance_sec:
                onset_matches += 1
                if pred["pitch"] == gt["pitch"]:
                    pitch_correct += 1
                break

    pitch_accuracy = pitch_correct / max(onset_matches, 1)

    # Duration accuracy for matched notes
    duration_errors = []
    for pred, gt in matches:
        if gt["duration_seconds"] > 0:
            error = abs(pred["duration_seconds"] - gt["duration_seconds"]) / gt[
                "duration_seconds"
            ]
            duration_errors.append(error)

    avg_duration_error = sum(duration_errors) / max(len(duration_errors), 1)

    return {
        "note_precision": precision,
        "note_recall": recall,
        "note_f1": f1,
        "pitch_accuracy": pitch_accuracy,
        "avg_duration_error": avg_duration_error,
        "num_predicted": len(pred_notes),
        "num_gt": len(gt_notes),
        "num_matched": tp,
    }
