#!/usr/bin/env bash
# ============================================================================
# run_full_evaluation.sh — single-command evaluation of the entire pipeline
#
# Produces the numbers needed for the IEEE paper:
#   * Detection mAP (Sec. V-A) — from YOLOv8 .evaluate()
#   * GAT vs GCN edge metrics (Sec. V-B) — via scripts/evaluate_gnn.py
#   * End-to-end MIDI F1 (Sec. V-C) — via scripts/evaluate_midi.py
#   * Inference runtime (Sec. V-D) — wall-clock per page
#
# All artefacts land under evaluation/ and are summarised in
# evaluation/results.json + evaluation/SUMMARY.md.
#
# Usage:
#   bash scripts/run_full_evaluation.sh [DEVICE]
#       DEVICE defaults to 'cuda'. Use 'cpu' on machines without an NVIDIA GPU.
#
# Pre-requisites (run once):
#   bash scripts/run_full_evaluation.sh --help
#   pip install mir_eval pretty_midi music21
#   python scripts/download_data.py
#   python scripts/prepare_data.py
#   python scripts/train_detector.py --phase both --device cuda --batch 8 --fast
#   python scripts/train_gnn.py --model-type gat --device cuda --save-dir checkpoints/gnn
#   python scripts/train_gnn.py --model-type gcn --device cuda --save-dir checkpoints/gnn_gcn
# ============================================================================

set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  sed -n '2,30p' "$0"
  exit 0
fi

DEVICE="${1:-cuda}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EVAL_DIR="evaluation"
PRED_DIR="$EVAL_DIR/predictions"
GT_DIR="$EVAL_DIR/groundtruth"
TEST_IMG_DIR="data/processed/muscima_yolo/images/test"
DET_WEIGHTS="checkpoints/detection/yolov8s_muscima_finetune/weights/best.pt"
GAT_WEIGHTS="checkpoints/gnn/gat_best.pt"
GCN_WEIGHTS="checkpoints/gnn_gcn/gcn_best.pt"
MUSCIMA_YAML="configs/detection/yolov8_muscima.yaml"
MUSICXML_DIR="data/raw/muscima_pp_v2/v2.0/data/musicxml"

mkdir -p "$EVAL_DIR" "$PRED_DIR" "$GT_DIR"

log()  { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[fail]\033[0m %s\n' "$*"; exit 1; }

# ---- Pre-flight checks ------------------------------------------------------
log "Pre-flight checks"
[[ -f "$DET_WEIGHTS"   ]] || fail "Missing detector weights: $DET_WEIGHTS  (run train_detector.py)"
[[ -f "$GAT_WEIGHTS"   ]] || fail "Missing GAT weights: $GAT_WEIGHTS  (run train_gnn.py --model-type gat)"
[[ -d "$TEST_IMG_DIR"  ]] || fail "Missing test images: $TEST_IMG_DIR  (run prepare_data.py)"
[[ -f "$GCN_WEIGHTS"   ]] || warn "GCN weights not found at $GCN_WEIGHTS — ablation row will be skipped"
[[ -d "$MUSICXML_DIR"  ]] || warn "MusicXML dir not found at $MUSICXML_DIR — MIDI ground truth must be supplied manually"

# ---- 1. Symbol detection (YOLO mAP) ----------------------------------------
log "[1/4] Symbol detection — YOLOv8 mAP on writer-disjoint test split"
python - <<PY
import json
from pathlib import Path
from ultralytics import YOLO

model = YOLO("$DET_WEIGHTS")
# split="test" forces evaluation on the test:/ key in the YAML
results = model.val(data="$MUSCIMA_YAML", device="$DEVICE", split="test", verbose=False)
metrics = {
    "mAP50":    float(results.box.map50),
    "mAP50_95": float(results.box.map),
    "precision": float(results.box.mp),
    "recall":    float(results.box.mr),
}
out = Path("$EVAL_DIR/detection_results.json")
out.write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
print("Wrote:", out)
PY

# ---- 2. GAT (and optional GCN) edge classification --------------------------
log "[2/4] Relationship parsing — GAT (vs GCN if available)"
GNN_ARGS=( --weights "$GAT_WEIGHTS" --model-type gat --device "$DEVICE"
           --output "$EVAL_DIR/gnn_results.json" )
if [[ -f "$GCN_WEIGHTS" ]]; then
  GNN_ARGS+=( --weights-baseline "$GCN_WEIGHTS" --baseline-type gcn )
fi
python scripts/evaluate_gnn.py "${GNN_ARGS[@]}"

# ---- 3. End-to-end inference + MIDI evaluation ------------------------------
log "[3/4] End-to-end inference on every test image"
shopt -s nullglob
imgs=( "$TEST_IMG_DIR"/*.png "$TEST_IMG_DIR"/*.jpg )
shopt -u nullglob
if (( ${#imgs[@]} == 0 )); then
  fail "No test images found in $TEST_IMG_DIR"
fi
echo "Processing ${#imgs[@]} test images..."
runtimes_file="$EVAL_DIR/runtimes.csv"
echo "image,seconds" > "$runtimes_file"

for img in "${imgs[@]}"; do
  stem="$(basename "$img")"
  stem="${stem%.*}"
  out_mid="$PRED_DIR/${stem}.mid"
  if [[ -f "$out_mid" ]]; then
    continue
  fi
  start=$(python -c "import time; print(time.time())")
  python scripts/run_inference.py \
      --image "$img" --output "$out_mid" \
      --device "$DEVICE" --confidence 0.3 \
      > /dev/null 2>&1 || warn "Inference failed for $stem"
  end=$(python -c "import time; print(time.time())")
  printf "%s,%.3f\n" "$stem" "$(python -c "print(${end} - ${start})")" >> "$runtimes_file"
done
echo "Predicted MIDI files: $(ls "$PRED_DIR"/*.mid 2>/dev/null | wc -l)"

log "[3b] MIDI ground truth + scoring"
MIDI_ARGS=( --pred-dir "$PRED_DIR" --gt-dir "$GT_DIR" --output "$EVAL_DIR/midi_results.json" )
if [[ -d "$MUSICXML_DIR" ]]; then
  MIDI_ARGS+=( --musicxml-dir "$MUSICXML_DIR" --build-gt )
fi
python scripts/evaluate_midi.py "${MIDI_ARGS[@]}" || warn "MIDI evaluation skipped — see logs"

# ---- 4. Aggregate everything into one JSON + Markdown summary ---------------
log "[4/4] Aggregating results"
python - <<'PY'
import json
from pathlib import Path
from statistics import mean

evald = Path("evaluation")
out = {"detection": None, "gnn": None, "midi": None, "runtime_sec": None}

for key, fname in [("detection", "detection_results.json"),
                   ("gnn",       "gnn_results.json"),
                   ("midi",      "midi_results.json")]:
    p = evald / fname
    if p.exists():
        try:
            out[key] = json.loads(p.read_text())
        except Exception as e:
            out[key] = {"error": f"failed to read {fname}: {e}"}

rt_csv = evald / "runtimes.csv"
if rt_csv.exists():
    times = []
    for line in rt_csv.read_text().strip().splitlines()[1:]:
        try:
            times.append(float(line.split(",")[1]))
        except Exception:
            pass
    if times:
        out["runtime_sec"] = {
            "n_images": len(times),
            "mean": float(mean(times)),
            "min": float(min(times)),
            "max": float(max(times)),
        }

(evald / "results.json").write_text(json.dumps(out, indent=2))

# --- Markdown summary -------------------------------------------------------
md = ["# Ink2MIDI — Full Evaluation Summary", ""]
det = out["detection"] or {}
md += ["## Detection (YOLOv8s, writer-disjoint test split)", ""]
md += [f"- mAP@0.5      : **{det.get('mAP50', float('nan')):.4f}**",
       f"- mAP@0.5:0.95 : **{det.get('mAP50_95', float('nan')):.4f}**", ""]

g = out["gnn"] or {}
if g.get("primary"):
    p = g["primary"]
    md += [f"## Relationship Parsing — {p['model_type'].upper()}", "",
           f"- AUROC               : **{p.get('auroc', float('nan')):.4f}**",
           f"- @0.5  P / R / F1    : "
           f"**{p['at_0.5']['precision']:.4f} / "
           f"{p['at_0.5']['recall']:.4f} / "
           f"{p['at_0.5']['f1']:.4f}**",
           f"- BestF1 (thr={p['best_f1']['threshold']}) : "
           f"**{p['best_f1']['f1']:.4f}**", ""]
if g.get("baseline"):
    b = g["baseline"]
    md += [f"### Baseline — {b['model_type'].upper()}",
           f"- @0.5  P / R / F1    : "
           f"{b['at_0.5']['precision']:.4f} / "
           f"{b['at_0.5']['recall']:.4f} / "
           f"{b['at_0.5']['f1']:.4f}",
           f"- AUROC               : {b.get('auroc', float('nan')):.4f}",
           f"- Delta F1            : {g.get('delta_f1_at_0.5', 0.0):+.4f}", ""]

m = (out["midi"] or {}).get("summary", {})
if m:
    md += ["## End-to-End MIDI", "",
           f"- Backend          : {m.get('backend', '?')}",
           f"- Paired files     : {m.get('n_paired_files', 0)}"]
    if "pitch_onset" in m:
        po = m["pitch_onset"]
        md += [f"- Pitch+Onset       P / R / F1 : "
               f"**{po['precision_mean']:.3f} / "
               f"{po['recall_mean']:.3f} / "
               f"{po['f1_mean']:.3f}**"]
    if "pitch_onset_offset" in m:
        poo = m["pitch_onset_offset"]
        md += [f"- Pitch+Onset+Offset P / R / F1 : "
               f"**{poo['precision_mean']:.3f} / "
               f"{poo['recall_mean']:.3f} / "
               f"{poo['f1_mean']:.3f}**"]
    md += [""]

if out["runtime_sec"]:
    rt = out["runtime_sec"]
    md += ["## Inference Runtime", "",
           f"- Pages timed   : {rt['n_images']}",
           f"- Mean / Min / Max : "
           f"**{rt['mean']:.2f}s / {rt['min']:.2f}s / {rt['max']:.2f}s** per page", ""]

(evald / "SUMMARY.md").write_text("\n".join(md))
print("Wrote: evaluation/results.json")
print("Wrote: evaluation/SUMMARY.md")
PY

log "Done."
echo "Open evaluation/SUMMARY.md for the headline numbers,"
echo "or evaluation/results.json for the full machine-readable dump."
