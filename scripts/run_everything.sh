#!/usr/bin/env bash
# ============================================================================
# run_everything.sh — true one-command end-to-end pipeline
#
# Wraps:
#   1. Python venv + dependency install
#   2. Dataset download
#   3. Data preparation (writer-disjoint splits + YOLO format)
#   4. Detector training (Phase 1 + Phase 2)
#   5. GAT training
#   6. GCN training (ablation baseline)
#   7. Full evaluation (detection + GNN + end-to-end MIDI + runtime)
#
# Each step is idempotent — it skips if the expected output already exists.
# Re-runs are therefore cheap; only missing steps are executed.
#
# Usage:
#     bash scripts/run_everything.sh                 # cuda (default)
#     bash scripts/run_everything.sh cpu             # cpu fallback
#     bash scripts/run_everything.sh cuda --force    # rerun every step
#     bash scripts/run_everything.sh --help
#
# Recommended: run inside tmux/screen so SSH disconnect doesn't kill it.
#     tmux new -s ink2midi
#     bash scripts/run_everything.sh cuda
#     # detach with Ctrl+B then D
# ============================================================================

set -euo pipefail

# ---------- CLI ----------
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  sed -n '2,30p' "$0"
  exit 0
fi
DEVICE="${1:-cuda}"
FORCE="${2:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p logs
TS="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="logs/run_everything_${TS}.log"

# All output also tee'd into the master log
exec > >(tee -a "$MASTER_LOG") 2>&1

# ---------- helpers ----------
log()  { printf '\n\033[1;36m============== %s ==============\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m[ok]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[fail]\033[0m %s\n' "$*"; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# Skip-or-run helper: skip <step-name> <marker-path> <command...>
# If marker exists and FORCE != --force, skip; otherwise run.
skip_or_run() {
  local name="$1"; shift
  local marker="$1"; shift
  if [[ -e "$marker" && "$FORCE" != "--force" ]]; then
    ok "$name — already done ($marker exists), skipping"
    return 0
  fi
  log "$name"
  "$@"
}

# ---------- 0. Pre-flight ----------
log "[0/7] Pre-flight"
have python3 || fail "python3 not found"
PYV=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python: $PYV"
echo "Device: $DEVICE"
if [[ "$DEVICE" == "cuda" ]]; then
  if have nvidia-smi; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | head -2
  else
    warn "DEVICE=cuda but nvidia-smi not available; will fall back if torch reports no CUDA"
  fi
fi
df -h "$ROOT" | head -2
echo "Master log: $MASTER_LOG"

# ---------- 1. venv + deps ----------
VENV_MARKER=".venv/_ink2midi_ready"
if [[ "$FORCE" == "--force" ]]; then rm -f "$VENV_MARKER"; fi
if [[ ! -e "$VENV_MARKER" ]]; then
  log "[1/7] Creating virtualenv + installing dependencies"
  if [[ ! -d ".venv" ]]; then
    python3 -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install --upgrade pip wheel setuptools
  if [[ "$DEVICE" == "cuda" ]]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  else
    pip install torch torchvision
  fi
  pip install torch-geometric
  pip install -e .
  pip install mir_eval pretty_midi music21 scikit-learn
  touch "$VENV_MARKER"
  ok "Dependencies installed"
else
  ok "[1/7] venv + deps — already done, skipping"
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# Quick GPU sanity check (does not fail the pipeline)
python -c "import torch; print('torch CUDA available:', torch.cuda.is_available(),
'| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')" || true

# ---------- 2. Datasets ----------
skip_or_run "[2/7] Downloading datasets" \
  "data/raw/.download_done" \
  bash -c '
    python scripts/download_data.py
    touch data/raw/.download_done
  '

# ---------- 3. Data preparation ----------
skip_or_run "[3/7] Preparing data (splits + YOLO format)" \
  "data/processed/muscima_yolo/images/test" \
  python scripts/prepare_data.py

# ---------- 4. Detector training (Phase 1 + Phase 2) ----------
DET_BEST="checkpoints/detection/yolov8s_muscima_finetune/weights/best.pt"
skip_or_run "[4/7] Training YOLOv8 detector (both phases)" \
  "$DET_BEST" \
  python scripts/train_detector.py \
    --phase both --device "$DEVICE" --batch 8 --fast

# ---------- 5. GAT training ----------
GAT_BEST="checkpoints/gnn/gat_best.pt"
skip_or_run "[5/7] Training GAT relationship parser" \
  "$GAT_BEST" \
  python scripts/train_gnn.py \
    --model-type gat --device "$DEVICE" --fast \
    --save-dir checkpoints/gnn

# ---------- 6. GCN training (ablation baseline) ----------
GCN_BEST="checkpoints/gnn_gcn/gcn_best.pt"
skip_or_run "[6/7] Training GCN baseline (ablation)" \
  "$GCN_BEST" \
  python scripts/train_gnn.py \
    --model-type gcn --device "$DEVICE" --fast \
    --save-dir checkpoints/gnn_gcn

# ---------- 7. Full evaluation ----------
log "[7/7] Running full evaluation"
bash scripts/run_full_evaluation.sh "$DEVICE"

# ---------- Done ----------
log "ALL DONE"
echo "Headline numbers:"
[[ -f evaluation/SUMMARY.md ]] && cat evaluation/SUMMARY.md
echo
echo "Full machine-readable dump : evaluation/results.json"
echo "Master log file            : $MASTER_LOG"
echo
echo "To pull results to your laptop:"
echo "  rsync -avz --progress USER@HOST:$ROOT/evaluation ~/ink2midi-results/"
