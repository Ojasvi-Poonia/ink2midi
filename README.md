<div align="center">

# OMR Handwritten

### Deep Learning-Based Optical Music Recognition: From Handwritten Sheet Music to Playable MIDI

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-0080ff.svg)](https://docs.ultralytics.com/)
[![PyG](https://img.shields.io/badge/PyG-Graph%20Neural%20Networks-3C2179.svg)](https://pyg.org/)

An end-to-end pipeline that converts images of **handwritten music scores** into **playable MIDI files** using YOLOv8 object detection, Graph Attention Networks, and rule-based music theory.

<br>

```
Sheet Music Image  -->  Symbol Detection  -->  Relationship Parsing  -->  MIDI File
    (input.png)          (YOLOv8)                (GAT)                  (output.mid)
```

</div>

---

## Highlights

- **End-to-end pipeline** -- single command to go from a scanned score image to a playable MIDI file
- **Two-phase transfer learning** -- pretrain on DeepScores V2 (digital), finetune on MUSCIMA++ (handwritten)
- **Graph Attention Network** for structural relationship parsing between detected music symbols
- **26 music symbol classes** covering noteheads, stems, beams, flags, rests, accidentals, clefs, time signatures, barlines, dots, and ties/slurs
- **Writer-disjoint evaluation** -- train/val/test splits ensure generalization to unseen handwriting styles
- **GPU-accelerated** with FP16 mixed precision and cuDNN benchmarking on NVIDIA GPUs

---

## Architecture

```
                          OMR Pipeline
 ============================================================

  Input Image
       |
       v
  +------------------+
  | Symbol Detection  |    YOLOv8s (11.2M params)
  | (26 classes)      |    Pretrained: DeepScores V2
  +------------------+    Finetuned:  MUSCIMA++
       |
       v
  +------------------+
  | Graph Construction|    k-NN spatial graph (k=8)
  | (k-NN)           |    Node features: class + geometry
  +------------------+    Edge features: dx, dy, dist, angle
       |
       v
  +------------------+
  | Relationship      |    3-layer GAT, 4 attention heads
  | Parsing (GAT)     |    Binary edge classification
  +------------------+    Union-find grouping
       |
       v
  +------------------+
  | Staff Analysis    |    Hough transform line detection
  | & Pitch Resolver  |    Clef-aware pitch mapping
  +------------------+
       |
       v
  +------------------+
  | Rhythm Resolver   |    Notehead type + beams + flags + dots
  | & MIDI Export     |    music21 Score --> MIDI file
  +------------------+
       |
       v
   output.mid
```

---

## Results

Trained and evaluated on an **NVIDIA RTX 3070 Ti** (8 GB VRAM, CUDA 12.2).

### Symbol Detection (YOLOv8s)

| Metric | Score |
|--------|-------|
| **mAP@0.5** | **0.630** |
| **mAP@0.5:0.95** | **0.371** |
| Precision | 0.705 |
| Recall | 0.518 |

<details>
<summary><b>Per-class detection results</b> (click to expand)</summary>

| Class | Instances | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|-----------|-----------|--------|---------|---------------|
| notehead_filled | 2982 | 0.847 | 0.698 | 0.802 | 0.355 |
| notehead_half | 210 | 0.721 | 0.629 | 0.703 | 0.320 |
| notehead_whole | 24 | 0.650 | 0.542 | 0.688 | 0.443 |
| stem | 2929 | 0.812 | 0.167 | 0.488 | 0.195 |
| beam | 919 | 0.828 | 0.560 | 0.723 | 0.436 |
| flag_eighth_up | 152 | 0.839 | 0.513 | 0.683 | 0.258 |
| flag_eighth_down | 167 | 0.872 | 0.695 | 0.801 | 0.402 |
| rest_quarter | 115 | 0.805 | 0.896 | 0.924 | 0.590 |
| rest_eighth | 163 | 0.937 | 0.724 | 0.850 | 0.537 |
| sharp | 295 | 0.890 | 0.661 | 0.785 | 0.428 |
| flat | 161 | 0.821 | 0.627 | 0.749 | 0.381 |
| treble_clef | 59 | 0.967 | 0.983 | 0.991 | 0.769 |
| bass_clef | 38 | 0.972 | 0.921 | 0.958 | 0.735 |
| alto_clef | 27 | 0.960 | 0.889 | 0.942 | 0.689 |
| tie_slur | 431 | 0.889 | 0.777 | 0.859 | 0.553 |
| bar_line | 445 | 0.793 | 0.535 | 0.654 | 0.317 |

</details>

### Training Time

| Phase | Duration |
|-------|----------|
| Phase 1: YOLOv8 pretrain (100 epochs) | ~1-2 hrs |
| Phase 2: YOLOv8 finetune (150 epochs) | ~15-30 min |
| Phase 3: GAT training (200 epochs max) | ~5-10 min |
| **Total** | **~1.5-2.5 hrs** |

---

## Quick Start

### Prerequisites

- Python 3.11 - 3.13
- NVIDIA GPU with CUDA support (recommended: 8 GB+ VRAM)

### Installation

```bash
git clone https://github.com/Ojasvi-Poonia/ink2midi.git
cd ink2midi

python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch Geometric
pip install torch-geometric

# Install the project
pip install -e .
```

### Training

```bash
# 1. Download datasets (MUSCIMA++, CVC-MUSCIMA, DeepScores V2)
python scripts/download_data.py

# 2. Prepare data (parse annotations, create YOLO format, writer-disjoint splits)
python scripts/prepare_data.py

# 3. Train symbol detector (two-phase transfer learning)
python scripts/train_detector.py --phase both --device cuda --batch 8 --fast

# 4. Train GNN relationship parser
python scripts/train_gnn.py --device cuda
```

### Inference

```bash
python scripts/run_inference.py \
  --image path/to/score.jpg \
  --output output.mid \
  --device cuda \
  --visualize
```

Both `.jpg` and `.png` input images are supported.

---

## Datasets

| Dataset | Images | Symbols | Type | Purpose |
|---------|--------|---------|------|---------|
| [MUSCIMA++ v2](https://github.com/OMR-Research/muscima-pp) | 140 | 91,254 | Handwritten | Finetune detector + train GNN |
| [CVC-MUSCIMA](https://datasets.cvc.uab.cat/muscima/) | 140 | -- | Handwritten | Source images for MUSCIMA++ |
| [DeepScores V2](https://zenodo.org/records/4012193) | 1,714 | 135 classes | Digital | Pretrain detector |

---

## Symbol Classes

The detector recognizes **26 classes** of music notation primitives:

| ID | Class | ID | Class | ID | Class |
|----|-------|----|---------|----|-------|
| 0 | `notehead_filled` | 9 | `rest_whole` | 18 | `treble_clef` |
| 1 | `notehead_half` | 10 | `rest_half` | 19 | `bass_clef` |
| 2 | `notehead_whole` | 11 | `rest_quarter` | 20 | `alto_clef` |
| 3 | `stem` | 12 | `rest_eighth` | 21 | `time_sig_digit` |
| 4 | `beam` | 13 | `rest_sixteenth` | 22 | `bar_line` |
| 5 | `flag_eighth_up` | 14 | `sharp` | 23 | `double_bar_line` |
| 6 | `flag_eighth_down` | 15 | `flat` | 24 | `dot` |
| 7 | `flag_sixteenth_up` | 16 | `natural` | 25 | `tie_slur` |
| 8 | `flag_sixteenth_down` | 17 | `double_sharp` | | |

---

## Project Structure

```
omr-handwritten/
├── configs/
│   ├── base.yaml                          # Global settings
│   ├── detection/
│   │   ├── yolov8_deepscores.yaml         # DeepScores pretrain config
│   │   └── yolov8_muscima.yaml            # MUSCIMA++ finetune config
│   ├── gnn/
│   │   └── gat_relationship.yaml          # GAT training config
│   └── pipeline/
│       └── inference.yaml                 # Inference pipeline config
│
├── scripts/
│   ├── download_data.py                   # Dataset download automation
│   ├── prepare_data.py                    # Data parsing & YOLO format conversion
│   ├── train_detector.py                  # YOLOv8 two-phase training
│   ├── train_gnn.py                       # GAT/GCN training
│   ├── run_inference.py                   # End-to-end inference
│   └── evaluate.py                        # Metrics computation
│
├── src/omr/
│   ├── data/
│   │   ├── muscima_parser.py              # MUSCIMA++ v2 XML annotation parser
│   │   ├── deepscores_parser.py           # DeepScores V2 JSON parser
│   │   ├── graph_builder.py               # k-NN notation graph construction
│   │   ├── yolo_converter.py              # YOLO format converter
│   │   ├── coco_converter.py              # COCO format converter
│   │   ├── augmentations.py               # Music-aware data augmentations
│   │   ├── datasets.py                    # PyTorch Dataset classes
│   │   └── download.py                    # Dataset download utilities
│   │
│   ├── detection/
│   │   ├── yolo_trainer.py                # YOLOv8 training & evaluation wrapper
│   │   ├── postprocess.py                 # Music-aware NMS & geometric validation
│   │   └── visualize.py                   # Detection overlay visualization
│   │
│   ├── relationship/
│   │   ├── gat_model.py                   # Graph Attention Network
│   │   ├── gcn_model.py                   # GCN ablation baseline
│   │   ├── graph_dataset.py               # PyG InMemoryDataset
│   │   ├── trainer.py                     # GNN training loop with early stopping
│   │   └── inference.py                   # Edge prediction + union-find grouping
│   │
│   ├── sequencer/
│   │   ├── staff_analysis.py              # Staff line detection (Hough transform)
│   │   ├── pitch_resolver.py              # Staff position -> MIDI pitch mapping
│   │   ├── rhythm_resolver.py             # Notehead + modifiers -> duration
│   │   ├── semantic_builder.py            # music21 Score assembly
│   │   └── midi_writer.py                 # MIDI file export
│   │
│   ├── pipeline/
│   │   └── inference.py                   # End-to-end OMR orchestration
│   │
│   ├── evaluation/
│   │   ├── detection_metrics.py           # Detection mAP computation
│   │   ├── graph_metrics.py               # Edge classification metrics
│   │   ├── midi_metrics.py                # MIDI-level evaluation
│   │   └── visualization.py               # Evaluation plots & confusion matrices
│   │
│   └── utils/
│       ├── config.py                      # OmegaConf configuration management
│       ├── device.py                      # Auto device selection (CUDA/MPS/CPU)
│       ├── logging.py                     # Structured logging
│       └── reproducibility.py             # Seed & determinism utilities
│
├── tests/                                 # Unit tests
└── pyproject.toml                         # Project metadata & dependencies
```

---

## Training Details

### Model Specifications

| Component | Architecture | Parameters | Details |
|-----------|-------------|------------|---------|
| **Detector** | YOLOv8s | 11.2M | 73 layers, 28.5 GFLOPs |
| **GNN** | GAT (3-layer) | ~200K | 4 attention heads, 128-d hidden |
| **Node features** | 34-dim | -- | 26-d class one-hot + 8 geometric |
| **Edge features** | 5-dim | -- | dx, dy, distance, angle, scale |

### Training Configuration

<details>
<summary><b>Phase 1: Pretrain on DeepScores V2</b></summary>

| Parameter | Value |
|-----------|-------|
| Base model | YOLOv8s (COCO pretrained) |
| Dataset | DeepScores V2 (1,714 pages) |
| Image size | 640 x 640 |
| Batch size | 16 |
| Epochs | 100 |
| Optimizer | SGD (momentum=0.937, weight_decay=5e-4) |
| Early stopping | Patience 30 epochs |
| Mosaic | 0.5 (reduced for music notation) |
| Rotation | 3.0 deg max |
| Horizontal/Vertical flip | Disabled (invalidates music semantics) |

</details>

<details>
<summary><b>Phase 2: Finetune on MUSCIMA++</b></summary>

| Parameter | Value |
|-----------|-------|
| Initial weights | Phase 1 best checkpoint |
| Dataset | MUSCIMA++ v2 (140 pages) |
| Splits | Writer-disjoint (train: writers 1-35, val: 36-42, test: 43-50) |
| Batch size | 8 |
| Epochs | 150 |
| Initial LR | 1e-3 (lower for domain adaptation) |

</details>

<details>
<summary><b>Phase 3: GAT Relationship Parser</b></summary>

| Parameter | Value |
|-----------|-------|
| Architecture | 3-layer GAT, 4 attention heads |
| Hidden dimension | 128 |
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| LR Scheduler | Cosine warm restarts (T_0=20) |
| Batch size | 64 |
| Epochs | 200 (early stopping, patience 30) |
| Edge classification threshold | 0.5 |
| k-NN connectivity | k=8, max radius 200px |

</details>

### Inference Defaults

| Parameter | Value |
|-----------|-------|
| Detection confidence | 0.3 |
| NMS IoU threshold | 0.45 |
| Edge threshold | 0.5 |
| Default tempo | 120 BPM |

---

## Performance Optimizations

| Optimization | Effect |
|-------------|--------|
| FP16 Mixed Precision (AMP) | ~1.5-2x speedup via Tensor Cores |
| cuDNN Benchmark mode | ~10-20% faster convolutions |
| Multi-worker DataLoaders | Eliminates CPU-GPU I/O bottleneck |

Enable all optimizations with the `--fast` flag:

```bash
python scripts/train_detector.py --device cuda --batch 16 --fast
```

---

## Testing

```bash
pytest tests/ -v --tb=short

# With coverage report
pytest --cov=omr tests/
```

---

## Checkpoints

Models are saved automatically during training:

| Model | Path |
|-------|------|
| YOLOv8 pretrain | `checkpoints/detection/yolov8s_deepscores_pretrain/weights/best.pt` |
| YOLOv8 finetune | `checkpoints/detection/yolov8s_muscima_finetune/weights/best.pt` |
| GAT | `checkpoints/gnn/gat_best.pt` |

> Checkpoints are excluded from version control. Train the models or obtain weights separately.

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Object Detection** | [YOLOv8](https://docs.ultralytics.com/) (Ultralytics) |
| **Graph Neural Networks** | [PyTorch Geometric](https://pyg.org/), GAT |
| **Deep Learning** | [PyTorch](https://pytorch.org/) 2.2+ |
| **Music Representation** | [music21](https://web.mit.edu/music21/), mido, pretty_midi |
| **Image Processing** | OpenCV, Pillow, scikit-image |
| **Experiment Tracking** | Weights & Biases, TensorBoard |
| **Configuration** | OmegaConf, PyYAML |

---

## References

- Hajic, J., & Pecina, P. (2017). [The MUSCIMA++ Dataset for Handwritten Optical Music Recognition](https://github.com/OMR-Research/muscima-pp). ICDAR 2017.
- Tuggener, L., et al. (2018). [DeepScores -- A Dataset for Segmentation, Detection and Classification of Tiny Objects](https://zenodo.org/records/4012193). ICPR 2018.
- Velickovic, P., et al. (2018). [Graph Attention Networks](https://arxiv.org/abs/1710.10903). ICLR 2018.
- Jocher, G., et al. (2023). [Ultralytics YOLOv8](https://docs.ultralytics.com/). Ultralytics.
- Cuthbert, M. S., & Ariza, C. (2010). [music21: A Toolkit for Computer-Aided Musicology](https://web.mit.edu/music21/). MIT.

---

## License

This project is licensed under the MIT License.

---

<div align="center">

**Built with PyTorch, YOLOv8, PyTorch Geometric, and music21**

</div>
