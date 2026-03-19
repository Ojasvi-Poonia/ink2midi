"""Evaluation visualization tools."""

import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(
    per_class_metrics: dict[str, dict],
    title: str = "Per-Class Detection Performance",
) -> plt.Figure:
    """Plot per-class precision/recall as a grouped bar chart."""
    classes = sorted(per_class_metrics.keys())
    precisions = [per_class_metrics[c]["precision"] for c in classes]
    recalls = [per_class_metrics[c]["recall"] for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2, precisions, width, label="Precision", color="#2196F3")
    ax.bar(x + width / 2, recalls, width, label="Recall", color="#FF9800")

    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    return fig


def plot_training_history(history: dict, title: str = "Training History") -> plt.Figure:
    """Plot training and validation loss/metrics curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    # F1
    if "val_f1" in history:
        axes[1].plot(history["val_f1"], label="Val F1", color="green")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("F1 Score")
        axes[1].set_title("Validation F1")
        axes[1].legend()

    fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_piano_roll_comparison(
    pred_notes: list[dict],
    gt_notes: list[dict],
    title: str = "Piano Roll: Predicted vs Ground Truth",
) -> plt.Figure:
    """Plot piano roll comparison of predicted vs GT MIDI notes."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, sharey=True)

    for ax, notes, label, color in [
        (axes[0], gt_notes, "Ground Truth", "#4CAF50"),
        (axes[1], pred_notes, "Predicted", "#2196F3"),
    ]:
        for n in notes:
            ax.barh(
                n["pitch"],
                n["duration_seconds"],
                left=n["onset_seconds"],
                height=0.8,
                color=color,
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
            )
        ax.set_ylabel("MIDI Pitch")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    axes[1].set_xlabel("Time (seconds)")
    fig.suptitle(title)
    plt.tight_layout()
    return fig
