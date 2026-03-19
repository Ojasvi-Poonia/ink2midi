"""Visualization tools for symbol detection results."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from omr.data.graph_builder import Detection
from omr.data.muscima_parser import CLASS_NAME_TO_ID

# Color palette for 26 classes (BGR for OpenCV)
CLASS_COLORS = {
    "notehead_filled": (0, 0, 255),      # Red
    "notehead_half": (0, 100, 255),       # Orange
    "notehead_whole": (0, 200, 255),      # Yellow-orange
    "stem": (255, 0, 0),                  # Blue
    "beam": (255, 100, 0),               # Light blue
    "flag_eighth_up": (200, 200, 0),     # Cyan
    "flag_eighth_down": (200, 200, 0),
    "flag_sixteenth_up": (200, 150, 0),
    "flag_sixteenth_down": (200, 150, 0),
    "rest_whole": (0, 200, 0),           # Green
    "rest_half": (0, 200, 0),
    "rest_quarter": (0, 200, 0),
    "rest_eighth": (0, 200, 0),
    "rest_sixteenth": (0, 200, 0),
    "sharp": (255, 0, 255),              # Magenta
    "flat": (255, 50, 200),
    "natural": (255, 100, 150),
    "double_sharp": (255, 0, 200),
    "treble_clef": (0, 255, 255),        # Yellow
    "bass_clef": (50, 255, 200),
    "alto_clef": (100, 255, 150),
    "time_sig_digit": (150, 150, 255),   # Light pink
    "bar_line": (128, 128, 128),         # Gray
    "double_bar_line": (100, 100, 100),
    "dot": (0, 255, 0),                  # Bright green
    "tie_slur": (200, 100, 255),         # Purple
}


def draw_detections(
    image: np.ndarray,
    detections: list[Detection],
    show_labels: bool = True,
    show_confidence: bool = True,
    thickness: int = 2,
    font_scale: float = 0.4,
) -> np.ndarray:
    """Draw detection bounding boxes on an image.

    Args:
        image: Input image (grayscale or BGR).
        detections: List of Detection objects.
        show_labels: Whether to show class names.
        show_confidence: Whether to show confidence scores.
        thickness: Box line thickness.
        font_scale: Label font scale.

    Returns:
        Image with drawn detections.
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image = image.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(c) for c in det.bbox]
        color = CLASS_COLORS.get(det.class_name, (255, 255, 255))

        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        if show_labels or show_confidence:
            parts = []
            if show_labels:
                parts.append(det.class_name)
            if show_confidence:
                parts.append(f"{det.confidence:.2f}")
            label = " ".join(parts)

            label_size, _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            cv2.rectangle(
                image,
                (x1, y1 - label_size[1] - 4),
                (x1 + label_size[0], y1),
                color,
                -1,
            )
            cv2.putText(
                image,
                label,
                (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1,
            )

    return image


def save_detection_visualization(
    image_path: str | Path,
    detections: list[Detection],
    output_path: str | Path,
) -> None:
    """Load an image, draw detections, and save."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    result = draw_detections(image, detections)
    cv2.imwrite(str(output_path), result)


def plot_class_distribution(detections: list[Detection], title: str = "Symbol Distribution"):
    """Plot bar chart of detected symbol class counts."""
    counts: dict[str, int] = {}
    for det in detections:
        counts[det.class_name] = counts.get(det.class_name, 0) + 1

    classes = sorted(counts.keys())
    values = [counts[c] for c in classes]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.barh(classes, values)
    ax.set_xlabel("Count")
    ax.set_title(title)
    plt.tight_layout()
    return fig
