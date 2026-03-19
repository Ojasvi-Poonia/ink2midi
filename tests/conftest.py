"""Shared test fixtures."""

import pytest
import numpy as np

from omr.data.graph_builder import Detection
from omr.data.muscima_parser import DocumentAnnotation, SymbolAnnotation, RelationshipAnnotation


@pytest.fixture
def sample_detections():
    """10 synthetic detections for testing."""
    return [
        Detection(bbox=(100, 200, 120, 220), confidence=0.95, class_id=0, class_name="notehead_filled"),
        Detection(bbox=(108, 220, 115, 280), confidence=0.90, class_id=3, class_name="stem"),
        Detection(bbox=(90, 195, 100, 210), confidence=0.85, class_id=14, class_name="sharp"),
        Detection(bbox=(200, 180, 220, 200), confidence=0.92, class_id=0, class_name="notehead_filled"),
        Detection(bbox=(208, 200, 215, 260), confidence=0.88, class_id=3, class_name="stem"),
        Detection(bbox=(100, 260, 220, 275), confidence=0.80, class_id=4, class_name="beam"),
        Detection(bbox=(50, 100, 80, 300), confidence=0.95, class_id=18, class_name="treble_clef"),
        Detection(bbox=(300, 100, 305, 300), confidence=0.90, class_id=22, class_name="bar_line"),
        Detection(bbox=(125, 198, 130, 205), confidence=0.75, class_id=24, class_name="dot"),
        Detection(bbox=(350, 200, 370, 240), confidence=0.85, class_id=11, class_name="rest_quarter"),
    ]


@pytest.fixture
def sample_document():
    """A synthetic DocumentAnnotation."""
    symbols = [
        SymbolAnnotation(
            symbol_id=0, class_name="notehead_filled", class_id=0,
            bbox=(0.1, 0.2, 0.02, 0.02), bbox_abs=(200, 100, 20, 20),
            image_id="test_page",
        ),
        SymbolAnnotation(
            symbol_id=1, class_name="stem", class_id=3,
            bbox=(0.108, 0.25, 0.007, 0.06), bbox_abs=(220, 108, 7, 60),
            image_id="test_page",
        ),
        SymbolAnnotation(
            symbol_id=2, class_name="sharp", class_id=14,
            bbox=(0.09, 0.195, 0.01, 0.015), bbox_abs=(195, 90, 10, 15),
            image_id="test_page",
        ),
    ]
    relationships = [
        RelationshipAnnotation(
            source_id=0, target_id=1,
            source_class="notehead_filled", target_class="stem",
            relationship_type="outlink",
        ),
        RelationshipAnnotation(
            source_id=2, target_id=0,
            source_class="sharp", target_class="notehead_filled",
            relationship_type="outlink",
        ),
    ]
    return DocumentAnnotation(
        image_id="test_page",
        image_path="",
        image_width=1000,
        image_height=1000,
        symbols=symbols,
        relationships=relationships,
    )


@pytest.fixture
def sample_image():
    """A synthetic grayscale score image with staff lines."""
    img = np.ones((500, 800), dtype=np.uint8) * 255  # White background

    # Draw 5 staff lines
    line_ys = [150, 165, 180, 195, 210]
    for y in line_ys:
        img[y : y + 2, 50:750] = 0  # Black lines

    return img
