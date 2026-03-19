"""Tests for detection post-processing."""

from omr.data.graph_builder import Detection
from omr.detection.postprocess import postprocess_detections, _compute_iou


class TestPostprocess:
    def test_confidence_filter(self):
        detections = [
            Detection(bbox=(0, 0, 10, 10), confidence=0.9, class_id=0, class_name="notehead_filled"),
            Detection(bbox=(20, 20, 30, 30), confidence=0.1, class_id=0, class_name="notehead_filled"),
        ]
        result = postprocess_detections(detections, score_threshold=0.3)
        assert len(result) == 1

    def test_min_area_filter(self):
        detections = [
            Detection(bbox=(0, 0, 2, 2), confidence=0.9, class_id=0, class_name="notehead_filled"),
            Detection(bbox=(0, 0, 100, 100), confidence=0.9, class_id=0, class_name="notehead_filled"),
        ]
        result = postprocess_detections(detections, min_area=16)
        assert len(result) == 1

    def test_stem_validation(self):
        """Stems must be elongated (height >> width)."""
        detections = [
            # Valid stem: tall and narrow
            Detection(bbox=(0, 0, 5, 50), confidence=0.9, class_id=3, class_name="stem"),
            # Invalid stem: square
            Detection(bbox=(0, 0, 20, 20), confidence=0.9, class_id=3, class_name="stem"),
        ]
        result = postprocess_detections(detections, validate_stems=True, stem_min_aspect=2.0)
        assert len(result) == 1

    def test_merge_duplicate_noteheads(self):
        detections = [
            Detection(bbox=(100, 100, 120, 120), confidence=0.95, class_id=0, class_name="notehead_filled"),
            Detection(bbox=(102, 102, 118, 118), confidence=0.85, class_id=0, class_name="notehead_filled"),
        ]
        result = postprocess_detections(detections, merge_noteheads_iou=0.5)
        assert len(result) == 1
        assert result[0].confidence == 0.95  # Keep highest confidence

    def test_compute_iou_identical(self):
        assert _compute_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0

    def test_compute_iou_no_overlap(self):
        assert _compute_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_compute_iou_partial(self):
        iou = _compute_iou((0, 0, 10, 10), (5, 5, 15, 15))
        assert 0.1 < iou < 0.5
