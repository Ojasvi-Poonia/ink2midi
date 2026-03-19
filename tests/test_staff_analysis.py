"""Tests for staff line detection and symbol assignment."""

import numpy as np
import pytest

from omr.data.graph_builder import Detection
from omr.sequencer.staff_analysis import StaffAnalyzer


@pytest.fixture
def analyzer():
    return StaffAnalyzer()


class TestStaffAnalyzer:
    def test_detect_staff_lines(self, analyzer, sample_image):
        staves = analyzer.detect_staff_lines(sample_image)
        assert len(staves) >= 1
        # First staff should have 5 lines
        assert len(staves[0].line_positions) == 5

    def test_staff_space(self, analyzer, sample_image):
        staves = analyzer.detect_staff_lines(sample_image)
        if staves:
            # Staff space should be approximately 15 (our fixture uses 15px spacing)
            assert abs(staves[0].staff_space - 15) < 5

    def test_assign_symbols(self, analyzer):
        from omr.sequencer.staff_analysis import Staff

        staves = [
            Staff(line_positions=[100, 115, 130, 145, 160], staff_space=15, top=100, bottom=160),
            Staff(line_positions=[250, 265, 280, 295, 310], staff_space=15, top=250, bottom=310),
        ]

        detections = [
            # Should be assigned to staff 0 (y=130, within staff 0)
            Detection(bbox=(50, 125, 70, 135), confidence=0.9, class_id=0, class_name="notehead_filled"),
            # Should be assigned to staff 1 (y=280, within staff 1)
            Detection(bbox=(50, 275, 70, 285), confidence=0.9, class_id=0, class_name="notehead_filled"),
        ]

        assignments = analyzer.assign_symbols_to_staves(detections, staves)
        assert len(assignments[0]) == 1
        assert len(assignments[1]) == 1

    def test_no_lines_detected(self, analyzer):
        blank = np.ones((200, 200), dtype=np.uint8) * 255
        staves = analyzer.detect_staff_lines(blank)
        assert len(staves) == 0
