"""Tests for rhythm resolution."""

import pytest

from omr.data.graph_builder import Detection
from omr.sequencer.rhythm_resolver import RhythmResolver


@pytest.fixture
def resolver():
    return RhythmResolver()


def _make_notehead(class_name="notehead_filled"):
    return Detection(
        bbox=(100, 200, 120, 220),
        confidence=1.0,
        class_id=0,
        class_name=class_name,
    )


def _make_detection(class_name, class_id=0):
    return Detection(
        bbox=(100, 200, 120, 250),
        confidence=1.0,
        class_id=class_id,
        class_name=class_name,
    )


class TestRhythmResolver:
    def test_whole_note(self, resolver):
        """Whole note = 4 beats."""
        notehead = _make_notehead("notehead_whole")
        duration = resolver.resolve_note_duration(notehead, {})
        assert duration == 4.0

    def test_half_note(self, resolver):
        """Half note = 2 beats."""
        notehead = _make_notehead("notehead_half")
        duration = resolver.resolve_note_duration(notehead, {"stem": [_make_detection("stem")]})
        assert duration == 2.0

    def test_quarter_note(self, resolver):
        """Filled notehead + stem (no beams) = quarter note = 1 beat."""
        notehead = _make_notehead("notehead_filled")
        connected = {"stem": [_make_detection("stem")]}
        duration = resolver.resolve_note_duration(notehead, connected)
        assert duration == 1.0

    def test_eighth_note_with_beam(self, resolver):
        """Filled notehead + 1 beam = eighth note = 0.5 beats."""
        notehead = _make_notehead("notehead_filled")
        connected = {
            "stem": [_make_detection("stem")],
            "beam": [_make_detection("beam")],
        }
        duration = resolver.resolve_note_duration(notehead, connected)
        assert duration == 0.5

    def test_eighth_note_with_flag(self, resolver):
        """Filled notehead + flag = eighth note."""
        notehead = _make_notehead("notehead_filled")
        connected = {
            "stem": [_make_detection("stem")],
            "flag_eighth_up": [_make_detection("flag_eighth_up")],
        }
        duration = resolver.resolve_note_duration(notehead, connected)
        assert duration == 0.5

    def test_sixteenth_note(self, resolver):
        """Filled notehead + 2 beams = sixteenth = 0.25 beats."""
        notehead = _make_notehead("notehead_filled")
        connected = {
            "stem": [_make_detection("stem")],
            "beam": [_make_detection("beam"), _make_detection("beam")],
        }
        duration = resolver.resolve_note_duration(notehead, connected)
        assert duration == 0.25

    def test_dotted_quarter(self, resolver):
        """Quarter note + dot = 1.5 beats."""
        notehead = _make_notehead("notehead_filled")
        connected = {
            "stem": [_make_detection("stem")],
            "dot": [_make_detection("dot")],
        }
        duration = resolver.resolve_note_duration(notehead, connected)
        assert duration == 1.5

    def test_dotted_half(self, resolver):
        """Half note + dot = 3 beats."""
        notehead = _make_notehead("notehead_half")
        connected = {"dot": [_make_detection("dot")]}
        duration = resolver.resolve_note_duration(notehead, connected)
        assert duration == 3.0

    def test_double_dotted_quarter(self, resolver):
        """Quarter + 2 dots = 1.75 beats."""
        notehead = _make_notehead("notehead_filled")
        connected = {
            "stem": [_make_detection("stem")],
            "dot": [_make_detection("dot"), _make_detection("dot")],
        }
        duration = resolver.resolve_note_duration(notehead, connected)
        assert duration == 1.75

    def test_rest_durations(self, resolver):
        """Test all rest types."""
        assert resolver.resolve_rest_duration(_make_detection("rest_whole")) == 4.0
        assert resolver.resolve_rest_duration(_make_detection("rest_half")) == 2.0
        assert resolver.resolve_rest_duration(_make_detection("rest_quarter")) == 1.0
        assert resolver.resolve_rest_duration(_make_detection("rest_eighth")) == 0.5
        assert resolver.resolve_rest_duration(_make_detection("rest_sixteenth")) == 0.25

    def test_measure_validation_valid(self, resolver):
        """4 quarter notes in 4/4 should be valid."""
        result = resolver.validate_measure_duration([1.0, 1.0, 1.0, 1.0], (4, 4))
        assert result["valid"] is True

    def test_measure_validation_invalid(self, resolver):
        """5 quarter notes in 4/4 should be invalid."""
        result = resolver.validate_measure_duration([1.0, 1.0, 1.0, 1.0, 1.0], (4, 4))
        assert result["valid"] is False

    def test_measure_validation_3_4(self, resolver):
        """3 quarter notes in 3/4 should be valid."""
        result = resolver.validate_measure_duration([1.0, 1.0, 1.0], (3, 4))
        assert result["valid"] is True
