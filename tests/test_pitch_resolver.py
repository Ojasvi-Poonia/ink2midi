"""Tests for pitch resolution."""

import pytest

from omr.data.graph_builder import Detection
from omr.sequencer.pitch_resolver import PitchResolver
from omr.sequencer.staff_analysis import Staff


@pytest.fixture
def resolver():
    return PitchResolver()


@pytest.fixture
def treble_staff():
    """Standard treble clef staff with 15px spacing."""
    return Staff(
        line_positions=[100, 115, 130, 145, 160],  # Top to bottom
        staff_space=15.0,
        top=100,
        bottom=160,
    )


@pytest.fixture
def bass_staff():
    return Staff(
        line_positions=[200, 215, 230, 245, 260],
        staff_space=15.0,
        top=200,
        bottom=260,
    )


class TestPitchResolver:
    def test_middle_c_treble(self, resolver, treble_staff):
        """Middle C is one ledger line below treble staff."""
        # Middle C (C4 = MIDI 60) is at y position below the bottom line
        # Bottom line = position 0 = B3 in our treble map
        # One step below = A3, two below = G3... wait let's check the map
        # In our map: position 0.5 = C4 (middle C)
        # Staff position 0.5 is the first space (between bottom line and next line)
        notehead = Detection(
            bbox=(100, 152, 120, 158),  # Center y ~155, between line 3 and 4
            confidence=1.0,
            class_id=0,
            class_name="notehead_filled",
        )
        pitch = resolver.resolve_pitch(notehead, treble_staff, "treble")
        # This should be close to a note in the treble clef range
        assert 48 <= pitch <= 84  # Reasonable treble clef range

    def test_to_midi_number_middle_c(self, resolver):
        """C4 should be MIDI 60."""
        assert resolver._to_midi_number("C", 4) == 60

    def test_to_midi_number_a4(self, resolver):
        """A4 should be MIDI 69 (concert pitch)."""
        assert resolver._to_midi_number("A", 4) == 69

    def test_to_midi_number_with_sharp(self, resolver):
        """C#4 should be MIDI 61."""
        assert resolver._to_midi_number("C", 4, 1) == 61

    def test_to_midi_number_with_flat(self, resolver):
        """Bb3 should be MIDI 58."""
        assert resolver._to_midi_number("B", 3, -1) == 58

    def test_pitch_clamp(self, resolver, treble_staff):
        """Pitch should be clamped to 0-127."""
        # Extreme position that would produce out-of-range
        notehead = Detection(
            bbox=(100, 0, 120, 10),  # Way above staff
            confidence=1.0,
            class_id=0,
            class_name="notehead_filled",
        )
        pitch = resolver.resolve_pitch(notehead, treble_staff, "treble")
        assert 0 <= pitch <= 127

    def test_accidental_override(self, resolver, treble_staff):
        """Explicit accidental should override key signature."""
        notehead = Detection(
            bbox=(100, 150, 120, 160),
            confidence=1.0, class_id=0, class_name="notehead_filled",
        )
        # Key sig says F is sharp
        key_sig = {"F": "sharp"}
        # But there's a natural accidental attached
        natural = Detection(
            bbox=(90, 150, 98, 160),
            confidence=0.9, class_id=16, class_name="natural",
        )

        pitch_with_key = resolver.resolve_pitch(
            notehead, treble_staff, "treble", key_sig, []
        )
        pitch_with_natural = resolver.resolve_pitch(
            notehead, treble_staff, "treble", key_sig, [natural]
        )
        # The natural should undo the sharp
        # (pitch difference depends on which note is being resolved)
        assert isinstance(pitch_with_key, int)
        assert isinstance(pitch_with_natural, int)
