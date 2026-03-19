"""Tests for semantic builder (music21 Score assembly)."""

from omr.sequencer.semantic_builder import ResolvedNote, SemanticBuilder, StaffData


class TestSemanticBuilder:
    def test_build_simple_score(self):
        builder = SemanticBuilder()

        staff = StaffData(
            resolved_notes=[
                ResolvedNote(midi_pitch=60, duration_quarters=1.0, x_position=100),
                ResolvedNote(midi_pitch=62, duration_quarters=1.0, x_position=200),
                ResolvedNote(midi_pitch=64, duration_quarters=1.0, x_position=300),
                ResolvedNote(midi_pitch=65, duration_quarters=1.0, x_position=400),
            ],
            clef_type="treble",
            time_sig=(4, 4),
        )

        score = builder.build_score([staff])
        assert len(score.parts) == 1

    def test_chord_grouping(self):
        builder = SemanticBuilder()

        # Two notes at same x-position should form a chord
        staff = StaffData(
            resolved_notes=[
                ResolvedNote(midi_pitch=60, duration_quarters=1.0, x_position=100),
                ResolvedNote(midi_pitch=64, duration_quarters=1.0, x_position=100),  # Same x
                ResolvedNote(midi_pitch=67, duration_quarters=1.0, x_position=200),
            ],
        )

        score = builder.build_score([staff])
        assert len(score.parts) == 1

    def test_rest_handling(self):
        builder = SemanticBuilder()

        staff = StaffData(
            resolved_notes=[
                ResolvedNote(midi_pitch=60, duration_quarters=1.0, x_position=100),
                ResolvedNote(midi_pitch=0, duration_quarters=1.0, x_position=200, is_rest=True),
                ResolvedNote(midi_pitch=62, duration_quarters=1.0, x_position=300),
            ],
        )

        score = builder.build_score([staff])
        assert len(score.parts) == 1

    def test_multiple_staves(self):
        builder = SemanticBuilder()

        treble = StaffData(
            resolved_notes=[
                ResolvedNote(midi_pitch=72, duration_quarters=1.0, x_position=100),
            ],
            clef_type="treble",
        )
        bass = StaffData(
            resolved_notes=[
                ResolvedNote(midi_pitch=48, duration_quarters=1.0, x_position=100),
            ],
            clef_type="bass",
        )

        score = builder.build_score([treble, bass])
        assert len(score.parts) == 2

    def test_barline_segmentation(self):
        builder = SemanticBuilder()

        staff = StaffData(
            resolved_notes=[
                ResolvedNote(midi_pitch=60, duration_quarters=1.0, x_position=100),
                ResolvedNote(midi_pitch=62, duration_quarters=1.0, x_position=200),
                ResolvedNote(midi_pitch=64, duration_quarters=1.0, x_position=400),
            ],
            barline_positions=[300.0],  # Barline between note 2 and 3
        )

        score = builder.build_score([staff])
        # Should have at least 1 measure in the part
        assert len(score.parts) == 1
