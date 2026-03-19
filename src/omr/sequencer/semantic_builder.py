"""Assemble resolved pitches and durations into a music21 Score."""

from dataclasses import dataclass, field

from music21 import chord, clef, duration, key, meter, note, stream, tempo

from omr.utils.logging import get_logger

logger = get_logger("sequencer.semantic_builder")


@dataclass
class ResolvedNote:
    """A note with resolved pitch and duration."""

    midi_pitch: int
    duration_quarters: float
    x_position: float  # Horizontal position for ordering
    is_rest: bool = False


@dataclass
class StaffData:
    """All resolved data for a single staff."""

    resolved_notes: list[ResolvedNote] = field(default_factory=list)
    clef_type: str = "treble"
    key_sig_fifths: int = 0  # Positive = sharps, negative = flats
    time_sig: tuple[int, int] = (4, 4)
    barline_positions: list[float] = field(default_factory=list)


class SemanticBuilder:
    """Build music21 Score from resolved staff data."""

    def build_score(
        self,
        staves: list[StaffData],
        tempo_bpm: int = 120,
    ) -> stream.Score:
        """Build complete music21 Score from analyzed staff data.

        Args:
            staves: List of StaffData objects.
            tempo_bpm: Tempo in BPM.

        Returns:
            music21 Score ready for MIDI export.
        """
        score = stream.Score()

        # Add tempo
        score.insert(0, tempo.MetronomeMark(number=tempo_bpm))

        for staff_idx, staff_data in enumerate(staves):
            part = self._build_part(staff_data, staff_idx)
            score.insert(0, part)

        logger.info(
            f"Built score with {len(staves)} parts, "
            f"{sum(len(s.resolved_notes) for s in staves)} total notes"
        )
        return score

    def _build_part(self, staff_data: StaffData, part_idx: int) -> stream.Part:
        """Build a single Part from staff data."""
        part = stream.Part()
        part.id = f"Part{part_idx}"

        # Sort notes by x-position (left-to-right reading order)
        sorted_notes = sorted(staff_data.resolved_notes, key=lambda n: n.x_position)

        # Segment into measures using barline positions
        if sorted_notes:
            measures = self._segment_into_measures(sorted_notes, staff_data.barline_positions)
        else:
            measures = [[]]  # One empty measure

        for m_idx, measure_notes in enumerate(measures):
            m = stream.Measure(number=m_idx + 1)

            # Put clef, key sig, time sig in the first measure
            if m_idx == 0:
                m.append(self._create_clef(staff_data.clef_type))
                m.append(key.KeySignature(staff_data.key_sig_fifths))
                num, denom = staff_data.time_sig
                m.append(meter.TimeSignature(f"{num}/{denom}"))

            if not measure_notes:
                # Empty measure: insert a whole rest
                r = note.Rest()
                r.duration = duration.Duration(4.0)
                m.append(r)
                part.append(m)
                continue

            # Group simultaneous notes into chords
            groups = self._group_simultaneous(measure_notes)

            for group in groups:
                if len(group) == 1:
                    n = group[0]
                    if n.is_rest:
                        r = note.Rest()
                        r.duration = duration.Duration(n.duration_quarters)
                        m.append(r)
                    else:
                        music_note = note.Note(n.midi_pitch)
                        music_note.duration = duration.Duration(n.duration_quarters)
                        m.append(music_note)
                else:
                    # Multiple simultaneous notes → chord
                    pitches = [n.midi_pitch for n in group if not n.is_rest]
                    if pitches:
                        c = chord.Chord(pitches)
                        c.duration = duration.Duration(group[0].duration_quarters)
                        m.append(c)

            part.append(m)

        return part

    def _create_clef(self, clef_type: str):
        """Create a music21 Clef object."""
        if clef_type == "bass":
            return clef.BassClef()
        elif clef_type == "alto":
            return clef.AltoClef()
        return clef.TrebleClef()

    def _segment_into_measures(
        self,
        notes: list[ResolvedNote],
        barline_positions: list[float],
    ) -> list[list[ResolvedNote]]:
        """Split notes into measures using barline x-positions."""
        if not barline_positions:
            # No barlines detected → everything in one measure
            return [notes]

        barlines = sorted(barline_positions)
        measures: list[list[ResolvedNote]] = []
        current_measure: list[ResolvedNote] = []

        bar_idx = 0
        for n in notes:
            # Advance past barlines
            while bar_idx < len(barlines) and n.x_position > barlines[bar_idx]:
                if current_measure:
                    measures.append(current_measure)
                    current_measure = []
                bar_idx += 1
            current_measure.append(n)

        if current_measure:
            measures.append(current_measure)

        return measures if measures else [notes]

    def _group_simultaneous(
        self,
        notes: list[ResolvedNote],
        x_tolerance: float = 15.0,
    ) -> list[list[ResolvedNote]]:
        """Group notes at approximately the same x-position as chords.

        Args:
            notes: Notes sorted by x-position.
            x_tolerance: Maximum x-distance for notes to be considered simultaneous.

        Returns:
            List of groups (each group may be a single note or a chord).
        """
        if not notes:
            return []

        groups: list[list[ResolvedNote]] = [[notes[0]]]

        for n in notes[1:]:
            if abs(n.x_position - groups[-1][0].x_position) <= x_tolerance:
                groups[-1].append(n)
            else:
                groups.append([n])

        return groups
