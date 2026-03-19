"""Convert vertical staff position to MIDI pitch."""

from omr.data.graph_builder import Detection
from omr.sequencer.staff_analysis import Staff
from omr.utils.logging import get_logger

logger = get_logger("sequencer.pitch_resolver")

# MIDI note numbers for reference
# C4 (middle C) = 60, D4 = 62, E4 = 64, F4 = 65, G4 = 67, A4 = 69, B4 = 71

# Note name to semitone offset within an octave
NOTE_SEMITONES = {
    "C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11,
}

# Accidental to semitone modification
ACCIDENTAL_SEMITONES = {
    "sharp": 1,
    "flat": -1,
    "natural": 0,
    "double_sharp": 2,
    "double_flat": -2,
}

# Staff position (0 = bottom line, 4 = top line) to (note_name, octave) by clef
# Positions can be half-integers for spaces (0.5 = first space, etc.)
#
# Treble clef (G clef): G4 on line 1 (second line from bottom)
#   Bottom line (pos 0) = E4, top line (pos 4) = F5
# Bass clef (F clef): F3 on line 3 (fourth line from bottom)
#   Bottom line (pos 0) = G2, top line (pos 4) = A3
# Alto clef (C clef): C4 on line 2 (middle line)
#   Bottom line (pos 0) = F3, top line (pos 4) = G4
CLEF_MAPS = {
    "treble": {
        # Ledger lines below
        -3: ("B", 2), -2.5: ("C", 3), -2: ("D", 3), -1.5: ("E", 3),
        -1: ("F", 3), -0.5: ("G", 3),
        # Staff: bottom line E4, spaces F4 G4 etc, top line F5
        0: ("E", 4), 0.5: ("F", 4), 1: ("G", 4), 1.5: ("A", 4),
        2: ("B", 4), 2.5: ("C", 5), 3: ("D", 5), 3.5: ("E", 5),
        4: ("F", 5),
        # Ledger lines above
        4.5: ("G", 5), 5: ("A", 5), 5.5: ("B", 5),
        6: ("C", 6), 6.5: ("D", 6), 7: ("E", 6), 7.5: ("F", 6),
    },
    "bass": {
        # Ledger lines below
        -3: ("A", 0), -2.5: ("B", 0), -2: ("C", 1), -1.5: ("D", 1),
        -1: ("E", 1), -0.5: ("F", 1),
        # Staff: bottom line G2, top line A3
        0: ("G", 2), 0.5: ("A", 2), 1: ("B", 2), 1.5: ("C", 3),
        2: ("D", 3), 2.5: ("E", 3), 3: ("F", 3), 3.5: ("G", 3),
        4: ("A", 3),
        # Ledger lines above
        4.5: ("B", 3), 5: ("C", 4), 5.5: ("D", 4),
        6: ("E", 4), 6.5: ("F", 4), 7: ("G", 4),
    },
    "alto": {
        # Ledger lines below
        -3: ("G", 2), -2.5: ("A", 2), -2: ("B", 2), -1.5: ("C", 3),
        -1: ("D", 3), -0.5: ("E", 3),
        # Staff: bottom line F3, middle line (pos 2) = C4, top line G4
        0: ("F", 3), 0.5: ("G", 3), 1: ("A", 3), 1.5: ("B", 3),
        2: ("C", 4), 2.5: ("D", 4), 3: ("E", 4), 3.5: ("F", 4),
        4: ("G", 4),
        # Ledger lines above
        4.5: ("A", 4), 5: ("B", 4), 5.5: ("C", 5),
        6: ("D", 5), 6.5: ("E", 5), 7: ("F", 5),
    },
}


class PitchResolver:
    """Convert vertical position on staff to MIDI pitch number."""

    def __init__(self):
        self.clef_maps = CLEF_MAPS

    def resolve_pitch(
        self,
        notehead: Detection,
        staff: Staff,
        clef: str = "treble",
        key_signature: dict[str, str] | None = None,
        attached_accidentals: list[Detection] | None = None,
    ) -> int:
        """Resolve a notehead to a MIDI pitch number (0-127).

        Args:
            notehead: Detected notehead with bounding box.
            staff: The staff it belongs to.
            clef: Clef type ('treble', 'bass', 'alto').
            key_signature: Dict mapping note names to accidentals,
                          e.g., {"F": "sharp", "C": "sharp"} for D major.
            attached_accidentals: Accidentals linked to this note by GNN.

        Returns:
            MIDI note number (e.g., 60 = middle C).
        """
        # Step 1: Compute staff position
        staff_pos = self._compute_staff_position(
            notehead.center[1], staff.line_positions, staff.staff_space
        )

        # Step 2: Map to diatonic pitch via clef
        note_name, octave = self._position_to_pitch(staff_pos, clef)

        # Step 3: Determine accidental
        accidental_semitones = 0

        # Apply key signature first
        if key_signature and note_name in key_signature:
            acc_type = key_signature[note_name]
            accidental_semitones = ACCIDENTAL_SEMITONES.get(acc_type, 0)

        # Courtesy/explicit accidentals override key signature
        if attached_accidentals:
            for acc in attached_accidentals:
                if acc.class_name in ACCIDENTAL_SEMITONES:
                    accidental_semitones = ACCIDENTAL_SEMITONES[acc.class_name]
                    break  # Use the first attached accidental

        # Step 4: Convert to MIDI number
        midi_number = self._to_midi_number(note_name, octave, accidental_semitones)
        return max(0, min(127, midi_number))

    def _compute_staff_position(
        self,
        y_px: float,
        line_positions: list[float],
        staff_space: float,
    ) -> float:
        """Compute position as float (0=bottom line, 4=top line).

        Uses interpolation between known line positions.
        Extrapolates for ledger lines.
        """
        if not line_positions or staff_space <= 0:
            return 2.0  # Default to middle of staff

        # line_positions[0] is top line (position 4)
        # line_positions[4] is bottom line (position 0)
        # Staff is ordered top-to-bottom in image coordinates

        bottom_line_y = line_positions[-1]  # Highest y = bottom line
        top_line_y = line_positions[0]  # Lowest y = top line

        # Position increases from bottom to top (musically)
        # In image coords, y increases downward
        if bottom_line_y == top_line_y:
            return 2.0

        # Linear interpolation: bottom_line=position 0, top_line=position 4
        # Half a staff_space = half a position
        position = (bottom_line_y - y_px) / (staff_space / 2.0) * 0.5

        # Round to nearest half position (line or space)
        return round(position * 2) / 2

    def _position_to_pitch(
        self, staff_pos: float, clef: str
    ) -> tuple[str, int]:
        """Map staff position to (note_name, octave)."""
        clef_map = self.clef_maps.get(clef, self.clef_maps["treble"])

        # Find nearest mapped position
        if staff_pos in clef_map:
            return clef_map[staff_pos]

        # Extrapolate for positions beyond the map
        # Use diatonic scale pattern
        diatonic_notes = ["C", "D", "E", "F", "G", "A", "B"]

        # Get reference point
        ref_pos = min(clef_map.keys(), key=lambda p: abs(p - staff_pos))
        ref_name, ref_octave = clef_map[ref_pos]

        # Calculate steps from reference
        steps = round((staff_pos - ref_pos) * 2)  # Each half-position = 1 diatonic step
        ref_idx = diatonic_notes.index(ref_name)

        new_idx = ref_idx + steps
        octave_offset = new_idx // 7
        note_idx = new_idx % 7

        return diatonic_notes[note_idx], ref_octave + octave_offset

    def _to_midi_number(
        self, note_name: str, octave: int, accidental_semitones: int = 0
    ) -> int:
        """Convert note name + octave + accidental to MIDI number.

        MIDI note number = (octave + 1) * 12 + semitone + accidental
        Middle C (C4) = 60
        """
        semitone = NOTE_SEMITONES.get(note_name, 0)
        return (octave + 1) * 12 + semitone + accidental_semitones
