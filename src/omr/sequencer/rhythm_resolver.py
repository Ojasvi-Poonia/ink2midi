"""Determine duration of each note from symbol relationships."""

from omr.data.graph_builder import Detection
from omr.utils.logging import get_logger

logger = get_logger("sequencer.rhythm_resolver")

# Base durations in quarter-note units
BASE_DURATIONS = {
    "notehead_whole": 4.0,
    "notehead_half": 2.0,
    "notehead_filled": 1.0,  # Default quarter; modified by beams/flags
}

# Rest durations in quarter-note units
REST_DURATIONS = {
    "rest_whole": 4.0,
    "rest_half": 2.0,
    "rest_quarter": 1.0,
    "rest_eighth": 0.5,
    "rest_sixteenth": 0.25,
}

# Flag types that indicate subdivision
FLAG_CLASSES = {
    "flag_eighth_up",
    "flag_eighth_down",
    "flag_sixteenth_up",
    "flag_sixteenth_down",
}


class RhythmResolver:
    """Determine duration of notes and rests from symbol relationships."""

    def resolve_note_duration(
        self,
        notehead: Detection,
        connected_symbols: dict[str, list[Detection]],
    ) -> float:
        """Resolve a notehead's duration in quarter-note units.

        Algorithm:
        1. Get base duration from notehead type
        2. Count beams and flags → subdivision
        3. Apply augmentation dots

        Args:
            notehead: The notehead detection.
            connected_symbols: Symbols linked by GNN, keyed by class name.

        Returns:
            Duration in quarter-note units (1.0 = quarter note).
        """
        base = BASE_DURATIONS.get(notehead.class_name, 1.0)

        # Whole notes and half notes without beams/flags keep their duration
        if notehead.class_name in ("notehead_whole", "notehead_half"):
            # Dots can still modify these
            num_dots = len(connected_symbols.get("dot", []))
            return self._apply_dots(base, num_dots)

        # For filled noteheads: count beams and flags
        num_beams = len(connected_symbols.get("beam", []))

        num_flags = 0
        for flag_class in FLAG_CLASSES:
            num_flags += len(connected_symbols.get(flag_class, []))

        # Determine subdivision level
        # Beams and flags indicate the same thing, take the max
        subdivision = max(num_beams, num_flags)

        if subdivision > 0:
            # Each beam/flag halves the duration
            # 1 beam = eighth (0.5), 2 beams = sixteenth (0.25), etc.
            base = base / (2**subdivision)

        # Apply dots
        num_dots = len(connected_symbols.get("dot", []))
        base = self._apply_dots(base, num_dots)

        return base

    def resolve_rest_duration(
        self,
        rest: Detection,
        connected_symbols: dict[str, list[Detection]] | None = None,
    ) -> float:
        """Resolve a rest symbol's duration.

        Args:
            rest: The rest detection.
            connected_symbols: Optional symbols linked by GNN (for dotted rests).

        Returns:
            Duration in quarter-note units.
        """
        base = REST_DURATIONS.get(rest.class_name, 1.0)
        if connected_symbols:
            num_dots = len(connected_symbols.get("dot", []))
            base = self._apply_dots(base, num_dots)
        return base

    def _apply_dots(self, duration: float, num_dots: int) -> float:
        """Apply augmentation dots to a duration.

        Single dot = duration * 1.5
        Double dot = duration * 1.75
        Triple dot = duration * 1.875
        """
        if num_dots <= 0:
            return duration

        total = duration
        added = duration
        for _ in range(num_dots):
            added /= 2
            total += added

        return total

    def validate_measure_duration(
        self,
        note_durations: list[float],
        time_signature: tuple[int, int] = (4, 4),
    ) -> dict:
        """Validate that note durations sum to expected beats per measure.

        Args:
            note_durations: List of durations in quarter-note units.
            time_signature: (numerator, denominator), e.g., (4, 4).

        Returns:
            Dict with validation results.
        """
        num, denom = time_signature
        # Expected total in quarter-note units
        expected = num * (4.0 / denom)
        actual = sum(note_durations)

        return {
            "expected": expected,
            "actual": actual,
            "difference": actual - expected,
            "valid": abs(actual - expected) < 0.01,
        }
