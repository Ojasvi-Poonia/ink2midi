"""Export music21 Score to MIDI file."""

from pathlib import Path

import mido
from music21 import stream

from omr.utils.logging import get_logger

logger = get_logger("sequencer.midi_writer")


class MIDIWriter:
    """Export music21 Score to MIDI file with optional post-processing."""

    def write(
        self,
        score: stream.Score,
        output_path: str | Path,
        tempo_bpm: int = 120,
        velocity: int = 80,
    ) -> Path:
        """Write Score to MIDI file.

        Args:
            score: music21 Score object.
            output_path: Output .mid file path.
            tempo_bpm: Tempo in BPM.
            velocity: Note velocity 0-127.

        Returns:
            Path to the written MIDI file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write MIDI via music21
        score.write("midi", fp=str(output_path))

        # Post-process with mido for velocity normalization
        if velocity != 80:
            self._adjust_velocity(output_path, velocity)

        logger.info(f"Wrote MIDI to {output_path}")
        return output_path

    def _adjust_velocity(self, midi_path: Path, target_velocity: int) -> None:
        """Use mido to adjust note velocities."""
        mid = mido.MidiFile(str(midi_path))

        for track in mid.tracks:
            for msg in track:
                if msg.type == "note_on" and msg.velocity > 0:
                    msg.velocity = target_velocity

        mid.save(str(midi_path))

    def midi_to_note_list(self, midi_path: str | Path) -> list[dict]:
        """Parse a MIDI file into a list of note events for evaluation.

        Returns:
            List of dicts with: pitch, onset_ticks, duration_ticks,
            onset_seconds, duration_seconds, velocity, track.
        """
        midi_path = Path(midi_path)
        mid = mido.MidiFile(str(midi_path))
        ticks_per_beat = mid.ticks_per_beat

        notes = []
        for track_idx, track in enumerate(mid.tracks):
            current_tick = 0
            current_time_sec = 0.0
            prev_tick = 0
            active_notes: dict[int, dict] = {}  # pitch -> note_on info

            # Track tempo changes for accurate time conversion
            tempo_us = 500000  # Default 120 BPM

            for msg in track:
                # Accumulate time in seconds using current tempo
                delta_ticks = msg.time
                current_tick += delta_ticks
                current_time_sec += (delta_ticks / ticks_per_beat) * (tempo_us / 1e6)

                if msg.type == "set_tempo":
                    tempo_us = msg.tempo

                if msg.type == "note_on" and msg.velocity > 0:
                    active_notes[msg.note] = {
                        "pitch": msg.note,
                        "onset_ticks": current_tick,
                        "onset_seconds": current_time_sec,
                        "velocity": msg.velocity,
                        "track": track_idx,
                    }
                elif msg.type == "note_off" or (
                    msg.type == "note_on" and msg.velocity == 0
                ):
                    if msg.note in active_notes:
                        note_info = active_notes.pop(msg.note)
                        duration_ticks = current_tick - note_info["onset_ticks"]
                        onset_sec = note_info["onset_seconds"]
                        duration_sec = current_time_sec - onset_sec

                        notes.append({
                            "pitch": note_info["pitch"],
                            "onset_ticks": note_info["onset_ticks"],
                            "duration_ticks": duration_ticks,
                            "onset_seconds": onset_sec,
                            "duration_seconds": duration_sec,
                            "velocity": note_info["velocity"],
                            "track": note_info["track"],
                        })

        # Sort by onset time
        notes.sort(key=lambda n: (n["onset_seconds"], n["pitch"]))
        return notes
