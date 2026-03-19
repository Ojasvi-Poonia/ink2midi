"""Tests for MIDI writer."""

import tempfile
from pathlib import Path

import pytest
from music21 import note, stream, tempo

from omr.sequencer.midi_writer import MIDIWriter


@pytest.fixture
def writer():
    return MIDIWriter()


@pytest.fixture
def simple_score():
    """A simple 4-note score for testing."""
    s = stream.Score()
    s.insert(0, tempo.MetronomeMark(number=120))

    p = stream.Part()
    for midi_pitch in [60, 62, 64, 65]:  # C D E F
        n = note.Note(midi_pitch)
        n.quarterLength = 1.0
        p.append(n)

    s.append(p)
    return s


class TestMIDIWriter:
    def test_write_creates_file(self, writer, simple_score):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.mid"
            result = writer.write(simple_score, path)
            assert result.exists()
            assert result.stat().st_size > 0

    def test_midi_to_note_list(self, writer, simple_score):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.mid"
            writer.write(simple_score, path)

            notes = writer.midi_to_note_list(path)
            assert len(notes) == 4
            assert notes[0]["pitch"] == 60  # C4
            assert notes[1]["pitch"] == 62  # D4
            assert notes[2]["pitch"] == 64  # E4
            assert notes[3]["pitch"] == 65  # F4

    def test_velocity_adjustment(self, writer, simple_score):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.mid"
            writer.write(simple_score, path, velocity=100)

            notes = writer.midi_to_note_list(path)
            for n in notes:
                assert n["velocity"] == 100

    def test_note_durations(self, writer, simple_score):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.mid"
            writer.write(simple_score, path)

            notes = writer.midi_to_note_list(path)
            # All notes are quarter notes at 120 BPM = 0.5 seconds each
            for n in notes:
                assert abs(n["duration_seconds"] - 0.5) < 0.1
