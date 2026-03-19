"""Tests for YOLO format conversion."""

import tempfile
from pathlib import Path

from omr.data.muscima_parser import DocumentAnnotation, SymbolAnnotation
from omr.data.yolo_converter import convert_to_yolo, verify_yolo_dataset


class TestYOLOConverter:
    def test_convert_basic(self, sample_document):
        with tempfile.TemporaryDirectory() as tmpdir:
            convert_to_yolo(
                [sample_document], tmpdir, split="train", copy_images=False
            )

            label_dir = Path(tmpdir) / "labels" / "train"
            assert label_dir.exists()

            label_files = list(label_dir.glob("*.txt"))
            assert len(label_files) == 1

            # Check content
            with open(label_files[0]) as f:
                lines = f.readlines()
            assert len(lines) == 3  # 3 symbols in sample_document

    def test_label_format(self, sample_document):
        with tempfile.TemporaryDirectory() as tmpdir:
            convert_to_yolo([sample_document], tmpdir, split="train", copy_images=False)

            label_file = Path(tmpdir) / "labels" / "train" / f"{sample_document.image_id}.txt"
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    assert len(parts) == 5
                    class_id = int(parts[0])
                    assert 0 <= class_id < 26
                    coords = [float(x) for x in parts[1:]]
                    assert all(0 <= c <= 1 for c in coords)

    def test_verify_valid_dataset(self, sample_document):
        with tempfile.TemporaryDirectory() as tmpdir:
            convert_to_yolo([sample_document], tmpdir, split="train", copy_images=False)
            result = verify_yolo_dataset(tmpdir)
            # Will report missing images since we didn't copy them
            assert "train" in result["splits"]

    def test_empty_document(self):
        doc = DocumentAnnotation(
            image_id="empty", image_path="", image_width=100, image_height=100
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            convert_to_yolo([doc], tmpdir, split="train", copy_images=False)
            label_dir = Path(tmpdir) / "labels" / "train"
            # No label file should be created for empty document
            assert len(list(label_dir.glob("*.txt"))) == 0
