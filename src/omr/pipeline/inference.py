"""End-to-end OMR pipeline: handwritten sheet music image → MIDI file."""

from pathlib import Path

import cv2
import torch

from omr.data.graph_builder import Detection, NotationGraphBuilder
from omr.detection.postprocess import postprocess_detections
from omr.detection.yolo_trainer import SymbolDetector
from omr.relationship.gat_model import NotationGAT
from omr.relationship.inference import build_symbol_groups, predict_relationships
from omr.sequencer.midi_writer import MIDIWriter
from omr.sequencer.pitch_resolver import PitchResolver
from omr.sequencer.rhythm_resolver import RhythmResolver
from omr.sequencer.semantic_builder import ResolvedNote, SemanticBuilder, StaffData
from omr.sequencer.staff_analysis import StaffAnalyzer
from omr.utils.device import get_device
from omr.utils.logging import get_logger

logger = get_logger("pipeline.inference")


class OMRPipeline:
    """End-to-end pipeline: handwritten sheet music image → MIDI file.

    Pipeline stages:
    1. Preprocess: Load image, binarize, resize
    2. Detect: YOLOv8 symbol detection → List[Detection]
    3. Parse: Build graph from detections, run GAT → edge predictions
    4. Reconstruct: Graph → connected symbol groups
    5. Analyze: Staff detection, symbol-to-staff assignment
    6. Resolve: Pitch and rhythm resolution for each note
    7. Assemble: Build music21 Score
    8. Export: Write MIDI file
    """

    def __init__(
        self,
        detector_weights: str = "checkpoints/detection/yolov8s_muscima_finetune/weights/best.pt",
        gnn_weights: str | None = "checkpoints/gnn/gat_best.pt",
        device: str = "cuda",
        confidence: float = 0.3,
        iou_threshold: float = 0.45,
        edge_threshold: float = 0.5,
        k_neighbors: int = 8,
        max_distance: float = 200.0,
    ):
        self.device = get_device(device)

        # Module 1: Symbol Detector
        self.detector = SymbolDetector(detector_weights)

        # Module 2: GNN Relationship Parser
        self.graph_builder = NotationGraphBuilder(k_neighbors, max_distance)
        self.gnn = None
        if gnn_weights and Path(gnn_weights).exists():
            self.gnn = NotationGAT.load(gnn_weights, device=self.device)

        # Module 3: Temporal Sequencer
        self.staff_analyzer = StaffAnalyzer()
        self.pitch_resolver = PitchResolver()
        self.rhythm_resolver = RhythmResolver()
        self.semantic_builder = SemanticBuilder()
        self.midi_writer = MIDIWriter()

        # Thresholds
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.edge_threshold = edge_threshold

        logger.info(
            f"OMR Pipeline initialized on {self.device} "
            f"(GNN: {'loaded' if self.gnn else 'disabled'})"
        )

    def process(
        self,
        image_path: str | Path,
        output_path: str | Path,
        tempo: int = 120,
    ) -> dict:
        """Process a single sheet music image to MIDI.

        Args:
            image_path: Path to input score image.
            output_path: Path for output MIDI file.
            tempo: Playback tempo in BPM.

        Returns:
            Dict with processing results and metadata.
        """
        image_path = Path(image_path)
        output_path = Path(output_path)

        # 1. Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        img_h, img_w = image.shape[:2]
        logger.info(f"Processing {image_path.name} ({img_w}x{img_h})")

        # 2. Symbol detection
        detections = self.detector.predict(
            str(image_path), conf=self.confidence, iou=self.iou_threshold
        )
        detections = postprocess_detections(detections)
        logger.info(f"Detected {len(detections)} symbols")

        # 3. Relationship parsing (if GNN is available)
        edges = []
        symbol_groups = [[i] for i in range(len(detections))]  # Default: each symbol alone

        if self.gnn and len(detections) >= 2:
            edges = predict_relationships(
                self.gnn,
                detections,
                img_w,
                img_h,
                self.graph_builder,
                self.edge_threshold,
                self.device,
            )
            symbol_groups = build_symbol_groups(detections, edges)
            logger.info(f"Predicted {len(edges)} relationships, {len(symbol_groups)} groups")

        # 4. Staff analysis
        staves = self.staff_analyzer.detect_staff_lines(image)
        staff_assignments = self.staff_analyzer.assign_symbols_to_staves(detections, staves)
        logger.info(f"Detected {len(staves)} staves")

        # 5. Resolve pitch and rhythm for each staff
        staff_data_list = []
        for staff_idx, staff in enumerate(staves):
            staff_dets = staff_assignments.get(staff_idx, [])
            clef_type = self._find_clef(staff_dets)
            key_sig = self._find_key_signature(staff_dets)
            time_sig = self._find_time_signature(staff_dets)
            barlines = self._find_barlines(staff_dets)

            resolved_notes = []
            for det in staff_dets:
                if "notehead" in det.class_name:
                    connected = self._get_connected_symbols(det, detections, edges)
                    accidentals = connected.get("accidentals", [])

                    pitch = self.pitch_resolver.resolve_pitch(
                        det, staff, clef_type, key_sig, accidentals
                    )
                    duration = self.rhythm_resolver.resolve_note_duration(det, connected)

                    resolved_notes.append(
                        ResolvedNote(
                            midi_pitch=pitch,
                            duration_quarters=duration,
                            x_position=det.center[0],
                        )
                    )
                elif "rest" in det.class_name:
                    connected = self._get_connected_symbols(det, detections, edges)
                    duration = self.rhythm_resolver.resolve_rest_duration(det, connected)
                    resolved_notes.append(
                        ResolvedNote(
                            midi_pitch=0,
                            duration_quarters=duration,
                            x_position=det.center[0],
                            is_rest=True,
                        )
                    )

            # Compute key_sig_fifths: positive for sharps, negative for flats
            key_sig_fifths = 0
            if key_sig:
                num_sharps = sum(1 for v in key_sig.values() if v == "sharp")
                num_flats = sum(1 for v in key_sig.values() if v == "flat")
                key_sig_fifths = num_sharps - num_flats

            staff_data_list.append(
                StaffData(
                    resolved_notes=resolved_notes,
                    clef_type=clef_type,
                    key_sig_fifths=key_sig_fifths,
                    time_sig=time_sig,
                    barline_positions=barlines,
                )
            )

        # 6. Build music21 Score
        score = self.semantic_builder.build_score(staff_data_list, tempo)

        # 7. Write MIDI
        self.midi_writer.write(score, output_path, tempo)

        total_notes = sum(len(s.resolved_notes) for s in staff_data_list)
        logger.info(f"Generated MIDI with {total_notes} notes -> {output_path}")

        return {
            "midi_path": str(output_path),
            "num_detections": len(detections),
            "num_edges": len(edges),
            "num_staves": len(staves),
            "num_notes": total_notes,
            "score": score,
        }

    def _find_clef(self, detections: list[Detection]) -> str:
        """Find the clef type from detected symbols."""
        for det in detections:
            if det.class_name == "treble_clef":
                return "treble"
            if det.class_name == "bass_clef":
                return "bass"
            if det.class_name == "alto_clef":
                return "alto"
        return "treble"  # Default

    def _find_key_signature(self, detections: list[Detection]) -> dict[str, str] | None:
        """Extract key signature from detected accidentals at the start of the staff."""
        # Find the clef position to use as reference for key signature region
        clef_right_x = 0.0
        for d in detections:
            if d.class_name in ("treble_clef", "bass_clef", "alto_clef"):
                clef_right_x = max(clef_right_x, d.bbox[2])  # x2 of clef bbox
                break

        # Key signature accidentals appear just after the clef
        # Use clef position + reasonable margin, or fallback to 200px
        key_sig_region_end = clef_right_x + 150 if clef_right_x > 0 else 200
        accidentals = [
            d for d in detections
            if d.class_name in ("sharp", "flat")
            and d.center[0] > clef_right_x
            and d.center[0] < key_sig_region_end
        ]
        if not accidentals:
            return None

        # Sort by x-position (left to right) for correct order assignment
        accidentals.sort(key=lambda d: d.center[0])

        # Key signatures are either all sharps or all flats, pick dominant type
        sharps = [d for d in accidentals if d.class_name == "sharp"]
        flats = [d for d in accidentals if d.class_name == "flat"]

        key_sig = {}
        if len(sharps) >= len(flats):
            # Sharp key: F C G D A E B
            sharp_order = ["F", "C", "G", "D", "A", "E", "B"]
            for i in range(min(len(sharps), len(sharp_order))):
                key_sig[sharp_order[i]] = "sharp"
        else:
            # Flat key: B E A D G C F
            flat_order = ["B", "E", "A", "D", "G", "C", "F"]
            for i in range(min(len(flats), len(flat_order))):
                key_sig[flat_order[i]] = "flat"

        return key_sig if key_sig else None

    def _find_time_signature(self, detections: list[Detection]) -> tuple[int, int]:
        """Find time signature from detected symbols."""
        # For now, default to 4/4
        # Full implementation would use time_sig_digit detections + their vertical positions
        return (4, 4)

    def _find_barlines(self, detections: list[Detection]) -> list[float]:
        """Find barline x-positions."""
        barlines = [
            det.center[0]
            for det in detections
            if det.class_name in ("bar_line", "double_bar_line")
        ]
        return sorted(barlines)

    def _get_connected_symbols(
        self,
        target: Detection,
        all_detections: list[Detection],
        edges: list[tuple[int, int, float]],
    ) -> dict[str, list[Detection]]:
        """Get symbols connected to a target detection via predicted edges.

        Returns:
            Dict mapping symbol category to list of connected detections.
        """
        target_idx = None
        for i, det in enumerate(all_detections):
            if det is target:
                target_idx = i
                break

        if target_idx is None:
            return {}

        connected_indices = set()
        for src, tgt, _ in edges:
            if src == target_idx:
                connected_indices.add(tgt)
            elif tgt == target_idx:
                connected_indices.add(src)

        result: dict[str, list[Detection]] = {}
        for idx in connected_indices:
            det = all_detections[idx]
            # Categorize connected symbols
            if det.class_name in ("sharp", "flat", "natural", "double_sharp", "double_flat"):
                result.setdefault("accidentals", []).append(det)
            else:
                result.setdefault(det.class_name, []).append(det)

        return result
