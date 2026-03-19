#!/usr/bin/env python3
"""Run end-to-end inference: sheet music image → MIDI file."""

import argparse
from pathlib import Path

from omr.pipeline.inference import OMRPipeline
from omr.utils.logging import setup_logging, get_logger

logger = get_logger("scripts.run_inference")


def main():
    parser = argparse.ArgumentParser(description="Convert sheet music image to MIDI")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output", default=None, help="Output MIDI path")
    parser.add_argument(
        "--detector-weights",
        default="checkpoints/detection/yolov8s_muscima_finetune/weights/best.pt",
    )
    parser.add_argument("--gnn-weights", default="checkpoints/gnn/gat_best.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--tempo", type=int, default=120)
    parser.add_argument("--visualize", action="store_true", help="Save detection visualization")
    args = parser.parse_args()

    setup_logging("INFO")

    # Set output path
    image_path = Path(args.image)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = image_path.with_suffix(".mid")

    # Initialize pipeline
    gnn_weights = args.gnn_weights if Path(args.gnn_weights).exists() else None
    pipeline = OMRPipeline(
        detector_weights=args.detector_weights,
        gnn_weights=gnn_weights,
        device=args.device,
        confidence=args.confidence,
    )

    # Process
    result = pipeline.process(image_path, output_path, tempo=args.tempo)

    logger.info(
        f"Done! {result['num_detections']} symbols detected, "
        f"{result['num_notes']} notes generated -> {result['midi_path']}"
    )

    # Optional visualization
    if args.visualize:
        from omr.detection.visualize import save_detection_visualization

        viz_path = image_path.with_name(f"{image_path.stem}_detections.png")
        detections = pipeline.detector.predict(str(image_path), conf=args.confidence)
        save_detection_visualization(image_path, detections, viz_path)
        logger.info(f"Detection visualization saved to {viz_path}")


if __name__ == "__main__":
    main()
