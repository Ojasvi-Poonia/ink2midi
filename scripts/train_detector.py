#!/usr/bin/env python3
"""Train YOLOv8 symbol detector with optional two-phase transfer learning."""

import argparse
from pathlib import Path

from omr.detection.yolo_trainer import SymbolDetector
from omr.utils.logging import setup_logging, get_logger
from omr.utils.reproducibility import set_seed

logger = get_logger("scripts.train_detector")


def _check_dataset_ready(yaml_path: str) -> bool:
    """Check if the dataset YAML exists and the data directory has images.

    Returns True only if the dataset is genuinely ready for training
    (config exists, data directory exists, and train images are present).
    """
    path = Path(yaml_path)
    if not path.exists():
        logger.error(f"Dataset config not found: {path}")
        return False

    # Parse the YAML to find the data path
    try:
        from omegaconf import OmegaConf
        config = OmegaConf.load(path)
        data_path = config.get("path", "")
    except Exception:
        # Fallback to basic yaml parsing
        try:
            import yaml
            with open(path) as f:
                config = yaml.safe_load(f)
            data_path = config.get("path", "")
        except Exception as e:
            logger.warning(f"Could not parse dataset YAML: {e}")
            return False

    if not data_path:
        logger.warning(f"No 'path' field in {path}")
        return False

    dp = Path(data_path)
    if not dp.exists():
        logger.warning(f"Dataset path does not exist: {dp}")
        return False

    # Check for images in train split
    train_images = dp / "images" / "train"
    if not train_images.exists():
        logger.warning(f"Train images directory missing: {train_images}")
        return False

    image_files = list(train_images.glob("*.png")) + list(train_images.glob("*.jpg"))
    if not image_files:
        logger.warning(f"No images found in {train_images}")
        return False

    logger.info(f"Dataset ready: {len(image_files)} training images in {train_images}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 symbol detector")
    parser.add_argument(
        "--phase",
        choices=["pretrain", "finetune", "both"],
        default="both",
        help="Training phase: pretrain on DeepScores, finetune on MUSCIMA++, or both",
    )
    parser.add_argument("--model", default="yolov8s.pt", help="Base model")
    parser.add_argument("--epochs-pretrain", type=int, default=100)
    parser.add_argument("--epochs-finetune", type=int, default=150)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fast", action="store_true",
        help="Enable cuDNN benchmark for faster training (slightly non-deterministic)",
    )
    parser.add_argument(
        "--deepscores-yaml",
        default="configs/detection/yolov8_deepscores.yaml",
    )
    parser.add_argument(
        "--muscima-yaml",
        default="configs/detection/yolov8_muscima.yaml",
    )
    args = parser.parse_args()

    setup_logging("INFO")
    set_seed(args.seed, fast=args.fast)

    detector = SymbolDetector(args.model)
    pretrained_weights = None

    # Phase 1: Pre-train on DeepScores V2
    if args.phase in ("pretrain", "both"):
        logger.info("=== Phase 1: Pre-training on DeepScores V2 ===")
        if _check_dataset_ready(args.deepscores_yaml):
            detector.train_pretrain(
                data_yaml=args.deepscores_yaml,
                epochs=args.epochs_pretrain,
                batch=args.batch,
                imgsz=args.imgsz,
                device=args.device,
            )
            pretrained_weights = "checkpoints/detection/yolov8s_deepscores_pretrain/weights/best.pt"
        else:
            logger.warning(
                "DeepScores V2 dataset not available. Skipping pre-training.\n"
                "Run: python scripts/download_data.py && python scripts/prepare_data.py --dataset deepscores"
            )

    # Phase 2: Fine-tune on MUSCIMA++
    if args.phase in ("finetune", "both"):
        if pretrained_weights is None:
            # Look for existing pretrained weights
            candidate = Path("checkpoints/detection/yolov8s_deepscores_pretrain/weights/best.pt")
            if candidate.exists():
                pretrained_weights = str(candidate)
                logger.info(f"Using existing pretrained weights: {pretrained_weights}")
            else:
                pretrained_weights = args.model
                logger.info("No pretrained weights found, fine-tuning from base model")

        logger.info("=== Phase 2: Fine-tuning on MUSCIMA++ ===")
        if _check_dataset_ready(args.muscima_yaml):
            detector.train_finetune(
                data_yaml=args.muscima_yaml,
                pretrained_weights=pretrained_weights,
                epochs=args.epochs_finetune,
                batch=args.batch,
                imgsz=args.imgsz,
                device=args.device,
            )

            # Evaluate
            logger.info("=== Evaluation ===")
            metrics = detector.evaluate(args.muscima_yaml, device=args.device)
            logger.info(
                f"Final metrics: mAP@0.5={metrics['mAP50']:.4f}, "
                f"mAP@0.5:0.95={metrics['mAP50_95']:.4f}"
            )
        else:
            logger.error(
                "MUSCIMA++ YOLO dataset not available. Cannot fine-tune.\n"
                "Ensure CVC-MUSCIMA images are downloaded and run:\n"
                "  python scripts/prepare_data.py --dataset muscima"
            )


if __name__ == "__main__":
    main()
