"""YOLOv8 training wrapper for music symbol detection."""

from pathlib import Path

from ultralytics import YOLO

from omr.data.graph_builder import Detection
from omr.utils.logging import get_logger

logger = get_logger("detection.yolo_trainer")


class SymbolDetector:
    """Wrapper around Ultralytics YOLOv8 for music symbol detection."""

    def __init__(self, model_path: str = "yolov8s.pt"):
        """Initialize with a pretrained or custom model.

        Args:
            model_path: Path to YOLOv8 weights or model name.
                        Use "yolov8s.pt" for pretrained COCO weights.
                        Use a custom path for fine-tuned weights.
        """
        self.model = YOLO(model_path)
        self._class_names = None
        logger.info(f"Loaded YOLOv8 model from {model_path}")

    def train(
        self,
        data_yaml: str,
        epochs: int = 150,
        imgsz: int = 640,
        batch: int = 16,
        device: str = "cuda",
        pretrained_weights: str | None = None,
        project: str = "checkpoints/detection",
        name: str = "yolov8s_run",
        **kwargs,
    ):
        """Train YOLOv8 on music symbol detection.

        Music-specific hyperparameters are applied automatically.

        Args:
            data_yaml: Path to dataset YAML config.
            epochs: Number of training epochs.
            imgsz: Input image size (square).
            batch: Batch size (16 recommended for RTX 3070 Ti 8GB).
            device: Training device ("cuda", "mps", "cpu").
            pretrained_weights: Optional path to pretrained weights for
                                transfer learning (e.g., DeepScores pretrain).
            project: Directory for checkpoints.
            name: Run name.
        """
        if pretrained_weights:
            self.model = YOLO(pretrained_weights)
            logger.info(f"Initializing from pretrained: {pretrained_weights}")

        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            amp=True,         # Mixed precision (FP16) — ~2x faster on NVIDIA GPUs
            workers=8,        # DataLoader workers for faster data loading
            # Music-specific hyperparameters
            mosaic=0.5,       # Reduced mosaic (full mosaic creates invalid combos)
            degrees=3.0,      # Very limited rotation
            scale=0.3,        # Moderate scale augmentation
            flipud=0.0,       # NEVER flip vertically (invalidates pitch)
            fliplr=0.0,       # No horizontal flip for music
            hsv_h=0.0,        # No hue shift (grayscale scores)
            hsv_s=0.0,
            hsv_v=0.2,        # Mild brightness variation
            # Loss gains
            box=7.5,
            cls=0.5,
            dfl=1.5,
            # Training settings
            patience=30,
            save_period=10,
            project=project,
            name=name,
            **kwargs,
        )
        return results

    def train_pretrain(
        self,
        data_yaml: str,
        epochs: int = 100,
        batch: int = 16,
        imgsz: int = 640,
        device: str = "cuda",
        project: str = "checkpoints/detection",
    ):
        """Phase 1: Pre-train on DeepScores V2 (synthetic data)."""
        logger.info("Phase 1: Pre-training on DeepScores V2")
        return self.train(
            data_yaml=data_yaml,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            project=project,
            name="yolov8s_deepscores_pretrain",
        )

    def train_finetune(
        self,
        data_yaml: str,
        pretrained_weights: str,
        epochs: int = 150,
        batch: int = 16,
        imgsz: int = 640,
        device: str = "cuda",
        project: str = "checkpoints/detection",
    ):
        """Phase 2: Fine-tune on MUSCIMA++ with lower learning rate."""
        logger.info("Phase 2: Fine-tuning on MUSCIMA++")
        return self.train(
            data_yaml=data_yaml,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            pretrained_weights=pretrained_weights,
            lr0=0.001,
            project=project,
            name="yolov8s_muscima_finetune",
        )

    def predict(
        self,
        image,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str = "cuda",
    ) -> list[Detection]:
        """Run inference on a single image.

        Args:
            image: Image path, numpy array, or PIL Image.
            conf: Confidence threshold.
            iou: NMS IoU threshold.
            imgsz: Inference image size.
            device: Inference device.

        Returns:
            List of Detection objects.
        """
        results = self.model.predict(
            image, conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy().tolist()
                confidence = boxes.conf[i].cpu().item()
                class_id = int(boxes.cls[i].cpu().item())
                class_name = result.names.get(class_id, f"class_{class_id}")

                detections.append(
                    Detection(
                        bbox=tuple(bbox),
                        confidence=confidence,
                        class_id=class_id,
                        class_name=class_name,
                    )
                )

        return detections

    def evaluate(self, data_yaml: str, device: str = "cuda") -> dict:
        """Run validation and return metrics."""
        results = self.model.val(data=data_yaml, device=device)
        return {
            "mAP50": results.box.map50,
            "mAP50_95": results.box.map,
            "per_class_ap50": results.box.ap50.tolist() if hasattr(results.box, 'ap50') else [],
        }
