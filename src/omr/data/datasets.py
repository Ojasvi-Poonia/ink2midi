"""PyTorch Dataset classes for music score data."""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from omr.data.augmentations import MusicScoreAugmentor
from omr.utils.logging import get_logger

logger = get_logger("data.datasets")


class MusicScoreDataset(Dataset):
    """Dataset for music score images with YOLO-format labels.

    Loads image-label pairs from the YOLO directory structure:
        root/images/{split}/*.png
        root/labels/{split}/*.txt
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        img_size: int = 640,
        augment: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / "images" / split
        self.labels_dir = self.root_dir / "labels" / split
        self.img_size = img_size
        self.augmentor = MusicScoreAugmentor() if augment else None

        # Collect all image paths
        self.image_paths = sorted(
            p
            for p in self.images_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        )

        logger.info(f"Loaded {len(self.image_paths)} images from {self.images_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        """Load an image-label pair.

        Returns:
            dict with:
            - image: torch.Tensor [3, H, W] normalized to [0, 1]
            - labels: torch.Tensor [N, 5] (class_id, x_c, y_c, w, h)
            - image_path: str
        """
        img_path = self.image_paths[idx]
        label_path = self.labels_dir / f"{img_path.stem}.txt"

        # Load image
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        # Load labels
        labels = self._load_labels(label_path)

        # Apply augmentations
        if self.augmentor is not None:
            image, labels = self.augmentor(image, labels)

        # Resize
        image = cv2.resize(image, (self.img_size, self.img_size))

        # Convert to tensor: [1, H, W] -> [3, H, W] (replicate grayscale)
        image = image.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image).unsqueeze(0).repeat(3, 1, 1)

        labels_tensor = torch.zeros(0, 5)
        if labels is not None and len(labels) > 0:
            labels_tensor = torch.from_numpy(labels).float()

        return {
            "image": image_tensor,
            "labels": labels_tensor,
            "image_path": str(img_path),
        }

    def _load_labels(self, label_path: Path) -> np.ndarray | None:
        """Load YOLO-format labels from a text file."""
        if not label_path.exists():
            return None

        labels = []
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    labels.append([float(x) for x in parts])

        if not labels:
            return None

        return np.array(labels, dtype=np.float32)
