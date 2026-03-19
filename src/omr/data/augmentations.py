"""Music-score-aware data augmentations.

These augmentations preserve musical validity — e.g., no vertical flips
(which would invert pitch meaning), no large rotations (which destroy
staff alignment).
"""

import random

import cv2
import numpy as np


class MusicScoreAugmentor:
    """Augmentation pipeline for music score images.

    All transforms operate on (image, bboxes) pairs where bboxes are
    in YOLO format: [[class_id, x_center, y_center, w, h], ...].
    """

    def __init__(
        self,
        rotation_max: float = 3.0,
        brightness_range: tuple[float, float] = (0.8, 1.2),
        noise_std: float = 10.0,
        elastic_alpha: float = 30.0,
        elastic_sigma: float = 5.0,
        p_rotation: float = 0.3,
        p_brightness: float = 0.5,
        p_noise: float = 0.3,
        p_elastic: float = 0.2,
        p_erosion_dilation: float = 0.2,
    ):
        self.rotation_max = rotation_max
        self.brightness_range = brightness_range
        self.noise_std = noise_std
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.p_rotation = p_rotation
        self.p_brightness = p_brightness
        self.p_noise = p_noise
        self.p_elastic = p_elastic
        self.p_erosion_dilation = p_erosion_dilation

    def __call__(
        self, image: np.ndarray, bboxes: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Apply random augmentations.

        Args:
            image: Input image (H, W) or (H, W, C).
            bboxes: Optional YOLO-format bboxes [N, 5].

        Returns:
            Augmented (image, bboxes) pair.
        """
        if random.random() < self.p_brightness:
            image = self._adjust_brightness(image)

        if random.random() < self.p_noise:
            image = self._add_gaussian_noise(image)

        if random.random() < self.p_rotation:
            image, bboxes = self._rotate(image, bboxes)

        if random.random() < self.p_elastic:
            image = self._elastic_deformation(image)

        if random.random() < self.p_erosion_dilation:
            image = self._random_morphology(image)

        return image, bboxes

    def _adjust_brightness(self, image: np.ndarray) -> np.ndarray:
        """Random brightness/contrast adjustment."""
        factor = random.uniform(*self.brightness_range)
        return np.clip(image * factor, 0, 255).astype(np.uint8)

    def _add_gaussian_noise(self, image: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to simulate scan artifacts."""
        noise = np.random.normal(0, self.noise_std, image.shape).astype(np.float32)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def _rotate(
        self, image: np.ndarray, bboxes: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Small rotation (max 3 degrees) to preserve staff alignment."""
        angle = random.uniform(-self.rotation_max, self.rotation_max)
        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), borderValue=255)

        # Bboxes in normalized coords are approximately invariant under
        # small rotations, so we don't transform them for <3 degrees.
        return rotated, bboxes

    def _elastic_deformation(self, image: np.ndarray) -> np.ndarray:
        """Elastic deformation to simulate handwriting variation."""
        h, w = image.shape[:2]
        dx = cv2.GaussianBlur(
            (np.random.rand(h, w).astype(np.float32) * 2 - 1),
            (0, 0),
            self.elastic_sigma,
        ) * self.elastic_alpha
        dy = cv2.GaussianBlur(
            (np.random.rand(h, w).astype(np.float32) * 2 - 1),
            (0, 0),
            self.elastic_sigma,
        ) * self.elastic_alpha

        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)

        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderValue=255)

    def _random_morphology(self, image: np.ndarray) -> np.ndarray:
        """Random erosion or dilation to simulate stroke thickness variation."""
        kernel_size = random.choice([2, 3])
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        if random.random() < 0.5:
            return cv2.erode(image, kernel, iterations=1)
        else:
            return cv2.dilate(image, kernel, iterations=1)
