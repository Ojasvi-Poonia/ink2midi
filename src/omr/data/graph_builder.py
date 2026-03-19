"""Build notation graphs from symbol detections for GNN processing."""

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.data import Data

from omr.data.muscima_parser import CLASS_NAME_TO_ID
from omr.utils.logging import get_logger

logger = get_logger("data.graph_builder")

NUM_CLASSES = len(CLASS_NAME_TO_ID)


@dataclass
class Detection:
    """A single detected symbol."""

    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    confidence: float
    class_id: int
    class_name: str

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        h = self.height
        if h == 0:
            return 0.0
        return self.width / h


class NotationGraphBuilder:
    """Construct a graph from symbol detections for GNN processing.

    Nodes: Each detected symbol becomes a node.
    Node features: [class_one_hot(26), x_center, y_center, width, height,
                     aspect_ratio, area, confidence, is_notehead] = 34 dims
    Edges: k-NN based on Euclidean distance between symbol centers.
    Edge features: [dx, dy, distance, angle, relative_scale] = 5 dims
    """

    def __init__(self, k_neighbors: int = 8, max_distance_px: float = 200.0):
        self.k = k_neighbors
        self.max_dist = max_distance_px

    def build_graph(
        self,
        detections: list[Detection],
        image_width: int,
        image_height: int,
        gt_edges: list[tuple[int, int]] | None = None,
    ) -> Data:
        """Build a PyG Data object from detections.

        Args:
            detections: List of detected symbols.
            image_width: Image width for normalization.
            image_height: Image height for normalization.
            gt_edges: Optional ground truth edges as (src_idx, tgt_idx) pairs
                      for training supervision.

        Returns:
            PyG Data object with x, edge_index, edge_attr, and optionally y.
        """
        if len(detections) == 0:
            return Data(
                x=torch.zeros(0, NUM_CLASSES + 8),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                edge_attr=torch.zeros(0, 5),
            )

        # Build node features
        node_features = self._encode_nodes(detections, image_width, image_height)

        # Build k-NN edge index
        centers = torch.tensor(
            [[d.center[0], d.center[1]] for d in detections], dtype=torch.float32
        )

        k = min(self.k, len(detections) - 1)
        if k <= 0:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
        else:
            edge_index = self._knn_graph(centers, k=k)

        # Filter edges beyond max distance
        if edge_index.shape[1] > 0:
            edge_index = self._filter_by_distance(edge_index, centers)

        # Compute edge features
        edge_attr = self._compute_edge_features(
            edge_index, detections, image_width, image_height
        )

        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(detections),
        )

        # Add ground truth edge labels if provided
        if gt_edges is not None:
            gt_set = set(gt_edges)
            labels = []
            for i in range(edge_index.shape[1]):
                src = edge_index[0, i].item()
                tgt = edge_index[1, i].item()
                labels.append(1.0 if (src, tgt) in gt_set else 0.0)
            data.y = torch.tensor(labels, dtype=torch.float32)

        return data

    def _encode_nodes(
        self,
        detections: list[Detection],
        img_w: int,
        img_h: int,
    ) -> torch.Tensor:
        """Encode each detection as a 34-dim feature vector.

        Features:
        - One-hot class encoding (26 dims)
        - Normalized center x, center y (2 dims)
        - Normalized width, height (2 dims)
        - Aspect ratio (1 dim)
        - Normalized area (1 dim)
        - Detection confidence (1 dim)
        - Is-notehead flag (1 dim)
        """
        features = []
        for det in detections:
            # One-hot class
            one_hot = [0.0] * NUM_CLASSES
            if 0 <= det.class_id < NUM_CLASSES:
                one_hot[det.class_id] = 1.0

            cx, cy = det.center
            feat = one_hot + [
                cx / img_w,
                cy / img_h,
                det.width / img_w,
                det.height / img_h,
                det.aspect_ratio,
                det.area / (img_w * img_h),
                det.confidence,
                1.0 if "notehead" in det.class_name else 0.0,
            ]
            features.append(feat)

        return torch.tensor(features, dtype=torch.float32)

    @staticmethod
    def _knn_graph(centers: torch.Tensor, k: int) -> torch.Tensor:
        """Pure-PyTorch k-NN graph (no torch-cluster dependency).

        For each node, finds its k nearest neighbors by Euclidean distance
        and creates directed edges from each neighbor to the node.

        Args:
            centers: [N, 2] tensor of (x, y) coordinates.
            k: Number of nearest neighbors.

        Returns:
            edge_index: [2, N*k] tensor of (source, target) pairs.
        """
        # Pairwise Euclidean distances
        dists = torch.cdist(centers, centers, p=2)  # [N, N]

        # Set self-distance to inf to exclude self-loops
        n = centers.shape[0]
        dists.fill_diagonal_(float("inf"))

        # Get k nearest neighbors for each node
        _, indices = dists.topk(k, dim=1, largest=False)  # [N, k]

        # Build edge_index: edges from neighbors to each node
        source = indices.reshape(-1)  # [N*k]
        target = torch.arange(n).unsqueeze(1).expand(-1, k).reshape(-1)  # [N*k]

        edge_index = torch.stack([source, target], dim=0)  # [2, N*k]
        return edge_index

    def _filter_by_distance(
        self, edge_index: torch.Tensor, centers: torch.Tensor
    ) -> torch.Tensor:
        """Remove edges where symbol centers are beyond max_distance."""
        src, tgt = edge_index
        diff = centers[src] - centers[tgt]
        dists = torch.norm(diff, dim=1)
        mask = dists <= self.max_dist
        return edge_index[:, mask]

    def _compute_edge_features(
        self,
        edge_index: torch.Tensor,
        detections: list[Detection],
        img_w: int,
        img_h: int,
    ) -> torch.Tensor:
        """Compute 5-dim features for each edge.

        Features:
        - Normalized dx (signed, direction matters for left-right ordering)
        - Normalized dy (signed, direction matters for above-below)
        - Normalized Euclidean distance
        - Angle in radians / pi (normalized to [-1, 1])
        - Log ratio of source area to target area
        """
        if edge_index.shape[1] == 0:
            return torch.zeros(0, 5, dtype=torch.float32)

        features = []
        diag = math.sqrt(img_w**2 + img_h**2)

        for i in range(edge_index.shape[1]):
            src_idx = edge_index[0, i].item()
            tgt_idx = edge_index[1, i].item()
            src = detections[src_idx]
            tgt = detections[tgt_idx]

            dx = (tgt.center[0] - src.center[0]) / img_w
            dy = (tgt.center[1] - src.center[1]) / img_h
            dist = math.sqrt(
                (tgt.center[0] - src.center[0]) ** 2
                + (tgt.center[1] - src.center[1]) ** 2
            ) / diag
            angle = math.atan2(
                tgt.center[1] - src.center[1],
                tgt.center[0] - src.center[0],
            ) / math.pi

            src_area = max(src.area, 1.0)
            tgt_area = max(tgt.area, 1.0)
            scale_ratio = math.log(src_area / tgt_area)

            features.append([dx, dy, dist, angle, scale_ratio])

        return torch.tensor(features, dtype=torch.float32)
