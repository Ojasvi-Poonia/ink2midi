"""PyTorch Geometric dataset for notation graphs."""

from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from omr.data.graph_builder import Detection, NotationGraphBuilder
from omr.data.muscima_parser import DocumentAnnotation, SymbolAnnotation
from omr.utils.logging import get_logger

logger = get_logger("relationship.graph_dataset")


class NotationGraphDataset(Dataset):
    """Dataset of notation graphs for GNN training.

    Each sample is a PyG Data object representing one page's notation graph,
    with ground truth edge labels from MUSCIMA++ relationships.
    """

    def __init__(
        self,
        documents: list[DocumentAnnotation],
        graph_builder: NotationGraphBuilder | None = None,
        use_gt_detections: bool = True,
        predicted_detections: dict[str, list[Detection]] | None = None,
    ):
        """Initialize from parsed documents.

        Args:
            documents: Parsed annotation documents.
            graph_builder: Graph builder instance (uses default if None).
            use_gt_detections: If True, build graphs from ground truth bboxes.
                              If False, use predicted_detections.
            predicted_detections: Dict mapping image_id to predicted detections
                                  (required if use_gt_detections is False).
        """
        self.graph_builder = graph_builder or NotationGraphBuilder()

        data_list = []
        for doc in documents:
            if use_gt_detections:
                graph = self._build_from_gt(doc)
            else:
                if predicted_detections and doc.image_id in predicted_detections:
                    dets = predicted_detections[doc.image_id]
                    graph = self._build_from_predictions(doc, dets)
                else:
                    continue

            if graph is not None and graph.num_nodes > 0:
                data_list.append(graph)

        self._data_list = data_list
        logger.info(
            f"Built {len(data_list)} notation graphs "
            f"(mode={'GT' if use_gt_detections else 'predicted'})"
        )

    def _build_from_gt(self, doc: DocumentAnnotation) -> Data | None:
        """Build a graph using ground truth symbol positions."""
        if not doc.symbols:
            return None

        # Convert SymbolAnnotation to Detection
        detections = []
        for sym in doc.symbols:
            top, left, w, h = sym.bbox_abs
            det = Detection(
                bbox=(left, top, left + w, top + h),
                confidence=1.0,
                class_id=sym.class_id,
                class_name=sym.class_name,
            )
            detections.append(det)

        # Build ground truth edges from relationships
        gt_edges = [
            (rel.source_id, rel.target_id)
            for rel in doc.relationships
        ]

        return self.graph_builder.build_graph(
            detections, doc.image_width, doc.image_height, gt_edges=gt_edges
        )

    def _build_from_predictions(
        self, doc: DocumentAnnotation, detections: list[Detection]
    ) -> Data | None:
        """Build graph from predicted detections with matched GT edges."""
        if not detections or not doc.symbols:
            return None

        # Match predicted detections to GT symbols using IoU
        gt_to_pred = self._match_detections(doc.symbols, detections)

        # Transfer GT edges to predicted detection indices
        gt_edges = []
        for rel in doc.relationships:
            if rel.source_id in gt_to_pred and rel.target_id in gt_to_pred:
                pred_src = gt_to_pred[rel.source_id]
                pred_tgt = gt_to_pred[rel.target_id]
                gt_edges.append((pred_src, pred_tgt))

        return self.graph_builder.build_graph(
            detections, doc.image_width, doc.image_height, gt_edges=gt_edges
        )

    def _match_detections(
        self,
        gt_symbols: list[SymbolAnnotation],
        pred_detections: list[Detection],
        iou_threshold: float = 0.5,
    ) -> dict[int, int]:
        """Match GT symbols to predicted detections by IoU.

        Returns:
            Dict mapping GT symbol_id to predicted detection index.
        """
        from omr.detection.postprocess import _compute_iou

        matches = {}
        used_preds = set()

        for sym in gt_symbols:
            top, left, w, h = sym.bbox_abs
            gt_box = (left, top, left + w, top + h)
            best_iou = 0.0
            best_pred = -1

            for j, det in enumerate(pred_detections):
                if j in used_preds:
                    continue
                iou = _compute_iou(gt_box, det.bbox)
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_pred = j

            if best_pred >= 0:
                matches[sym.symbol_id] = best_pred
                used_preds.add(best_pred)

        return matches

    def __len__(self) -> int:
        return len(self._data_list)

    def __getitem__(self, idx: int) -> Data:
        return self._data_list[idx]
