"""GNN inference: predict edges from detections."""

import torch

from omr.data.graph_builder import Detection, NotationGraphBuilder
from omr.utils.logging import get_logger

logger = get_logger("relationship.inference")


def predict_relationships(
    model: torch.nn.Module,
    detections: list[Detection],
    image_width: int,
    image_height: int,
    graph_builder: NotationGraphBuilder | None = None,
    edge_threshold: float = 0.5,
    device: torch.device | None = None,
) -> list[tuple[int, int, float]]:
    """Predict notation graph edges from symbol detections.

    Args:
        model: Trained GAT or GCN model.
        detections: List of detected symbols.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        graph_builder: Graph builder instance.
        edge_threshold: Probability threshold for edge prediction.
        device: Inference device.

    Returns:
        List of (source_idx, target_idx, probability) tuples
        for predicted edges.
    """
    if len(detections) < 2:
        return []

    graph_builder = graph_builder or NotationGraphBuilder()
    device = device or torch.device("cpu")

    # Build candidate graph
    graph = graph_builder.build_graph(detections, image_width, image_height)

    if graph.edge_index.shape[1] == 0:
        return []

    # Move model and graph to device
    model = model.to(device)
    try:
        graph = graph.to(device)
    except RuntimeError:
        device = torch.device("cpu")
        graph = graph.to(device)
        model = model.to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        logits = model(graph).squeeze(-1)
        probs = torch.sigmoid(logits)

    # Filter by threshold
    edges = []
    for i in range(graph.edge_index.shape[1]):
        prob = probs[i].item()
        if prob >= edge_threshold:
            src = graph.edge_index[0, i].item()
            tgt = graph.edge_index[1, i].item()
            edges.append((src, tgt, prob))

    logger.debug(
        f"Predicted {len(edges)} edges from "
        f"{graph.edge_index.shape[1]} candidates"
    )
    return edges


def build_symbol_groups(
    detections: list[Detection],
    edges: list[tuple[int, int, float]],
) -> list[list[int]]:
    """Group connected symbols using union-find on predicted edges.

    Returns:
        List of connected components, each a list of detection indices.
    """
    n = len(detections)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for src, tgt, _ in edges:
        if src < n and tgt < n:
            union(src, tgt)

    # Collect groups
    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    return list(groups.values())
