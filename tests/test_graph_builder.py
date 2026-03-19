"""Tests for notation graph construction."""

import torch

from omr.data.graph_builder import NotationGraphBuilder


class TestNotationGraphBuilder:
    def test_build_graph_basic(self, sample_detections):
        builder = NotationGraphBuilder(k_neighbors=4, max_distance_px=300)
        graph = builder.build_graph(sample_detections, image_width=800, image_height=500)

        # Check node features
        assert graph.x.shape[0] == len(sample_detections)
        assert graph.x.shape[1] == 34  # 26 one-hot + 8 spatial

        # Check edges exist
        assert graph.edge_index.shape[0] == 2
        assert graph.edge_index.shape[1] > 0

        # Check edge features
        assert graph.edge_attr.shape[1] == 5

    def test_build_graph_with_gt_edges(self, sample_detections):
        builder = NotationGraphBuilder(k_neighbors=4, max_distance_px=300)
        gt_edges = [(0, 1), (2, 0)]
        graph = builder.build_graph(
            sample_detections, 800, 500, gt_edges=gt_edges
        )

        # Should have labels
        assert hasattr(graph, "y")
        assert graph.y is not None
        assert len(graph.y) == graph.edge_index.shape[1]

    def test_empty_detections(self):
        builder = NotationGraphBuilder()
        graph = builder.build_graph([], 800, 500)

        assert graph.x.shape[0] == 0
        assert graph.edge_index.shape[1] == 0

    def test_single_detection(self, sample_detections):
        builder = NotationGraphBuilder()
        graph = builder.build_graph([sample_detections[0]], 800, 500)

        assert graph.x.shape[0] == 1
        assert graph.edge_index.shape[1] == 0  # No edges for single node

    def test_max_distance_filter(self, sample_detections):
        # Very small distance should filter most edges
        builder = NotationGraphBuilder(k_neighbors=4, max_distance_px=10)
        graph = builder.build_graph(sample_detections, 800, 500)

        # With max_distance=10, very few edges should survive
        assert graph.edge_index.shape[1] < len(sample_detections) * 4

    def test_node_features_one_hot(self, sample_detections):
        builder = NotationGraphBuilder(k_neighbors=4, max_distance_px=300)
        graph = builder.build_graph(sample_detections, 800, 500)

        # Check first detection's one-hot (class_id=0 -> notehead_filled)
        one_hot = graph.x[0, :26]
        assert one_hot[0] == 1.0
        assert one_hot.sum() == 1.0

    def test_is_notehead_flag(self, sample_detections):
        builder = NotationGraphBuilder(k_neighbors=4, max_distance_px=300)
        graph = builder.build_graph(sample_detections, 800, 500)

        # First detection is notehead_filled -> is_notehead should be 1
        assert graph.x[0, 33] == 1.0  # Last feature is is_notehead
        # Second detection is stem -> is_notehead should be 0
        assert graph.x[1, 33] == 0.0
