"""Tests for GAT and GCN model forward passes."""

import torch
from torch_geometric.data import Data

from omr.relationship.gat_model import NotationGAT
from omr.relationship.gcn_model import NotationGCN


def _make_random_graph(num_nodes=20, num_edges=50):
    """Create a random PyG graph for testing."""
    x = torch.randn(num_nodes, 34)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 5)
    y = torch.randint(0, 2, (num_edges,)).float()
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


class TestNotationGAT:
    def test_forward_shape(self):
        model = NotationGAT(hidden_dim=64, num_heads=2, num_layers=2)
        data = _make_random_graph()
        output = model(data)
        assert output.shape == (data.edge_index.shape[1], 1)

    def test_gradient_flow(self):
        model = NotationGAT(hidden_dim=64, num_heads=2, num_layers=2)
        data = _make_random_graph()
        output = model(data)
        loss = output.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_small_graph(self):
        model = NotationGAT(hidden_dim=32, num_heads=2, num_layers=1)
        data = _make_random_graph(num_nodes=3, num_edges=4)
        output = model(data)
        assert output.shape == (4, 1)


class TestNotationGCN:
    def test_forward_shape(self):
        model = NotationGCN(hidden_dim=64, num_layers=2)
        data = _make_random_graph()
        output = model(data)
        assert output.shape == (data.edge_index.shape[1], 1)

    def test_gradient_flow(self):
        model = NotationGCN(hidden_dim=64, num_layers=2)
        data = _make_random_graph()
        output = model(data)
        loss = output.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
