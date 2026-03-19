"""GCN baseline model for edge classification (ablation comparison)."""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class NotationGCN(nn.Module):
    """GCN-based edge classifier for music notation relationships.

    Serves as an ablation baseline against the GAT model.
    Uses uniform aggregation instead of learned attention.
    """

    def __init__(
        self,
        node_feat_dim: int = 34,
        edge_feat_dim: int = 5,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gcn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for _ in range(num_layers):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # Edge classifier
        edge_input_dim = hidden_dim * 2 + edge_feat_dim
        self.edge_classifier = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data) -> torch.Tensor:
        """Forward pass."""
        x = self.node_encoder(data.x)

        for gcn, norm in zip(self.gcn_layers, self.layer_norms):
            x_prev = x
            x = gcn(x, data.edge_index)
            x = norm(x)
            x = torch.relu(x)
            x = x + x_prev  # Residual

        src, tgt = data.edge_index
        edge_repr = torch.cat([x[src], data.edge_attr, x[tgt]], dim=-1)
        return self.edge_classifier(edge_repr)

    @classmethod
    def load(cls, weights_path: str, device: torch.device | None = None):
        """Load a trained model."""
        checkpoint = torch.load(weights_path, map_location=device or "cpu", weights_only=False)
        config = checkpoint.get("config", {})
        model = cls(
            node_feat_dim=config.get("node_feat_dim", 34),
            edge_feat_dim=config.get("edge_feat_dim", 5),
            hidden_dim=config.get("hidden_dim", 128),
            num_layers=config.get("num_layers", 3),
            dropout=0.0,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        if device:
            model = model.to(device)
        model.eval()
        return model
