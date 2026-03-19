"""Graph Attention Network for edge classification in notation graphs."""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class NotationGAT(nn.Module):
    """GAT-based edge classifier for music notation relationships.

    Architecture:
    1. Node encoder MLP: 34-dim → hidden_dim
    2. N GAT layers with multi-head attention + residual connections
    3. Edge classifier MLP: [src_emb || edge_feat || tgt_emb] → binary

    GAT is preferred over GCN for notation parsing because:
    - Variable-importance attention handles heterogeneous symbol types
    - Attention weights are interpretable
    - Better at learning which neighbor relationships matter most
    """

    def __init__(
        self,
        node_feat_dim: int = 34,
        edge_feat_dim: int = 5,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for i in range(num_layers):
            in_channels = hidden_dim if i == 0 else hidden_dim * num_heads
            self.gat_layers.append(
                GATConv(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    add_self_loops=True,
                )
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim * num_heads))

        # Projection for residual connections after first layer
        self.input_proj = nn.Linear(hidden_dim, hidden_dim * num_heads)

        # Edge classifier: [src_emb || edge_feat || tgt_emb] → binary
        edge_input_dim = hidden_dim * num_heads * 2 + edge_feat_dim
        self.edge_classifier = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data) -> torch.Tensor:
        """Forward pass.

        Args:
            data: PyG Data with x, edge_index, edge_attr.

        Returns:
            edge_logits: [E, 1] logit for each candidate edge.
        """
        x = self.node_encoder(data.x)

        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            x_prev = x
            x = gat(x, data.edge_index)
            x = norm(x)
            x = torch.relu(x)

            # Residual connection
            if i == 0:
                x = x + self.input_proj(x_prev)
            else:
                x = x + x_prev

        # Edge classification
        src, tgt = data.edge_index
        edge_repr = torch.cat([x[src], data.edge_attr, x[tgt]], dim=-1)
        edge_logits = self.edge_classifier(edge_repr)

        return edge_logits

    def get_attention_weights(self, data) -> list[torch.Tensor]:
        """Extract attention weights from each GAT layer for visualization."""
        x = self.node_encoder(data.x)
        attention_weights = []

        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            x_prev = x
            x, (edge_index, alpha) = gat(
                x, data.edge_index, return_attention_weights=True
            )
            attention_weights.append(alpha)
            x = norm(x)
            x = torch.relu(x)

            if i == 0:
                x = x + self.input_proj(x_prev)
            else:
                x = x + x_prev

        return attention_weights

    @classmethod
    def load(cls, weights_path: str, device: torch.device | None = None, **kwargs):
        """Load a trained model from a checkpoint."""
        checkpoint = torch.load(weights_path, map_location=device or "cpu", weights_only=False)

        # Extract model config from checkpoint or use defaults
        config = checkpoint.get("config", {})
        model = cls(
            node_feat_dim=config.get("node_feat_dim", 34),
            edge_feat_dim=config.get("edge_feat_dim", 5),
            hidden_dim=config.get("hidden_dim", 128),
            num_heads=config.get("num_heads", 4),
            num_layers=config.get("num_layers", 3),
            dropout=config.get("dropout", 0.0),  # No dropout at inference
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        if device:
            model = model.to(device)
        model.eval()
        return model
