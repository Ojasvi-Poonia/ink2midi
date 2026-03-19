"""GNN training loop for edge classification."""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_geometric.loader import DataLoader

from omr.relationship.graph_dataset import NotationGraphDataset
from omr.utils.device import get_device, safe_to_device
from omr.utils.logging import get_logger

logger = get_logger("relationship.trainer")


class GNNTrainer:
    """Train GAT or GCN edge classifier.

    Handles class imbalance (~90% negative edges) via pos_weight in BCE loss.
    Auto-falls back to CPU if GPU fails for PyG operations.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cuda",
        patience: int = 30,
        use_amp: bool = True,
    ):
        self.model = model
        self.device = get_device(device)
        self.model = self.model.to(self.device)
        self.patience = patience

        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=20, T_mult=2)

        # Mixed precision (FP16) — ~2x faster on NVIDIA Tensor Cores
        self.use_amp = use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        if self.use_amp:
            logger.info("Mixed precision (FP16) enabled — using CUDA Tensor Cores")

        self.pos_weight = 1.0  # Default, recomputed from training data in fit()
        self._fallback_cpu = False

    def fit(
        self,
        train_dataset: NotationGraphDataset,
        val_dataset: NotationGraphDataset,
        epochs: int = 200,
        batch_size: int = 32,
        save_dir: str = "checkpoints/gnn",
    ) -> dict:
        """Full training loop.

        Returns:
            Dict with training history and best metrics.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=(self.device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size,
            num_workers=4, pin_memory=(self.device.type == "cuda"),
        )

        # Compute class imbalance weight
        self.pos_weight = self._compute_pos_weight(train_dataset)
        logger.info(f"Positive edge weight: {self.pos_weight:.2f}")

        best_val_f1 = 0.0
        patience_counter = 0
        history = {"train_loss": [], "val_loss": [], "val_f1": []}

        for epoch in range(epochs):
            train_loss = self._train_epoch(train_loader)
            val_loss, val_metrics = self._eval_epoch(val_loader)
            self.scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_f1"].append(val_metrics["f1"])

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val F1: {val_metrics['f1']:.4f} | "
                    f"Val P: {val_metrics['precision']:.4f} | "
                    f"Val R: {val_metrics['recall']:.4f}"
                )

            # Save best model
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                patience_counter = 0
                model_name = "gcn_best.pt" if hasattr(self.model, "gcn_layers") else "gat_best.pt"
                self._save_checkpoint(save_dir / model_name, epoch, val_metrics)
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        logger.info(f"Best validation F1: {best_val_f1:.4f}")
        return {"history": history, "best_f1": best_val_f1}

    def _train_epoch(self, dataloader: DataLoader) -> float:
        """Run one training epoch with optional mixed precision."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            batch = self._to_device(batch)
            if not hasattr(batch, "y") or batch.y is None:
                continue

            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(batch).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(
                    logits,
                    batch.y,
                    pos_weight=torch.tensor([self.pos_weight], device=logits.device),
                )

            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _eval_epoch(self, dataloader: DataLoader) -> tuple[float, dict]:
        """Run one evaluation epoch."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        all_preds = []
        all_labels = []

        for batch in dataloader:
            batch = self._to_device(batch)
            if not hasattr(batch, "y") or batch.y is None:
                continue

            logits = self.model(batch).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(
                logits,
                batch.y,
                pos_weight=torch.tensor([self.pos_weight], device=logits.device),
            )

            preds = (torch.sigmoid(logits) > 0.5).cpu()
            labels = batch.y.cpu()

            all_preds.append(preds)
            all_labels.append(labels)
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        if all_preds:
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            metrics = self._compute_metrics(all_preds, all_labels)
        else:
            metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        return avg_loss, metrics

    def _compute_metrics(
        self, preds: torch.Tensor, labels: torch.Tensor
    ) -> dict:
        """Compute precision, recall, F1 for edge prediction."""
        tp = ((preds == 1) & (labels == 1)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        return {"precision": precision, "recall": recall, "f1": f1}

    def _compute_pos_weight(self, dataset: NotationGraphDataset) -> float:
        """Compute positive class weight for imbalanced edge classification."""
        num_pos = 0
        num_neg = 0
        for data in dataset:
            if hasattr(data, "y") and data.y is not None:
                num_pos += (data.y == 1).sum().item()
                num_neg += (data.y == 0).sum().item()

        if num_pos == 0:
            return 1.0
        return num_neg / num_pos

    def _to_device(self, batch):
        """Move batch to device with fallback to CPU."""
        if self._fallback_cpu:
            return batch.to("cpu")
        try:
            return batch.to(self.device)
        except RuntimeError as e:
            logger.warning(f"Falling back to CPU for GNN ({e})")
            self._fallback_cpu = True
            self.device = torch.device("cpu")
            self.model = self.model.to(self.device)
            self.use_amp = False
            self.scaler = None
            return batch.to(self.device)

    def _save_checkpoint(self, path: Path, epoch: int, metrics: dict) -> None:
        """Save model checkpoint."""
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
                "config": {
                    "node_feat_dim": self.model.node_encoder[0].in_features,
                    "edge_feat_dim": 5,
                    "hidden_dim": self.model.hidden_dim if hasattr(self.model, "hidden_dim") else 128,
                    "num_heads": self.model.num_heads if hasattr(self.model, "num_heads") else 4,
                    "num_layers": len(self.model.gat_layers) if hasattr(self.model, "gat_layers") else len(self.model.gcn_layers) if hasattr(self.model, "gcn_layers") else 3,
                },
            },
            path,
        )
        logger.info(f"Saved checkpoint to {path} (F1={metrics['f1']:.4f})")
