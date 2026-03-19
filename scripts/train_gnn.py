#!/usr/bin/env python3
"""Train GNN relationship parser (GAT or GCN)."""

import argparse
import json
from pathlib import Path

from omr.data.graph_builder import NotationGraphBuilder
from omr.data.muscima_parser import MUSCIMAParser
from omr.relationship.gat_model import NotationGAT
from omr.relationship.gcn_model import NotationGCN
from omr.relationship.graph_dataset import NotationGraphDataset
from omr.relationship.trainer import GNNTrainer
from omr.utils.logging import setup_logging, get_logger
from omr.utils.reproducibility import set_seed

logger = get_logger("scripts.train_gnn")


def _find_annotations_dir(base_dir: Path) -> Path:
    """Find MUSCIMA++ annotations directory, searching common paths.

    The extracted MUSCIMA++ v2 zip creates:
        base_dir/v2.0/data/annotations/CVC-MUSCIMA_W-XX_N-YY_D-ideal.xml
    """
    candidates = [
        base_dir / "v2.0" / "data" / "annotations",
        base_dir / "data" / "annotations",
        base_dir / "annotations",
        base_dir,
    ]

    for candidate in candidates:
        if candidate.exists():
            xml_count = len(list(candidate.glob("CVC-MUSCIMA_*.xml")))
            if xml_count > 0:
                logger.info(f"Found {xml_count} annotation XMLs at: {candidate}")
                return candidate

    # Fallback: recursive search for annotation XMLs (skip spec files)
    for xml_file in base_dir.rglob("CVC-MUSCIMA_*.xml"):
        ann_dir = xml_file.parent
        xml_count = len(list(ann_dir.glob("CVC-MUSCIMA_*.xml")))
        if xml_count > 10:
            logger.info(f"Found {xml_count} annotations at: {ann_dir}")
            return ann_dir

    logger.warning(f"No MUSCIMA++ annotation XMLs found under {base_dir}")
    return base_dir


def main():
    parser = argparse.ArgumentParser(description="Train GNN relationship parser")
    parser.add_argument(
        "--model-type", choices=["gat", "gcn"], default="gat"
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--k-neighbors", type=int, default=8)
    parser.add_argument("--max-distance", type=float, default=200.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument(
        "--fast", action="store_true",
        help="Enable cuDNN benchmark for faster training (slightly non-deterministic)",
    )
    parser.add_argument("--annotations-dir", default="data/raw/muscima_pp_v2")
    parser.add_argument("--images-dir", default="data/raw/muscima_pp_v2/images")
    parser.add_argument("--splits-file", default="data/splits/muscima_splits.json")
    parser.add_argument("--save-dir", default="checkpoints/gnn")
    args = parser.parse_args()

    setup_logging("INFO")
    set_seed(args.seed, fast=args.fast)

    # Parse MUSCIMA++ annotations (needed for ground truth relationships)
    logger.info("Parsing MUSCIMA++ annotations for GNN training...")
    muscima_parser = MUSCIMAParser()

    # Find the actual annotations directory
    ann_dir = _find_annotations_dir(Path(args.annotations_dir))

    documents = muscima_parser.parse_directory(ann_dir, Path(args.images_dir))

    if not documents:
        logger.error(
            "No documents found. Ensure MUSCIMA++ is downloaded.\n"
            "Run: python scripts/download_data.py\n"
            "Then: python scripts/prepare_data.py"
        )
        return

    logger.info(
        f"Parsed {len(documents)} documents with "
        f"{sum(len(d.symbols) for d in documents)} symbols and "
        f"{sum(len(d.relationships) for d in documents)} relationships"
    )

    # Split documents
    splits_file = Path(args.splits_file)
    if splits_file.exists():
        with open(splits_file) as f:
            splits = json.load(f)
        train_ids = set(splits["train"])
        val_ids = set(splits["val"])
        train_docs = [d for d in documents if d.image_id in train_ids]
        val_docs = [d for d in documents if d.image_id in val_ids]
    else:
        logger.warning("Splits file not found, using 80/20 split")
        n_train = int(len(documents) * 0.8)
        train_docs = documents[:n_train]
        val_docs = documents[n_train:]

    logger.info(f"Training documents: {len(train_docs)}, Validation: {len(val_docs)}")

    if not train_docs:
        logger.error("No training documents after splitting. Check split file.")
        return

    # Build graph datasets
    graph_builder = NotationGraphBuilder(
        k_neighbors=args.k_neighbors,
        max_distance_px=args.max_distance,
    )

    logger.info("Building training graphs...")
    train_dataset = NotationGraphDataset(train_docs, graph_builder, use_gt_detections=True)
    logger.info("Building validation graphs...")
    val_dataset = NotationGraphDataset(val_docs, graph_builder, use_gt_detections=True)

    if len(train_dataset) == 0:
        logger.error("No training graphs could be built")
        return

    logger.info(f"Train graphs: {len(train_dataset)}, Val graphs: {len(val_dataset)}")

    # Create model
    if args.model_type == "gat":
        model = NotationGAT(
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    else:
        model = NotationGCN(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )

    logger.info(f"Model: {args.model_type.upper()} with {sum(p.numel() for p in model.parameters())} parameters")

    # Train
    trainer = GNNTrainer(
        model=model,
        lr=args.lr,
        device=args.device,
        patience=args.patience,
    )

    results = trainer.fit(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
    )

    logger.info(f"Training complete. Best F1: {results['best_f1']:.4f}")


if __name__ == "__main__":
    main()
