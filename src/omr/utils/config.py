"""Configuration loading and management using OmegaConf."""

from pathlib import Path
from omegaconf import OmegaConf, DictConfig


def load_config(config_path: str | Path) -> DictConfig:
    """Load a YAML configuration file.

    Supports merging with a base config if 'base' key is present.
    """
    config_path = Path(config_path)
    cfg = OmegaConf.load(config_path)

    if "base" in cfg:
        base_path = config_path.parent / cfg.base
        base_cfg = OmegaConf.load(base_path)
        cfg = OmegaConf.merge(base_cfg, cfg)

    OmegaConf.resolve(cfg)
    return cfg


def save_config(cfg: DictConfig, output_path: str | Path) -> None:
    """Save configuration to a YAML file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_path)
