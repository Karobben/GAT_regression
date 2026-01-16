"""
Utility functions for training and evaluation.
"""
import random
import numpy as np
import torch
import logging
from pathlib import Path
from typing import Optional

from .run_logging import (
    get_run_id, create_run_dir, save_config, init_metrics_csv,
    append_metrics, save_cache_link, save_eval_predictions
)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: str = "auto") -> torch.device:
    """Get torch device."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def setup_logging(log_dir: Optional[Path] = None, level: int = logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            *([logging.FileHandler(log_dir / "training.log")] if log_dir else [])
        ]
    )
    return logging.getLogger(__name__)


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

