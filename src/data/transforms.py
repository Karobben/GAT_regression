"""
Data transforms for graph augmentation (optional).
"""
from torch_geometric.data import Data
from typing import Optional
import torch


class NormalizeNodeFeatures:
    """Normalize node features (optional transform)."""
    
    def __call__(self, data: Data) -> Data:
        """Normalize node features to [0, 1] range."""
        x = data.x.clone()
        # Normalize each feature dimension
        x_min = x.min(dim=0, keepdim=True)[0]
        x_max = x.max(dim=0, keepdim=True)[0]
        x_range = x_max - x_min
        x_range[x_range == 0] = 1.0  # Avoid division by zero
        x = (x - x_min) / x_range
        data.x = x
        return data


class IdentityTransform:
    """Identity transform (no-op)."""
    
    def __call__(self, data: Data) -> Data:
        return data

