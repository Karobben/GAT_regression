"""
Regression loss functions for absolute value prediction.
These losses are alternatives to ranking loss when absolute values matter.
"""
import torch
import torch.nn as nn


class MSELoss(nn.Module):
    """
    Mean Squared Error loss for regression.
    Suitable when absolute binding values are important.
    """
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction=reduction)
    
    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss.
        
        Args:
            scores: Predicted scores (B,)
            targets: Ground truth targets (B,)
        
        Returns:
            MSE loss
        """
        return self.mse(scores, targets)


class L1Loss(nn.Module):
    """
    L1 (Mean Absolute Error) loss for regression.
    More robust to outliers than MSE.
    """
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
        self.l1 = nn.L1Loss(reduction=reduction)
    
    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute L1 loss.
        
        Args:
            scores: Predicted scores (B,)
            targets: Ground truth targets (B,)
        
        Returns:
            L1 loss
        """
        return self.l1(scores, targets)


class SmoothL1Loss(nn.Module):
    """
    Smooth L1 (Huber) loss for regression.
    Combines benefits of L1 and L2 losses.
    """
    def __init__(self, reduction: str = "mean", beta: float = 1.0):
        """
        Initialize Smooth L1 loss.
        
        Args:
            reduction: "mean" or "sum"
            beta: Threshold for transition between L1 and L2 (default: 1.0)
        """
        super().__init__()
        self.reduction = reduction
        self.beta = beta
        self.smooth_l1 = nn.SmoothL1Loss(reduction=reduction, beta=beta)
    
    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Smooth L1 loss.
        
        Args:
            scores: Predicted scores (B,)
            targets: Ground truth targets (B,)
        
        Returns:
            Smooth L1 loss
        """
        return self.smooth_l1(scores, targets)


