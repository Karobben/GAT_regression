"""Loss functions for GNN ranker."""
from .pairwise_rank_loss import PairwiseRankingLoss, compute_pairwise_accuracy
from .regression_losses import MSELoss, L1Loss, SmoothL1Loss

__all__ = [
    "PairwiseRankingLoss",
    "compute_pairwise_accuracy",
    "MSELoss",
    "L1Loss",
    "SmoothL1Loss",
    "get_loss_function"
]


def get_loss_function(loss_type: str, **kwargs):
    """
    Factory function to create loss functions.
    
    Args:
        loss_type: Type of loss function. Options:
            - "pairwise_rank" or "rank": Pairwise ranking loss (default)
            - "mse": Mean squared error loss
            - "l1": L1 (mean absolute error) loss
            - "smooth_l1": Smooth L1 loss (Huber loss)
        **kwargs: Additional arguments for the loss function
    
    Returns:
        Loss function module
    """
    loss_type = loss_type.lower()
    
    if loss_type in ["pairwise_rank", "rank", "ranking"]:
        return PairwiseRankingLoss(**kwargs)
    elif loss_type == "mse":
        return MSELoss(**kwargs)
    elif loss_type == "l1":
        return L1Loss(**kwargs)
    elif loss_type in ["smooth_l1", "huber"]:
        return SmoothL1Loss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Options: 'pairwise_rank', 'mse', 'l1', 'smooth_l1'")
