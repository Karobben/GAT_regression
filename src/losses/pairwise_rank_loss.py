"""
Pairwise ranking loss for training GNN ranker.
Implements RankNet-style logistic loss for ranking within batches.
ROBUST VERSION: Vectorized, handles ties, ensures gradients flow.
"""
import torch
import torch.nn as nn
from typing import Optional
import warnings


class PairwiseRankingLoss(nn.Module):
    """
    Pairwise ranking loss for batch-wise ranking.
    
    For each pair (i, j) where y_i > y_j + margin_eps, we enforce score_i > score_j.
    Uses logistic loss: softplus(-(score_i - score_j)) = log(1 + exp(-(score_i - score_j)))
    
    This loss is designed for ranking tasks where we care about relative ordering
    rather than absolute values. It's particularly useful when the scale of labels
    may vary or when we only have relative comparisons.
    
    FIXES for collapse prevention:
    - Vectorized pairwise computation (more efficient, clearer gradients)
    - Handles ties properly (ignores |dy| <= tie_eps)
    - Ensures loss is never zero when there are valid pairs
    - Temperature scaling for numerical stability
    """
    
    def __init__(
        self,
        margin_eps: float = 0.0,
        tie_eps: float = 1e-6,
        weight_by_diff: bool = True,
        reduction: str = "mean",
        temperature: float = 1.0
    ):
        """
        Initialize pairwise ranking loss.
        
        Args:
            margin_eps: Margin for pairwise comparisons. Only pairs where
                       y_i > y_j + margin_eps contribute to loss.
            tie_eps: Pairs with |y_i - y_j| <= tie_eps are ignored (ties)
            weight_by_diff: If True, weight loss by |y_i - y_j| to emphasize
                          confident orderings.
            reduction: "mean" or "sum" for loss reduction
            temperature: Temperature scaling for score differences (ds / temperature)
        """
        super().__init__()
        self.margin_eps = margin_eps
        self.tie_eps = tie_eps
        self.weight_by_diff = weight_by_diff
        self.reduction = reduction
        self.temperature = temperature
    
    def forward(
        self,
        scores: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pairwise ranking loss (ROBUST VERSION).
        
        Args:
            scores: Predicted scores (B,) where B is batch size, MUST have requires_grad=True
            targets: Ground-truth labels (B,)
            
        Returns:
            loss: Scalar loss value with gradients
        """
        B = scores.shape[0]
        
        # Verify inputs
        assert scores.shape == targets.shape, f"Shape mismatch: scores {scores.shape} vs targets {targets.shape}"
        
        if B < 2:
            # Return a small dummy loss that has gradients
            if scores.requires_grad:
                return 0.01 * scores.std() if scores.numel() > 1 else 0.01 * scores.abs()
            else:
                return torch.tensor(0.0, device=scores.device, requires_grad=False)
        
        # Vectorized pairwise differences: (B, B) matrices
        # dy[i,j] = y[i] - y[j], ds[i,j] = s[i] - s[j]
        dy = targets[:, None] - targets[None, :]  # (B, B)
        ds = scores[:, None] - scores[None, :]    # (B, B)
        
        # Apply temperature scaling for numerical stability
        ds = ds / self.temperature
        
        # Create mask for valid pairs: y_i > y_j + margin_eps AND not a tie
        # Ignore diagonal (self-pairs) and ties
        valid_mask = (dy > self.margin_eps) & (torch.abs(dy) > self.tie_eps)
        
        # Use lower triangle to avoid double counting (i,j) and (j,i)
        # Lower triangle means i > j, so we're checking pairs where y[i] > y[j]
        lower_triangle = torch.tril(torch.ones(B, B, device=scores.device, dtype=torch.bool), diagonal=-1)
        valid_mask_lower = valid_mask & lower_triangle
        num_lower = valid_mask_lower.sum().item()
        
        
        # If no valid pairs in lower triangle, check upper triangle
        # Upper triangle: i < j
        # For pair (i,j) where i < j: if y[i] > y[j], then dy[i,j] = y[i] - y[j] > 0
        # So we check dy > margin_eps in upper triangle (same condition as lower, just different triangle)
        if num_lower == 0:
            upper_triangle = torch.triu(torch.ones(B, B, device=scores.device, dtype=torch.bool), diagonal=1)
            # Check upper triangle: dy[i,j] > margin_eps means y[i] > y[j] when i < j
            valid_mask_upper = (dy > self.margin_eps) & (torch.abs(dy) > self.tie_eps) & upper_triangle
            num_upper = valid_mask_upper.sum().item()
            if num_upper > 0:
                # Use upper triangle: for y[i] > y[j] when i < j, we want s[i] > s[j]
                # ds[i,j] = s[i] - s[j], which is what we want
                valid_mask = valid_mask_upper
                # No need to flip ds - it's already in the right direction
                num_valid_pairs = num_upper
            else:
                valid_mask = valid_mask_lower  # Keep lower (empty)
                num_valid_pairs = 0
        else:
            valid_mask = valid_mask_lower
            num_valid_pairs = num_lower
        
        if num_valid_pairs == 0:
            # No valid pairs - return small loss to maintain gradients
            # This can happen with small batches, many ties, or when all targets are equal
            target_range = targets.max().item() - targets.min().item()
            if target_range > self.margin_eps + self.tie_eps:
                # Targets DO vary, but no valid pairs found - this is unexpected
                warnings.warn(
                    f"No valid pairs found despite target variation (B={B}, margin_eps={self.margin_eps}, "
                    f"target_range={target_range:.4f}). This may indicate a bug. Returning dummy loss."
                )
            # Return a loss that encourages some variation
            # Use scores directly to maintain gradient flow
            if scores.requires_grad:
                return 0.01 * scores.std() if scores.numel() > 1 else 0.01 * scores.abs().mean()
            else:
                return torch.tensor(0.01, device=scores.device, requires_grad=False) * scores.std()
        
        # Compute loss: for valid pairs where y_i > y_j, we want s_i > s_j
        # Loss = softplus(-ds) = log(1 + exp(-ds))
        # When ds > 0 (s_i > s_j), loss is small; when ds < 0, loss is large
        loss_per_pair = torch.nn.functional.softplus(-ds[valid_mask])  # (num_valid_pairs,)
        
        # Optional: weight by |dy| to emphasize confident orderings
        if self.weight_by_diff:
            weights = torch.abs(dy[valid_mask]).detach()  # Detach to not affect gradient flow
            loss_per_pair = loss_per_pair * weights
        
        # Aggregate loss
        if self.reduction == "mean":
            loss = loss_per_pair.mean()
        else:  # sum
            loss = loss_per_pair.sum()
        
        # Note: loss.requires_grad will be False during evaluation (torch.no_grad())
        # This is expected and fine - we only need gradients during training
        
        return loss


def compute_pairwise_accuracy(
    scores: torch.Tensor,
    targets: torch.Tensor,
    margin_eps: float = 0.0
) -> float:
    """
    Compute pairwise ranking accuracy.
    
    For each pair (i, j) where |y_i - y_j| > margin_eps, check if
    the predicted ordering matches the ground-truth ordering.
    
    Args:
        scores: Predicted scores (B,)
        targets: Ground-truth labels (B,)
        margin_eps: Margin for considering pairs (same as loss)
        
    Returns:
        accuracy: Fraction of correctly ordered pairs
    """
    B = scores.shape[0]
    
    if B < 2:
        return 1.0
    
    # Expand to get all pairs
    targets_i = targets.unsqueeze(1)  # (B, 1)
    targets_j = targets.unsqueeze(0)  # (1, B)
    target_diff = targets_i - targets_j  # (B, B)
    
    scores_i = scores.unsqueeze(1)  # (B, 1)
    scores_j = scores.unsqueeze(0)  # (1, B)
    score_diff = scores_i - scores_j  # (B, B)
    
    # Valid pairs: where |y_i - y_j| > margin_eps
    valid_mask = (torch.abs(target_diff) > margin_eps)
    
    # Correct predictions: sign(score_diff) == sign(target_diff) for valid pairs
    correct = (torch.sign(score_diff) == torch.sign(target_diff)) & valid_mask
    
    # Use lower triangle to avoid double counting
    mask_lower = torch.tril(torch.ones(B, B, device=scores.device, dtype=torch.bool), diagonal=-1)
    valid_pairs = valid_mask & mask_lower
    correct_pairs = correct & mask_lower
    
    num_valid = valid_pairs.sum().item()
    num_correct = correct_pairs.sum().item()
    
    if num_valid == 0:
        return 1.0  # No valid pairs, consider it correct
    
    return num_correct / num_valid

