"""
Unit tests for pairwise ranking loss.
"""
import torch
import numpy as np

# Add src to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from losses.pairwise_rank_loss import PairwiseRankingLoss, compute_pairwise_accuracy


def test_pairwise_ranking_loss_basic():
    """Test basic pairwise ranking loss."""
    criterion = PairwiseRankingLoss(margin_eps=0.0, weight_by_diff=False)
    
    # Simple case: scores should match ordering of targets
    scores = torch.tensor([1.0, 2.0, 3.0])  # Increasing
    targets = torch.tensor([1.0, 2.0, 3.0])  # Increasing
    
    loss = criterion(scores, targets)
    assert loss.item() >= 0  # Loss should be non-negative
    
    # Reverse ordering should give higher loss
    scores_reverse = torch.tensor([3.0, 2.0, 1.0])  # Decreasing
    loss_reverse = criterion(scores_reverse, targets)
    
    # Reverse ordering should have higher loss
    assert loss_reverse.item() > loss.item()


def test_pairwise_ranking_loss_margin():
    """Test pairwise ranking loss with margin."""
    criterion_no_margin = PairwiseRankingLoss(margin_eps=0.0)
    criterion_margin = PairwiseRankingLoss(margin_eps=1.0)
    
    scores = torch.tensor([1.0, 1.5, 2.0])
    targets = torch.tensor([1.0, 2.0, 3.0])  # Differences: 1.0, 1.0, 2.0
    
    loss_no_margin = criterion_no_margin(scores, targets)
    loss_margin = criterion_margin(scores, targets)
    
    # With margin, fewer pairs contribute, so loss should be different
    assert loss_no_margin.item() != loss_margin.item()


def test_pairwise_ranking_loss_weighting():
    """Test pairwise ranking loss with weighting by difference."""
    criterion_no_weight = PairwiseRankingLoss(weight_by_diff=False)
    criterion_weight = PairwiseRankingLoss(weight_by_diff=True)
    
    scores = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([1.0, 2.0, 10.0])  # Large difference between 2 and 3
    
    loss_no_weight = criterion_no_weight(scores, targets)
    loss_weight = criterion_weight(scores, targets)
    
    # Weighted version should emphasize large differences
    assert loss_weight.item() != loss_no_weight.item()


def test_pairwise_accuracy():
    """Test pairwise accuracy computation."""
    # Perfect ordering
    scores = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([1.0, 2.0, 3.0])
    
    acc = compute_pairwise_accuracy(scores, targets, margin_eps=0.0)
    assert acc == 1.0  # Perfect accuracy
    
    # Reverse ordering
    scores_reverse = torch.tensor([3.0, 2.0, 1.0])
    acc_reverse = compute_pairwise_accuracy(scores_reverse, targets, margin_eps=0.0)
    assert acc_reverse == 0.0  # All pairs wrong
    
    # Partial ordering
    scores_partial = torch.tensor([1.0, 3.0, 2.0])
    acc_partial = compute_pairwise_accuracy(scores_partial, targets, margin_eps=0.0)
    assert 0.0 < acc_partial < 1.0  # Some pairs correct


def test_pairwise_accuracy_margin():
    """Test pairwise accuracy with margin."""
    scores = torch.tensor([1.0, 1.1, 1.2])
    targets = torch.tensor([1.0, 1.05, 1.1])  # Small differences
    
    # With margin, pairs within margin are ignored
    acc_no_margin = compute_pairwise_accuracy(scores, targets, margin_eps=0.0)
    acc_margin = compute_pairwise_accuracy(scores, targets, margin_eps=0.2)
    
    # With large margin, fewer pairs are considered
    assert acc_margin >= acc_no_margin


def test_edge_cases():
    """Test edge cases for loss."""
    criterion = PairwiseRankingLoss()
    
    # Single sample (should return small dummy loss)
    scores = torch.tensor([1.0], requires_grad=True)
    targets = torch.tensor([1.0])
    loss = criterion(scores, targets)
    assert loss.item() >= 0.0
    assert loss.requires_grad
    
    # Two samples
    scores = torch.tensor([1.0, 2.0], requires_grad=True)
    targets = torch.tensor([1.0, 2.0])
    loss = criterion(scores, targets)
    assert loss.item() >= 0.0
    assert loss.requires_grad
    
    # All equal targets (no valid pairs) - should return small dummy loss
    scores = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    targets = torch.tensor([1.0, 1.0, 1.0])
    loss = criterion(scores, targets)
    assert loss.item() >= 0.0
    assert loss.requires_grad


def test_gradients_exist():
    """Test that gradients flow through the loss."""
    criterion = PairwiseRankingLoss(margin_eps=0.0, weight_by_diff=False)
    
    # Perfect ordering - should have low loss and gradients
    scores = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    targets = torch.tensor([1.0, 2.0, 3.0, 4.0])
    
    loss = criterion(scores, targets)
    assert loss.requires_grad, "Loss must have requires_grad=True"
    
    # Backward pass
    loss.backward()
    
    # Check gradients exist and are non-zero
    assert scores.grad is not None, "Gradients must exist"
    assert torch.abs(scores.grad).sum() > 0, "Gradients must be non-zero"
    
    # Reverse ordering - should have higher loss
    scores2 = torch.tensor([4.0, 3.0, 2.0, 1.0], requires_grad=True)
    loss2 = criterion(scores2, targets)
    loss2.backward()
    
    assert loss2.item() > loss.item(), "Reverse ordering should have higher loss"
    assert scores2.grad is not None and torch.abs(scores2.grad).sum() > 0


def test_perfect_vs_reversed_ordering():
    """Test that perfect ordering gives lower loss than reversed."""
    criterion = PairwiseRankingLoss(margin_eps=0.0, weight_by_diff=False)
    
    targets = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Perfect ordering
    scores_perfect = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
    loss_perfect = criterion(scores_perfect, targets)
    
    # Reversed ordering
    scores_reversed = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0], requires_grad=True)
    loss_reversed = criterion(scores_reversed, targets)
    
    # Perfect should have much lower loss
    assert loss_perfect.item() < loss_reversed.item(), \
        f"Perfect ordering loss ({loss_perfect.item():.4f}) should be < reversed ({loss_reversed.item():.4f})"
    
    # Both should have gradients
    loss_perfect.backward()
    loss_reversed.backward()
    assert scores_perfect.grad is not None
    assert scores_reversed.grad is not None


if __name__ == "__main__":
    print("Running loss tests...")
    
    test_pairwise_ranking_loss_basic()
    print("✓ test_pairwise_ranking_loss_basic")
    
    test_pairwise_ranking_loss_margin()
    print("✓ test_pairwise_ranking_loss_margin")
    
    test_pairwise_ranking_loss_weighting()
    print("✓ test_pairwise_ranking_loss_weighting")
    
    test_pairwise_accuracy()
    print("✓ test_pairwise_accuracy")
    
    test_pairwise_accuracy_margin()
    print("✓ test_pairwise_accuracy_margin")
    
    test_edge_cases()
    print("✓ test_edge_cases")
    
    test_gradients_exist()
    print("✓ test_gradients_exist")
    
    test_perfect_vs_reversed_ordering()
    print("✓ test_perfect_vs_reversed_ordering")
    
    print("\nAll loss tests passed!")

