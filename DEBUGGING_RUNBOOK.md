# GAT Ranking Model Debugging Runbook

## Overview
This document provides a step-by-step guide to debug and fix model collapse issues where the model predicts constant values.

## Quick Start: Overfit Sanity Test

### Command
```bash
python train.py --manifest manifest.csv --overfit-n 32
```

### What It Does
- Trains on first 32 samples only (no validation split)
- Runs for 500 epochs with higher learning rate (5e-3)
- Disables dropout for overfitting
- Logs detailed metrics every 10 epochs:
  - Loss
  - Pairwise accuracy
  - Spearman correlation
  - Score mean/std
  - Gradient norms

### Expected Results if Fixed
After ~50-100 epochs, you should see:
- **Pairwise accuracy > 0.9** (ideally > 0.95)
- **Spearman correlation > 0.8** (ideally > 0.9)
- **Score std > 0.1** (not collapsing to constant)
- **Gradient norms > 0.01** (gradients flowing)
- **Loss decreasing** (not stuck at constant)

### Red Flags (Still Broken)
- Pairwise accuracy ~0.5-0.6 (random)
- Spearman ~0.0-0.2 (no correlation)
- Score std < 0.01 (collapsed to constant)
- Gradient norms ~0.0 (no gradients)
- Loss not decreasing

## Key Fixes Implemented

### 1. Robust Ranking Loss (`losses/pairwise_rank_loss.py`)
**Problem**: Original loss could return zero when no valid pairs, breaking gradients.

**Fix**:
- Vectorized pairwise computation: `dy = y[:,None] - y[None,:]`
- Proper tie handling: ignores pairs with `|dy| <= tie_eps`
- Ensures loss always has gradients (returns small dummy loss if no pairs)
- Temperature scaling for numerical stability

**Key Changes**:
```python
# Vectorized computation
dy = targets[:, None] - targets[None, :]  # (B, B)
ds = scores[:, None] - scores[None, :]    # (B, B)
ds = ds / temperature  # Temperature scaling

# Valid pairs: y_i > y_j + margin AND not a tie
valid_mask = (dy > margin_eps) & (torch.abs(dy) > tie_eps)
loss = softplus(-ds[valid_mask]).mean()
```

### 2. Data Pipeline Assertions (`data/dataset.py`)
**Problem**: Silent failures in data loading (NaN, wrong shapes, empty graphs).

**Fix**:
- Assertions for node/edge counts
- NaN checks in features
- y dtype and shape validation
- Consistent float32 dtype

**Key Checks**:
```python
assert data.num_nodes > 0
assert data.edge_index.shape[1] > 0
assert not torch.isnan(data.x).any()
assert data.y.dtype == torch.float32
```

### 3. Training Stabilizers (`train.py`, `models/gat_ranker.py`)
**Problem**: Model collapse due to unstable training.

**Fixes**:
- **Gradient clipping**: `grad_clip=1.0` (configurable)
- **LayerNorm in score head**: Stabilizes activations
- **Debug logging**: Score mean/std, gradient norms
- **Overfit mode**: Tests if model can learn at all

**Model Changes**:
```python
# Added LayerNorm in score head
nn.Linear(pool_dim, hidden_dim),
nn.LayerNorm(hidden_dim),  # NEW
nn.ReLU(),
```

### 4. Enhanced Evaluation (`eval.py`)
**Problem**: Limited visibility into model behavior.

**Fix**:
- Reports score mean/std (detects collapse)
- Pairwise accuracy and Spearman
- Scatter plot visualization

## Step-by-Step Debugging

### Step 1: Run Overfit Test
```bash
python train.py --manifest manifest.csv --overfit-n 32
```

**Check**:
- Does loss decrease?
- Do gradients exist (grad_norm > 0)?
- Does score std increase?

### Step 2: If Overfit Test Fails

#### A. Check Loss Function
```bash
cd tests
python test_loss.py
```

Should pass all tests, especially:
- `test_gradients_exist`: Verifies gradients flow
- `test_perfect_vs_reversed_ordering`: Verifies loss direction

#### B. Check Data Pipeline
Add debug prints in `dataset.py`:
```python
print(f"Sample {idx}: nodes={data.num_nodes}, edges={data.edge_index.shape[1]}, y={data.y.item()}")
```

Verify:
- Graphs have nodes and edges
- y values vary (not all same)
- No NaN values

#### C. Check Model Forward
Add debug hook in `models/gat_ranker.py`:
```python
# After global pooling
print(f"Graph emb: mean={x_pooled.mean():.4f}, std={x_pooled.std():.4f}")
```

Verify:
- Graph embeddings have variation (std > 0.01)
- Not collapsing to constant

### Step 3: If Overfit Test Passes

Model can learn! Now train on full data with safe defaults:

```bash
python train.py --manifest manifest.csv
```

**Recommended Config** (in `config_example.yaml`):
```yaml
training:
  batch_size: 16
  learning_rate: 1e-3
  grad_clip: 1.0
  enable_debug_logs: true
  
loss:
  margin_eps: 0.0
  tie_eps: 1e-6
  temperature: 1.0
  weight_by_diff: true
```

## Common Issues and Solutions

### Issue: Loss is Zero
**Cause**: No valid pairs in batch (all targets equal or within margin)

**Solution**:
- Check target values vary: `print(targets)`
- Reduce `margin_eps` or increase batch size
- Check `tie_eps` is not too large

### Issue: Gradients are Zero
**Cause**: Loss detached, model frozen, or no valid pairs

**Solution**:
- Verify `scores.requires_grad == True`
- Check loss has gradients: `assert loss.requires_grad`
- Ensure valid pairs exist (see above)

### Issue: Score Std is Zero
**Cause**: Model collapsed to constant prediction

**Solution**:
- Check if overfit test passes (if not, see Step 2)
- Increase learning rate
- Disable dropout temporarily
- Check data has variation

### Issue: Pairwise Accuracy ~0.5
**Cause**: Model not learning (random predictions)

**Solution**:
- Run overfit test first
- Check loss is decreasing
- Verify gradients exist
- Increase model capacity or learning rate

## Next Steps After Fixes

1. **If overfit test passes**: Train on full data with recommended config
2. **If overfit test fails**: 
   - Check loss function tests
   - Verify data pipeline
   - Check model architecture
   - Consider simpler baseline (linear model)

3. **Monitor training**:
   - Watch score std (should increase)
   - Watch pairwise accuracy (should improve)
   - Watch gradient norms (should be > 0.01)

## Files Modified

1. `src/losses/pairwise_rank_loss.py` - Robust loss implementation
2. `train.py` - Overfit mode, debug logging, gradient clipping
3. `eval.py` - Enhanced metrics reporting
4. `src/data/dataset.py` - Data validation assertions
5. `src/models/gat_ranker.py` - LayerNorm in score head
6. `src/config.py` - New config options (grad_clip, overfit_n, etc.)
7. `tests/test_loss.py` - Gradient and ordering tests

## Summary

The key fixes prevent model collapse by:
1. **Ensuring gradients flow** (robust loss, no zero returns)
2. **Validating data** (assertions catch issues early)
3. **Stabilizing training** (gradient clipping, LayerNorm)
4. **Providing visibility** (debug logs, overfit test)

Run the overfit test first - if it passes, the model can learn. If it fails, use the debugging steps to identify the issue.


