"""
Training script for GAT ranker model.
"""
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import numpy as np
from scipy.stats import spearmanr, pearsonr
from pathlib import Path
import argparse
from tqdm import tqdm
import json

# Try to import matplotlib (optional)
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Graphic reports will not be generated.")

from src.config import Config
from src.data.dataset import AntibodyAntigenDataset
from src.data.transforms import IdentityTransform, NormalizeNodeFeatures
from src.models.gat_ranker import GATRanker
from src.losses.pairwise_rank_loss import PairwiseRankingLoss, compute_pairwise_accuracy
from src.losses import get_loss_function
from src.utils import set_seed, get_device, setup_logging, count_parameters
from src.utils.run_logging import (
    create_run_dir, save_config, init_metrics_csv, append_metrics, save_cache_link
)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 0.0,
    enable_debug: bool = False
) -> dict:
    """
    Train for one epoch with debug logging and gradient clipping.
    
    Args:
        model: Model to train
        loader: Data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device
        grad_clip: Gradient clipping norm (0 = disabled)
        enable_debug: Enable detailed debug logging
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    all_scores = []
    all_targets = []
    grad_norms = []
    
    for batch in loader:
        batch = batch.to(device)
        
        # Forward pass
        scores = model.forward_batch(batch)
        
        # Extract targets - handle both [B, 1] and [B] shapes
        if batch.y.dim() > 1:
            targets = batch.y.squeeze(-1)  # (B,)
        else:
            targets = batch.y  # (B,)
        
        # Verify shapes match
        assert scores.shape == targets.shape, f"Shape mismatch: scores {scores.shape} vs targets {targets.shape}"
        
        # Verify scores have gradients
        assert scores.requires_grad, "Scores must have requires_grad=True!"
        
        # Compute loss
        loss = criterion(scores, targets)
        
        # Verify loss has gradients
        assert loss.requires_grad, "Loss must have requires_grad=True!"
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            grad_norms.append(grad_norm.item())
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Collect for metrics
        all_scores.append(scores.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())
    
    # Compute metrics
    all_scores = np.concatenate(all_scores)
    all_targets = np.concatenate(all_targets)
    
    metrics = {
        "loss": total_loss / num_batches if num_batches > 0 else 0.0,
        "score_mean": float(all_scores.mean()),
        "score_std": float(all_scores.std()),
        "target_mean": float(all_targets.mean()),
        "target_std": float(all_targets.std()),
    }
    
    if grad_clip > 0 and grad_norms:
        metrics["grad_norm_mean"] = float(np.mean(grad_norms))
        metrics["grad_norm_max"] = float(np.max(grad_norms))
    
    if enable_debug:
        metrics["score_min"] = float(all_scores.min())
        metrics["score_max"] = float(all_scores.max())
        metrics["target_min"] = float(all_targets.min())
        metrics["target_max"] = float(all_targets.max())
    
    return metrics


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    margin_eps: float = 0.0,
    loss_type: str = "pairwise_rank"
) -> dict:
    """Evaluate model on validation/test set."""
    model.eval()
    all_scores = []
    all_targets = []
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            # Forward pass
            scores = model.forward_batch(batch)
            
            # Extract targets - handle both [B, 1] and [B] shapes
            if batch.y.dim() > 1 and batch.y.shape[-1] == 1:
                targets = batch.y.squeeze(-1)  # (B,)
            elif batch.y.dim() == 1:
                targets = batch.y  # (B,)
            else:
                # Flatten if needed
                targets = batch.y.view(-1)  # (B,)
            
            # Verify shapes match
            assert scores.shape == targets.shape, f"Shape mismatch: scores {scores.shape} vs targets {targets.shape}"
            
            # Compute loss
            loss = criterion(scores, targets)
            
            all_scores.append(scores.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            total_loss += loss.item()
            num_batches += 1
    
    # Concatenate all predictions and targets
    all_scores = np.concatenate(all_scores)
    all_targets = np.concatenate(all_targets)
    
    # Base metrics
    metrics = {
        "loss": total_loss / num_batches if num_batches > 0 else 0.0,
        "score_mean": float(all_scores.mean()),
        "score_std": float(all_scores.std()),
        "score_min": float(all_scores.min()),
        "score_max": float(all_scores.max()),
        "target_mean": float(all_targets.mean()),
        "target_std": float(all_targets.std()),
    }
    
    # Compute loss-specific metrics
    if loss_type in ["pairwise_rank", "rank", "ranking"]:
        # Ranking metrics for ranking loss
        if len(all_scores) > 1:
            spearman_corr, spearman_p = spearmanr(all_scores, all_targets)
            metrics["spearman"] = spearman_corr
            metrics["spearman_p"] = spearman_p
        else:
            metrics["spearman"] = 0.0
            metrics["spearman_p"] = float('nan')
        
        # Pairwise accuracy
        scores_tensor = torch.tensor(all_scores)
        targets_tensor = torch.tensor(all_targets)
        pairwise_acc = compute_pairwise_accuracy(scores_tensor, targets_tensor, margin_eps=margin_eps)
        metrics["pairwise_acc"] = pairwise_acc
    else:
        # Regression metrics for MSE/L1/SmoothL1
        # Mean Absolute Error (MAE)
        mae = np.mean(np.abs(all_scores - all_targets))
        metrics["mae"] = float(mae)
        
        # Root Mean Squared Error (RMSE)
        rmse = np.sqrt(np.mean((all_scores - all_targets) ** 2))
        metrics["rmse"] = float(rmse)
        
        # R² (coefficient of determination)
        ss_res = np.sum((all_targets - all_scores) ** 2)
        ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
        if ss_tot > 0:
            r2 = 1 - (ss_res / ss_tot)
        else:
            r2 = float('nan')
        metrics["r2"] = float(r2) if not np.isnan(r2) else float('nan')
        
        # Pearson correlation
        if len(all_scores) > 1:
            from scipy.stats import pearsonr
            pearson_corr, pearson_p = pearsonr(all_scores, all_targets)
            metrics["pearson"] = float(pearson_corr)
            metrics["pearson_p"] = float(pearson_p) if not np.isnan(pearson_p) else float('nan')
        else:
            metrics["pearson"] = 0.0
            metrics["pearson_p"] = float('nan')
    
    return metrics


def get_predictions_and_targets(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> tuple:
    """
    Get predictions and targets from model.
    
    Returns:
        (predictions, targets) as numpy arrays
    """
    model.eval()
    all_scores = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            scores = model.forward_batch(batch)
            
            # Extract targets
            if batch.y.dim() > 1 and batch.y.shape[-1] == 1:
                targets = batch.y.squeeze(-1)
            elif batch.y.dim() == 1:
                targets = batch.y
            else:
                targets = batch.y.view(-1)
            
            all_scores.append(scores.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    return np.concatenate(all_scores), np.concatenate(all_targets)


def create_training_report(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_history: list,
    val_history: list,
    config,
    log_dir: Path,
    device: torch.device
):
    """
    Create graphic report with scatter plots and training curves.
    
    Args:
        model: Trained model
        train_loader: Training data loader
        val_loader: Validation data loader
        train_history: List of training metrics per epoch
        val_history: List of validation metrics per epoch
        config: Config object
        log_dir: Directory to save plots
        device: Device
    """
    if not HAS_MATPLOTLIB:
        print("Skipping graphic report (matplotlib not available)")
        return
    
    print("\nGenerating training report...")
    
    # Get predictions on validation set
    val_predictions, val_targets = get_predictions_and_targets(model, val_loader, device)
    
    # Get predictions on training set (for comparison)
    train_predictions, train_targets = get_predictions_and_targets(model, train_loader, device)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Validation scatter plot (most important)
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(val_targets, val_predictions, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # Add diagonal line
    min_val = min(val_targets.min(), val_predictions.min())
    max_val = max(val_targets.max(), val_predictions.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect prediction')
    
    # Add trend line
    if len(val_predictions) > 1:
        z = np.polyfit(val_targets, val_predictions, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min_val, max_val, 100)
        ax1.plot(x_trend, p(x_trend), 'b-', alpha=0.7, linewidth=2, label='Linear fit')
    
    # Compute metrics for title
    if config.loss.loss_type in ["pairwise_rank", "rank", "ranking"]:
        spearman_corr, spearman_p = spearmanr(val_predictions, val_targets)
        title = f'Validation: Predicted vs Ground Truth\n'
        title += f'Spearman ρ = {spearman_corr:.4f}'
        if not np.isnan(spearman_p):
            title += f' (p = {spearman_p:.2e})'
    else:
        mae = np.mean(np.abs(val_predictions - val_targets))
        rmse = np.sqrt(np.mean((val_predictions - val_targets) ** 2))
        ss_res = np.sum((val_targets - val_predictions) ** 2)
        ss_tot = np.sum((val_targets - np.mean(val_targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else float('nan')
        pearson_corr, pearson_p = pearsonr(val_predictions, val_targets) if len(val_predictions) > 1 else (0.0, float('nan'))
        
        title = f'Validation: Predicted vs Ground Truth\n'
        title += f'MAE = {mae:.4f}, RMSE = {rmse:.4f}, R² = {r2:.4f}'
        if not np.isnan(pearson_corr):
            title += f'\nPearson r = {pearson_corr:.4f}'
    
    ax1.set_xlabel('Ground Truth', fontsize=12)
    ax1.set_ylabel('Predicted', fontsize=12)
    ax1.set_title(title, fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    ax1.set_aspect('auto')
    
    # 2. Training scatter plot
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(train_targets, train_predictions, alpha=0.6, s=50, edgecolors='black', linewidth=0.5, color='green')
    
    min_val = min(train_targets.min(), train_predictions.min())
    max_val = max(train_targets.max(), train_predictions.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect prediction')
    
    if len(train_predictions) > 1:
        z = np.polyfit(train_targets, train_predictions, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min_val, max_val, 100)
        ax2.plot(x_trend, p(x_trend), 'b-', alpha=0.7, linewidth=2, label='Linear fit')
    
    if config.loss.loss_type in ["pairwise_rank", "rank", "ranking"]:
        spearman_corr, _ = spearmanr(train_predictions, train_targets)
        title = f'Training: Predicted vs Ground Truth\nSpearman ρ = {spearman_corr:.4f}'
    else:
        mae = np.mean(np.abs(train_predictions - train_targets))
        rmse = np.sqrt(np.mean((train_predictions - train_targets) ** 2))
        ss_res = np.sum((train_targets - train_predictions) ** 2)
        ss_tot = np.sum((train_targets - np.mean(train_targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else float('nan')
        title = f'Training: Predicted vs Ground Truth\nMAE = {mae:.4f}, RMSE = {rmse:.4f}, R² = {r2:.4f}'
    
    ax2.set_xlabel('Ground Truth', fontsize=12)
    ax2.set_ylabel('Predicted', fontsize=12)
    ax2.set_title(title, fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    ax2.set_aspect('auto')
    
    # 3. Training loss curve
    ax3 = plt.subplot(2, 3, 3)
    epochs = range(1, len(train_history) + 1)
    train_losses = [h['loss'] for h in train_history]
    val_losses = [h['loss'] for h in val_history]
    ax3.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax3.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Loss', fontsize=12)
    ax3.set_title('Training and Validation Loss', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best')
    
    # 4. Primary metric curve (only validation, since train doesn't compute these metrics)
    ax4 = plt.subplot(2, 3, 4)
    if config.loss.loss_type in ["pairwise_rank", "rank", "ranking"]:
        metric_name = config.training.val_metric if config.training.val_metric in ["spearman", "pairwise_acc"] else "spearman"
        val_metrics = [h.get(metric_name, 0.0) for h in val_history]
        metric_label = "Spearman" if metric_name == "spearman" else "Pairwise Acc"
    else:
        metric_name = config.training.val_metric if config.training.val_metric in ["r2", "rmse", "mae"] else "r2"
        val_metrics = [h.get(metric_name, 0.0) for h in val_history]
        metric_label = metric_name.upper()
    
    # Plot validation metrics
    if val_metrics:
        epochs_metric = range(1, len(val_metrics) + 1)
        ax4.plot(epochs_metric, val_metrics, 'r-', label=f'Val {metric_label}', linewidth=2)
    
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel(metric_label, fontsize=12)
    ax4.set_title(f'Validation {metric_label} Over Time', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='best')
    
    # 5. Score distribution comparison
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(val_targets, bins=20, alpha=0.5, label='Ground Truth', color='blue', edgecolor='black')
    ax5.hist(val_predictions, bins=20, alpha=0.5, label='Predictions', color='red', edgecolor='black')
    ax5.set_xlabel('Value', fontsize=12)
    ax5.set_ylabel('Frequency', fontsize=12)
    ax5.set_title('Validation: Distribution Comparison', fontsize=12)
    ax5.legend(loc='best')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Residual plot (for regression)
    ax6 = plt.subplot(2, 3, 6)
    residuals = val_predictions - val_targets
    ax6.scatter(val_targets, residuals, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax6.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax6.set_xlabel('Ground Truth', fontsize=12)
    ax6.set_ylabel('Residuals (Predicted - Ground Truth)', fontsize=12)
    ax6.set_title('Validation: Residual Plot', fontsize=12)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    report_path = log_dir / "training_report.png"
    plt.savefig(report_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training report saved to: {report_path}")
    
    # Also save individual scatter plot (most important)
    fig2, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(val_targets, val_predictions, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    min_val = min(val_targets.min(), val_predictions.min())
    max_val = max(val_targets.max(), val_predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect prediction')
    
    if len(val_predictions) > 1:
        z = np.polyfit(val_targets, val_predictions, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min_val, max_val, 100)
        ax.plot(x_trend, p(x_trend), 'b-', alpha=0.7, linewidth=2, label='Linear fit')
    
    if config.loss.loss_type in ["pairwise_rank", "rank", "ranking"]:
        spearman_corr, spearman_p = spearmanr(val_predictions, val_targets)
        title = f'Validation: Predicted vs Ground Truth\n'
        title += f'Spearman ρ = {spearman_corr:.4f}'
        if not np.isnan(spearman_p):
            title += f' (p = {spearman_p:.2e})'
    else:
        mae = np.mean(np.abs(val_predictions - val_targets))
        rmse = np.sqrt(np.mean((val_predictions - val_targets) ** 2))
        ss_res = np.sum((val_targets - val_predictions) ** 2)
        ss_tot = np.sum((val_targets - np.mean(val_targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else float('nan')
        pearson_corr, pearson_p = pearsonr(val_predictions, val_targets) if len(val_predictions) > 1 else (0.0, float('nan'))
        
        title = f'Validation: Predicted vs Ground Truth\n'
        title += f'MAE = {mae:.4f}, RMSE = {rmse:.4f}, R² = {r2:.4f}'
        if not np.isnan(pearson_corr):
            title += f'\nPearson r = {pearson_corr:.4f}'
    
    ax.set_xlabel('Ground Truth', fontsize=12)
    ax.set_ylabel('Predicted', fontsize=12)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_aspect('auto')
    
    scatter_path = log_dir / "validation_scatter.png"
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Validation scatter plot saved to: {scatter_path}")


def create_synthetic_dataset(num_samples: int = 20, save_path: str = "data/synthetic_manifest.csv"):
    """Create a synthetic dataset for testing (random graphs)."""
    import pandas as pd
    from pathlib import Path
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Create synthetic manifest
    data = {
        "pdb_path": [f"synthetic_{i}.pdb" for i in range(num_samples)],
        "y": np.random.uniform(0, 10, num_samples).tolist(),
        "antibody_chains": ["H,L"] * num_samples,
        "antigen_chains": ["A"] * num_samples
    }
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"Created synthetic manifest: {save_path}")
    print("Note: You'll need to create actual PDB files or modify dataset to generate random graphs.")


def main():
    parser = argparse.ArgumentParser(description="Train GAT ranker model")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML file")
    parser.add_argument("--manifest", type=str, default=None, help="Path to manifest CSV")
    parser.add_argument("--pdb-dir", type=str, default=None, 
                       help="Base directory for PDB files (optional if manifest uses absolute paths)")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic dataset for testing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--overfit-n", type=int, default=0, 
                       help="Overfit sanity test: train only on first N samples (no val split)")
    parser.add_argument("--epoch", type=int, default=None,
                       help="Number of training epochs (overrides config)")
    parser.add_argument("--loss", type=str, default=None,
                       help="Loss function type: 'pairwise_rank', 'mse', 'l1', 'smooth_l1' (overrides config)")
    parser.add_argument("--graph-cache-dir", type=str, default=None,
                       help="Directory to cache preprocessed graphs (overrides config)")
    parser.add_argument("--rebuild-cache", action="store_true",
                       help="Ignore existing cache and rebuild all graphs")
    parser.add_argument("--hash-pdb-contents", action="store_true",
                       help="Hash PDB file contents for cache key (slower but more robust)")
    parser.add_argument("--save-all-epochs", action="store_true",
                       help="Save model checkpoint at the end of each epoch (in addition to best model)")
    parser.add_argument("--runs-dir", type=str, default="runs",
                       help="Directory to save run artifacts (default: runs)")
    parser.add_argument("--run-id", type=str, default=None,
                       help="Run ID (default: auto-generated timestamp)")
    parser.add_argument("--preload-workers", type=int, default=None,
                       help="Number of workers for parallel graph preloading (default: auto, 0=sequential)")
    parser.add_argument("--skip-health-check", action="store_true",
                       help="Skip graph health check after loading (faster but less safe)")
    parser.add_argument("--save-unhealthy-dir", type=str, default=None,
                       help="Directory to save unhealthy graphs lists by issue type (default: runs/<run_id>/unhealthy_graphs/)")
    args = parser.parse_args()
    
    '''For Debugging
    args.manifest = 'manifest.csv'
    args.loss = 'mse'
    '''
    # Set seed
    set_seed(args.seed)
    
    # Load config
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()
    
    # Override config from command line
    if args.manifest:
        config.data.manifest_csv = args.manifest
    if args.pdb_dir:
        config.data.pdb_dir = args.pdb_dir
    if args.epoch is not None:
        config.training.num_epochs = args.epoch
        print(f"Overriding num_epochs to {args.epoch}")
    if args.loss is not None:
        config.loss.loss_type = args.loss.lower()
        print(f"Overriding loss_type to '{args.loss}' (was '{getattr(config.loss, 'loss_type', 'pairwise_rank')}')")
    if args.graph_cache_dir is not None:
        config.data.graph_cache_dir = args.graph_cache_dir
        print(f"Overriding graph_cache_dir to {args.graph_cache_dir}")
    if args.rebuild_cache:
        config.data.rebuild_cache = True
        print("Rebuilding cache (ignoring existing cache files)")
    if args.hash_pdb_contents:
        config.data.hash_pdb_contents = True
        print("Using PDB file content hashing for cache keys")
    if args.save_all_epochs:
        config.training.save_all_epochs = True
        print("Will save model checkpoint at the end of each epoch")
    if args.preload_workers is not None:
        config.training.preload_workers = args.preload_workers
        worker_desc = f"{args.preload_workers} workers" if args.preload_workers > 0 else "sequential"
        print(f"Overriding preload_workers to {worker_desc}")

    # Setup device
    device = get_device(config.training.device)
    print(f"Using device: {device}")
    
    # Create run directory for logging
    runs_dir = Path(args.runs_dir)
    run_dir = create_run_dir(runs_dir, args.run_id)
    print(f"Run directory: {run_dir}")
    
    # Save config to run directory
    save_config(run_dir, config, format="yaml")
    print(f"Config saved to: {run_dir / 'config.yaml'}")
    
    # Save cache link
    save_cache_link(run_dir, Path(config.data.graph_cache_dir))
    
    # Setup logging - use run_dir for all outputs
    logger = setup_logging(run_dir)
    
    # Create synthetic dataset if requested
    if args.synthetic:
        synthetic_manifest = "data/synthetic_manifest.csv"
        create_synthetic_dataset(num_samples=20, save_path=synthetic_manifest)
        config.data.manifest_csv = synthetic_manifest
        print("WARNING: Synthetic dataset created. You may need to modify dataset.py to handle missing PDB files.")
    
    # Load dataset
    try:
        dataset = AntibodyAntigenDataset(
            manifest_csv=config.data.manifest_csv,
            pdb_dir=config.data.pdb_dir,
            default_antibody_chains=config.data.default_antibody_chains,
            default_antigen_chains=config.data.default_antigen_chains,
            noncovalent_cutoff=config.graph.noncovalent_cutoff,
            interface_cutoff=config.graph.interface_cutoff,
            use_covalent_edges=config.graph.use_covalent_edges,
            use_noncovalent_edges=config.graph.use_noncovalent_edges,
            allow_duplicate_edges=config.graph.allow_duplicate_edges,
            include_residue_index=config.graph.include_residue_index,
            add_interface_features_to_x=config.graph.add_interface_features_to_x,
            transform=IdentityTransform(),  # Can add NormalizeNodeFeatures() if needed
            graph_cache_dir=config.data.graph_cache_dir,
            hash_pdb_contents=config.data.hash_pdb_contents,
            rebuild_cache=config.data.rebuild_cache,
            cache_stats=config.data.cache_stats
        )
        print(f"Loaded dataset with {len(dataset)} samples")
        
        # Preload all graphs into cache to avoid slow loading during training
        print("Preloading graphs into cache (this may take a while for the first time)...")
        preload_workers = getattr(config.training, 'preload_workers', None)
        dataset.preload_all(verbose=True, num_workers=preload_workers)

        # Check graph health after loading (unless skipped)
        if not args.skip_health_check:
            print("\nChecking graph health...")

            # Determine save directory for unhealthy graphs
            save_unhealthy_dir = None
            if args.save_unhealthy_dir:
                save_unhealthy_dir = args.save_unhealthy_dir
            elif run_dir:  # Use run directory if available
                save_unhealthy_dir = run_dir / "unhealthy_graphs"

            health_report = dataset.check_graph_health(verbose=True, save_unhealthy_dir=save_unhealthy_dir)

            # Warn if there are serious issues
            if health_report["issues_found"]:
                serious_issues = [
                    "no_nodes", "no_edges", "no_interface_nodes",
                    "missing_is_interface", "missing_chain_role", "missing_edge_type"
                ]
                serious_count = sum(len(health_report["issues"][issue]) for issue in serious_issues)

                if serious_count > 0:
                    print(f"\n⚠️  WARNING: Found {serious_count} graphs with serious issues!")
                    print("These graphs may cause training problems. Consider:")
                    print("  - Checking your PDB files for missing chains")
                    print("  - Verifying antibody/antigen chain assignments")
                    print("  - Using --rebuild-cache to regenerate problematic graphs")
                    print("  - Filtering out problematic samples from your manifest")

                    if save_unhealthy_dir:
                        print(f"  - Review unhealthy graphs lists: {save_unhealthy_dir}")

                    # Ask user if they want to continue
                    response = input("\nContinue training anyway? (y/N): ").strip().lower()
                    if response not in ['y', 'yes']:
                        print("Training aborted.")
                        return
        else:
            print("\nSkipping graph health check (--skip-health-check)")
            health_report = None
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Hint: Use --synthetic flag to create a test dataset")
        return
    
    # Get sample to determine feature dimensions
    sample = dataset[0]
    node_feature_dim = sample.x.shape[1]
    # Get num_edge_types from edge_type if available, else from edge_attr
    if hasattr(sample, 'edge_type') and sample.edge_type is not None:
        num_edge_types = int(sample.edge_type.max().item()) + 1
    elif sample.edge_attr is not None and sample.edge_attr.shape[1] > 0:
        # Backward compatibility: try to extract from edge_attr
        num_edge_types = int(sample.edge_attr[:, 0].max().item()) + 1 if sample.edge_attr.shape[1] > 0 else 2
    else:
        num_edge_types = 2
        num_edge_types = max(num_edge_types, 3)
    
    # Update config
    config.update_model_dims(node_feature_dim, num_edge_types)
    
    # Overfit sanity test mode
    if args.overfit_n > 0:
        print(f"\n{'='*60}")
        print(f"OVERFIT SANITY TEST MODE: Training on first {args.overfit_n} samples")
        print(f"{'='*60}")
        config.training.overfit_n = args.overfit_n
        config.training.num_epochs = 500  # More epochs for overfit test
        config.training.learning_rate = 5e-3  # Higher LR for overfit
        config.training.log_interval = 10  # Log every 10 epochs
        config.training.enable_debug_logs = True
        config.training.grad_clip = 1.0
        config.model.dropout = 0.0  # Disable dropout for overfit test
        config.training.val_split = 0.0  # No validation split
        config.training.batch_size = min(args.overfit_n, 32)  # Smaller batch for overfit
        
        # Use only first N samples
        overfit_indices = list(range(min(args.overfit_n, len(dataset))))
        train_dataset = torch.utils.data.Subset(dataset, overfit_indices)
        val_dataset = train_dataset  # Evaluate on same set
        print(f"Using {len(train_dataset)} samples for overfit test")
    else:
        # Normal train/val split
        dataset_size = len(dataset)
        val_size = int(dataset_size * config.training.val_split)
        train_size = dataset_size - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed)
        )
    
    # Create data loaders
    # Use persistent_workers to keep workers alive between epochs (faster)
    # Pin memory for faster GPU transfer
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        persistent_workers=config.training.num_workers > 0,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        persistent_workers=config.training.num_workers > 0,
        pin_memory=torch.cuda.is_available()
    )
    
    # Create model
    model = GATRanker(
        node_feature_dim=config.model.node_feature_dim,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout,
        use_edge_types=config.model.use_edge_types,
        num_edge_types=config.model.num_edge_types
    ).to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Create loss function based on config
    loss_type = getattr(config.loss, 'loss_type', 'pairwise_rank')
    
    if loss_type in ["pairwise_rank", "rank", "ranking"]:
        # Pairwise ranking loss
        criterion = get_loss_function(
            "pairwise_rank",
            margin_eps=config.loss.margin_eps,
            tie_eps=getattr(config.loss, 'tie_eps', 1e-6),
            weight_by_diff=config.loss.weight_by_diff,
            reduction=config.loss.reduction,
            temperature=getattr(config.loss, 'temperature', 1.0)
        )
    elif loss_type == "mse":
        criterion = get_loss_function("mse", reduction=config.loss.reduction)
    elif loss_type == "l1":
        criterion = get_loss_function("l1", reduction=config.loss.reduction)
    elif loss_type in ["smooth_l1", "huber"]:
        criterion = get_loss_function(
            "smooth_l1",
            reduction=config.loss.reduction,
            beta=getattr(config.loss, 'beta', 1.0)
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    print(f"Using loss function: {loss_type}")
    if loss_type != "pairwise_rank":
        print(f"  Note: Using regression metrics (MAE, RMSE, R²) for evaluation.")
    else:
        print(f"  Note: Using ranking metrics (Spearman, pairwise accuracy) for evaluation.")
    
    # Initialize metrics CSV files
    train_metrics_csv = init_metrics_csv(run_dir, "train", loss_type)
    val_metrics_csv = init_metrics_csv(run_dir, "val", loss_type) if config.training.val_split > 0 else None
    print(f"Metrics will be saved to: {train_metrics_csv}")
    if val_metrics_csv:
        print(f"Validation metrics will be saved to: {val_metrics_csv}")
    
    # Create optimizer
    if config.training.optimizer.lower() == "adamw":
        optimizer = AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
    else:
        optimizer = Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
    
    # Create scheduler
    scheduler = None
    if config.training.scheduler == "cosine":
        scheduler_params = config.training.scheduler_params.get("cosine", {}).copy()
        # T_max defaults to num_epochs if not specified
        if "T_max" not in scheduler_params:
            scheduler_params["T_max"] = config.training.num_epochs
        scheduler = CosineAnnealingLR(
            optimizer,
            **scheduler_params
        )
    elif config.training.scheduler == "step":
        scheduler = StepLR(
            optimizer,
            **config.training.scheduler_params.get("step", {})
        )
    
    # Training loop
    best_val_metric = -np.inf
    train_history = []
    val_history = []
    
    # Determine the metric name for tracking (will be updated during training)
    best_metric_name = None
    
    print("\nStarting training...")
    for epoch in range(1, config.training.num_epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device,
            grad_clip=config.training.grad_clip,
            enable_debug=config.training.enable_debug_logs
        )
        train_history.append(train_metrics)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Append training metrics to CSV
        append_metrics(
            train_metrics_csv,
            epoch,
            "train",
            train_metrics,
            loss_type,
            lr=current_lr
        )
        
        # Validate
        val_metrics = evaluate(
            model, val_loader, criterion, device,
            margin_eps=config.loss.margin_eps,
            loss_type=config.loss.loss_type
        )
        val_history.append(val_metrics)
        
        # Append validation metrics to CSV
        if val_metrics_csv:
            append_metrics(
                val_metrics_csv,
                epoch,
                "val",
                val_metrics,
                loss_type,
                lr=current_lr
            )
        
        # Update scheduler
        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
        
        # Log
        if epoch % config.training.log_interval == 0 or config.training.overfit_n > 0:
            # Select primary metric based on loss type
            if config.loss.loss_type in ["pairwise_rank", "rank", "ranking"]:
                val_metric = val_metrics.get(config.training.val_metric, val_metrics.get("spearman", 0.0))
            else:
                # For regression losses, use R² or RMSE as primary metric
                val_metric = val_metrics.get("r2", val_metrics.get("rmse", 0.0))
            
            # Detailed logging for overfit mode or debug mode
            if config.training.overfit_n > 0 or config.training.enable_debug_logs:
                log_str = (
                    f"Epoch {epoch:3d} | "
                    f"Loss: {train_metrics['loss']:.4f} | "
                )
                
                if config.loss.loss_type in ["pairwise_rank", "rank", "ranking"]:
                    log_str += (
                        f"Pairwise Acc: {val_metrics.get('pairwise_acc', 0.0):.4f} | "
                        f"Spearman: {val_metrics.get('spearman', 0.0):.4f} | "
                    )
                else:
                    log_str += (
                        f"MAE: {val_metrics.get('mae', 0.0):.4f} | "
                        f"RMSE: {val_metrics.get('rmse', 0.0):.4f} | "
                        f"R²: {val_metrics.get('r2', 0.0):.4f} | "
                    )
                
                log_str += (
                    f"Score: μ={train_metrics['score_mean']:.3f}, σ={train_metrics['score_std']:.3f} | "
                    f"Target: μ={train_metrics['target_mean']:.3f}, σ={train_metrics['target_std']:.3f}"
                )
                if 'grad_norm_mean' in train_metrics:
                    log_str += f" | Grad: {train_metrics['grad_norm_mean']:.3f}"
                print(log_str)
            else:
                # Show loss type in output for clarity
                loss_type_str = f"[{config.loss.loss_type.upper()}]" if config.loss.loss_type != "pairwise_rank" else ""
                
                if config.loss.loss_type in ["pairwise_rank", "rank", "ranking"]:
                    # Ranking metrics
                    print(
                        f"Epoch {epoch:3d} | "
                        f"Train Loss: {train_metrics['loss']:.4f} {loss_type_str} | "
                        f"Val Loss: {val_metrics['loss']:.4f} | "
                        f"Val Spearman: {val_metrics.get('spearman', 0.0):.4f} | "
                        f"Val Pairwise Acc: {val_metrics.get('pairwise_acc', 0.0):.4f}"
                    )
                else:
                    # Regression metrics
                    print(
                        f"Epoch {epoch:3d} | "
                        f"Train Loss: {train_metrics['loss']:.4f} {loss_type_str} | "
                        f"Val Loss: {val_metrics['loss']:.4f} | "
                        f"Val MAE: {val_metrics.get('mae', 0.0):.4f} | "
                        f"Val RMSE: {val_metrics.get('rmse', 0.0):.4f} | "
                        f"Val R²: {val_metrics.get('r2', 0.0):.4f}"
                    )
        
        # Save best model
        if config.training.save_best:
            # Select primary metric based on loss type
            if config.loss.loss_type in ["pairwise_rank", "rank", "ranking"]:
                # For ranking loss, use spearman or pairwise_acc
                if config.training.val_metric == "spearman":
                    val_metric = val_metrics.get("spearman", 0.0)
                    metric_name = "spearman"
                elif config.training.val_metric == "pairwise_acc":
                    val_metric = val_metrics.get("pairwise_acc", 0.0)
                    metric_name = "pairwise_acc"
                else:
                    val_metric = val_metrics.get("spearman", val_metrics.get("pairwise_acc", 0.0))
                    metric_name = "spearman" if "spearman" in val_metrics else "pairwise_acc"
            else:
                # For regression losses, use R² (higher is better) or RMSE/MAE (lower is better, so negate)
                # Default to R² if val_metric is not appropriate for regression
                if config.training.val_metric == "r2":
                    val_metric = val_metrics.get("r2", -np.inf)
                    metric_name = "r2"
                elif config.training.val_metric == "rmse":
                    # Negate RMSE so higher is better (for consistency with other metrics)
                    val_metric = -val_metrics.get("rmse", np.inf)
                    metric_name = "rmse"
                elif config.training.val_metric == "mae":
                    # Negate MAE so higher is better
                    val_metric = -val_metrics.get("mae", np.inf)
                    metric_name = "mae"
                else:
                    # Default to R² for regression if val_metric is not regression-appropriate
                    val_metric = val_metrics.get("r2", -np.inf)
                    metric_name = "r2"
            
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                best_metric_name = metric_name  # Store the metric name
                checkpoint_path = run_dir / "best_model.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metric": val_metric,
                    "config": config
                }, checkpoint_path)
                # Show the actual metric value (not negated for RMSE/MAE)
                if config.loss.loss_type not in ["pairwise_rank", "rank", "ranking"]:
                    if metric_name in ["rmse", "mae"]:
                        display_metric = -val_metric
                    else:
                        display_metric = val_metric
                    print(f"Saved best model (val {metric_name}: {display_metric:.4f})")
                else:
                    print(f"Saved best model (val {metric_name}: {val_metric:.4f})")
        
        # Save model checkpoint at end of each epoch if requested
        if config.training.save_all_epochs:
            epoch_checkpoints_dir = run_dir / "epoch_checkpoints"
            epoch_checkpoints_dir.mkdir(parents=True, exist_ok=True)
            epoch_checkpoint_path = epoch_checkpoints_dir / f"epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "config": config
            }, epoch_checkpoint_path)
            if epoch % config.training.log_interval == 0 or config.training.overfit_n > 0:
                print(f"Saved epoch {epoch} checkpoint to: {epoch_checkpoint_path}")
    
    # Save training history
    with open(run_dir / "history.json", "w") as f:
        json.dump({
            "train": train_history,
            "val": val_history
        }, f, indent=2)
    
    # Use the stored metric name, or determine from loss type
    if best_metric_name is None:
        if config.loss.loss_type in ["pairwise_rank", "rank", "ranking"]:
            best_metric_name = config.training.val_metric if config.training.val_metric in ["spearman", "pairwise_acc"] else "spearman"
        else:
            best_metric_name = config.training.val_metric if config.training.val_metric in ["r2", "rmse", "mae"] else "r2"
    
    # Show the actual metric value (not negated for RMSE/MAE)
    if config.loss.loss_type not in ["pairwise_rank", "rank", "ranking"]:
        if best_metric_name in ["rmse", "mae"]:
            display_metric = -best_val_metric
        else:
            display_metric = best_val_metric
        print(f"\nTraining complete! Best val {best_metric_name}: {display_metric:.4f}")
    else:
        print(f"\nTraining complete! Best val {best_metric_name}: {best_val_metric:.4f}")
    print(f"Checkpoints saved to: {run_dir}")
    
    # Generate graphic report
    create_training_report(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_history=train_history,
        val_history=val_history,
        config=config,
        log_dir=run_dir,
        device=device
    )


if __name__ == "__main__":
    main()

