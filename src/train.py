"""
Training script for GAT ranker model.
"""
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path
import argparse
from tqdm import tqdm
import json

from config import Config
from data.dataset import AntibodyAntigenDataset
from data.transforms import IdentityTransform, NormalizeNodeFeatures
from models.gat_ranker import GATRanker
from losses.pairwise_rank_loss import PairwiseRankingLoss, compute_pairwise_accuracy
from losses import get_loss_function
from utils import set_seed, get_device, setup_logging, count_parameters


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
    args = parser.parse_args()
    
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

    # test
    # config.data.manifest_csv = 'manifest.csv'
    
    # Setup device
    device = get_device(config.training.device)
    print(f"Using device: {device}")
    
    # Setup logging
    log_dir = Path(config.training.save_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_dir)
    
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
            bound_cutoff=config.graph.bound_cutoff,
            unbound_cutoff=config.graph.unbound_cutoff,
            use_sequential_edges=config.graph.use_sequential_edges,
            include_residue_index=config.graph.include_residue_index,
            transform=IdentityTransform(),  # Can add NormalizeNodeFeatures() if needed
            graph_cache_dir=config.data.graph_cache_dir,
            hash_pdb_contents=config.data.hash_pdb_contents,
            rebuild_cache=config.data.rebuild_cache,
            cache_stats=config.data.cache_stats
        )
        print(f"Loaded dataset with {len(dataset)} samples")
        
        # Preload all graphs into cache to avoid slow loading during training
        print("Preloading graphs into cache (this may take a while for the first time)...")
        dataset.preload_all(verbose=True)
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Hint: Use --synthetic flag to create a test dataset")
        return
    
    # Get sample to determine feature dimensions
    sample = dataset[0]
    node_feature_dim = sample.x.shape[1]
    num_edge_types = int(sample.edge_attr[:, 0].max().item()) + 1 if sample.edge_attr is not None else 2
    if config.graph.use_sequential_edges:
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
        
        # Validate
        val_metrics = evaluate(
            model, val_loader, criterion, device,
            margin_eps=config.loss.margin_eps,
            loss_type=config.loss.loss_type
        )
        val_history.append(val_metrics)
        
        # Update scheduler
        if scheduler:
            scheduler.step()
        
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
                checkpoint_path = log_dir / "best_model.pt"
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
    
    # Save training history
    with open(log_dir / "history.json", "w") as f:
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
    print(f"Checkpoints saved to: {log_dir}")


if __name__ == "__main__":
    main()

