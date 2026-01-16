"""
Evaluation script for GAT ranker model.
Rewritten to match train.py exactly for consistent results.
"""
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import numpy as np
from scipy.stats import spearmanr, pearsonr
from pathlib import Path
import argparse
import json

# Try to import matplotlib (optional)
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Scatter plot will not be generated.")

from src.config import Config
from src.data.dataset import AntibodyAntigenDataset
from src.data.transforms import IdentityTransform
from src.models.gat_ranker import GATRanker
from src.losses.pairwise_rank_loss import compute_pairwise_accuracy
from src.losses import get_loss_function
from src.utils import get_device, set_seed, count_parameters
from src.utils.run_logging import save_eval_predictions


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    margin_eps: float = 0.0,
    loss_type: str = "pairwise_rank"
) -> dict:
    """
    Evaluate model on validation/test set.
    This is the EXACT same function as in train.py.
    """
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
            pearson_corr, pearson_p = pearsonr(all_scores, all_targets)
            metrics["pearson"] = float(pearson_corr)
            metrics["pearson_p"] = float(pearson_p) if not np.isnan(pearson_p) else float('nan')
        else:
            metrics["pearson"] = 0.0
            metrics["pearson_p"] = float('nan')
    
    return metrics


def create_scatter_plot(
    scores: np.ndarray,
    targets: np.ndarray,
    metrics: dict,
    loss_type: str,
    output_path: Path
):
    """Create a scatter plot showing predicted vs ground truth."""
    if not HAS_MATPLOTLIB:
        print("Skipping scatter plot generation (matplotlib not available)")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create scatter plot
    ax.scatter(targets, scores, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # Get independent ranges for x and y axes
    x_min, x_max = targets.min(), targets.max()
    y_min, y_max = scores.min(), scores.max()
    
    # Add some padding (5% on each side)
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_padding = x_range * 0.05 if x_range > 0 else 0.1
    y_padding = y_range * 0.05 if y_range > 0 else 0.1
    
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    
    # Add diagonal line (perfect prediction)
    diag_min = min(x_min - x_padding, y_min - y_padding)
    diag_max = max(x_max + x_padding, y_max + y_padding)
    ax.plot([diag_min, diag_max], [diag_min, diag_max], 'r--', alpha=0.5, label='Perfect prediction')
    
    # Add trend line (linear fit)
    if len(scores) > 1:
        z = np.polyfit(targets, scores, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(x_min - x_padding, x_max + x_padding, 100)
        ax.plot(x_trend, p(x_trend), 'b-', alpha=0.7, linewidth=2, label='Linear fit')
    
    # Labels and title
    ax.set_xlabel('Ground-truth Target Values', fontsize=12)
    ax.set_ylabel('Predicted Scores', fontsize=12)
    
    # Add metrics to title based on loss type
    if loss_type in ["pairwise_rank", "rank", "ranking"]:
        title = f'Predicted vs. Ground-truth\n'
        title += f'Spearman ρ = {metrics.get("spearman", 0.0):.4f}'
        if not np.isnan(metrics.get("spearman_p", float('nan'))):
            title += f' (p = {metrics["spearman_p"]:.2e})'
        title += f', Pairwise Acc = {metrics.get("pairwise_acc", 0.0):.4f}'
    else:
        title = f'Predicted vs. Ground-truth ({loss_type.upper()})\n'
        title += f'MAE = {metrics.get("mae", 0.0):.4f}, RMSE = {metrics.get("rmse", 0.0):.4f}, R² = {metrics.get("r2", 0.0):.4f}'
        if not np.isnan(metrics.get("pearson_p", float('nan'))):
            title += f'\nPearson r = {metrics.get("pearson", 0.0):.4f} (p = {metrics["pearson_p"]:.2e})'
    
    ax.set_title(title, fontsize=11)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_aspect('auto')
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Scatter plot created: {output_path}")


def evaluate_model(
    model_path: str,
    manifest_csv: str,
    pdb_dir: str = None,
    config_path: str = None,
    batch_size: int = 16,
    device: str = "auto",
    graph_cache_dir: str = None,
    rebuild_cache: bool = False,
    seed: int = 42
):
    """
    Evaluate trained model on a dataset.
    This function follows the EXACT same logic as train.py for consistency.
    """
    # Set seed for reproducibility
    set_seed(seed)
    
    # Setup device (same as train.py)
    device = get_device(device)
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading checkpoint from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Load config (same priority as train.py)
    if "config" in checkpoint:
        config = checkpoint["config"]
        print("Loaded config from checkpoint")
    elif config_path:
        config = Config.from_yaml(config_path)
        print(f"Loaded config from: {config_path}")
    else:
        print("Warning: No config found. Using defaults.")
        config = Config()
    
    # Override cache settings from command line (same as train.py)
    if graph_cache_dir is not None:
        config.data.graph_cache_dir = graph_cache_dir
        print(f"Overriding graph_cache_dir to {graph_cache_dir}")
    if rebuild_cache:
        config.data.rebuild_cache = True
        print("Rebuilding cache (ignoring existing cache files)")
    
    # Load dataset (EXACT same as train.py)
    print(f"Loading dataset from: {manifest_csv}")
    try:
        dataset = AntibodyAntigenDataset(
            manifest_csv=manifest_csv,
            pdb_dir=pdb_dir,
            default_antibody_chains=config.data.default_antibody_chains,
            default_antigen_chains=config.data.default_antigen_chains,
            noncovalent_cutoff=config.graph.noncovalent_cutoff,
            interface_cutoff=config.graph.interface_cutoff,
            use_covalent_edges=config.graph.use_covalent_edges,
            use_noncovalent_edges=config.graph.use_noncovalent_edges,
            allow_duplicate_edges=config.graph.allow_duplicate_edges,
            include_residue_index=config.graph.include_residue_index,
            add_interface_features_to_x=config.graph.add_interface_features_to_x,
            transform=IdentityTransform(),  # Same as train.py
            graph_cache_dir=config.data.graph_cache_dir,
            hash_pdb_contents=config.data.hash_pdb_contents,
            rebuild_cache=config.data.rebuild_cache,
            cache_stats=config.data.cache_stats
        )
        print(f"Loaded dataset with {len(dataset)} samples")
        
        # Preload all graphs into cache (same as train.py)
        print("Preloading graphs into cache (this may take a while for the first time)...")
        # Use multiprocessing if available
        num_workers = getattr(config.training, 'preload_workers', None)  # Default: auto-detect
        dataset.preload_all(verbose=True, num_workers=num_workers)
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    
    # Get sample to determine feature dimensions (same as train.py)
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
    
    # Update config (same as train.py)
    config.update_model_dims(node_feature_dim, num_edge_types)
    print(f"Model dimensions: node_feature_dim={config.model.node_feature_dim}, num_edge_types={config.model.num_edge_types}")
    
    # Create model (EXACT same as train.py)
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
    
    # Load model weights (same compatibility handling as before)
    state_dict = checkpoint["model_state_dict"]
    
    # # Handle compatibility: old GATConv used lin_src/lin_dst, new uses lin
    # new_state_dict = {}
    # for key, value in state_dict.items():
    #     if "lin_src" in key:
    #         new_key = key.replace("lin_src", "lin")
    #         new_state_dict[new_key] = value
    #     elif "lin_dst" in key:
    #         # Skip lin_dst (we use lin_src which we already mapped)
    #         continue
    #     else:
    #         new_state_dict[key] = value
    new_state_dict = state_dict  # Assume compatible for simplicity
    
    # Load state dict
    print("Loading model state dict...")
    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("Model state dict loaded successfully (strict mode)")
    except RuntimeError as e:
        print(f"Warning: Strict loading failed: {e}")
        print("Attempting non-strict loading...")
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        if missing_keys:
            print(f"Missing keys (using defaults): {missing_keys}")
            print("WARNING: Model may not work correctly with missing keys!")
        if unexpected_keys:
            print(f"Unexpected keys (ignored): {unexpected_keys}")
    
    # Ensure model is in eval mode
    model.eval()
    print(f"Model is in {'training' if model.training else 'eval'} mode")
    
    # Create loss function (EXACT same as train.py)
    loss_type = getattr(config.loss, 'loss_type', 'pairwise_rank')
    
    if loss_type in ["pairwise_rank", "rank", "ranking"]:
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
    
    # Create data loader (same settings as train.py validation loader)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffling for evaluation
        num_workers=0,  # Use 0 for determinism
        persistent_workers=False,
        pin_memory=torch.cuda.is_available()
    )
    
    # Evaluate using the EXACT same function as train.py
    print(f"Evaluating model on {len(dataset)} samples with batch_size={batch_size}...")
    metrics = evaluate(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        margin_eps=config.loss.margin_eps,
        loss_type=loss_type
    )
    
    # Print results
    print("\n" + "="*50)
    if loss_type in ["pairwise_rank", "rank", "ranking"]:
        print("Evaluation Results (Ranking Metrics)")
        print("="*50)
        print(f"Number of samples: {len(dataset)}")
        if not np.isnan(metrics.get("spearman_p", float('nan'))):
            print(f"Spearman correlation: {metrics.get('spearman', 0.0):.4f} (p={metrics['spearman_p']:.4e})")
        else:
            print(f"Spearman correlation: {metrics.get('spearman', 0.0):.4f} (p-value not available)")
        print(f"Pairwise accuracy: {metrics.get('pairwise_acc', 0.0):.4f}")
    else:
        print(f"Evaluation Results (Regression Metrics - {loss_type.upper()})")
        print("="*50)
        print(f"Number of samples: {len(dataset)}")
        print(f"Mean Absolute Error (MAE): {metrics.get('mae', 0.0):.4f}")
        print(f"Root Mean Squared Error (RMSE): {metrics.get('rmse', 0.0):.4f}")
        r2 = metrics.get('r2', 0.0)
        if not np.isnan(r2):
            print(f"R² (coefficient of determination): {r2:.4f}")
        else:
            print(f"R²: {r2}")
        if not np.isnan(metrics.get("pearson_p", float('nan'))):
            print(f"Pearson correlation: {metrics.get('pearson', 0.0):.4f} (p={metrics['pearson_p']:.4e})")
        else:
            print(f"Pearson correlation: {metrics.get('pearson', 0.0):.4f} (p-value not available)")
    
    print(f"Loss: {metrics.get('loss', 0.0):.4f}")
    print(f"Score range: [{metrics.get('score_min', 0.0):.4f}, {metrics.get('score_max', 0.0):.4f}]")
    print(f"Target range: [{metrics.get('target_mean', 0.0):.4f} ± {metrics.get('target_std', 0.0):.4f}]")
    print("="*50)
    
    # Get predictions and targets for saving
    all_scores = []
    all_targets = []
    sample_ids = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = batch.to(device)
            scores = model.forward_batch(batch)
            
            if batch.y.dim() > 1 and batch.y.shape[-1] == 1:
                targets = batch.y.squeeze(-1)
            elif batch.y.dim() == 1:
                targets = batch.y
            else:
                targets = batch.y.view(-1)
            
            all_scores.append(scores.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
            # Get sample IDs from dataset (use pdb_path from manifest)
            batch_start = batch_idx * batch_size
            for i in range(len(scores)):
                idx = batch_start + i
                if idx < len(dataset):
                    # Try to get pdb_path from dataset
                    row = dataset.df.iloc[idx]
                    pdb_path = Path(row["pdb_path"])
                    # Use filename or full path as sample ID
                    sample_id = pdb_path.name if pdb_path.name else str(pdb_path)
                    sample_ids.append(sample_id)
                else:
                    sample_ids.append(f"sample_{idx}")
    
    all_scores = np.concatenate(all_scores)
    all_targets = np.concatenate(all_targets)
    
    # Save results
    output_dir = Path(model_path).parent
    results = {
        "num_samples": len(all_scores),
        "loss_type": loss_type,
        **metrics,
        "scores": all_scores.tolist(),
        "targets": all_targets.tolist()
    }
    
    output_path = output_dir / "eval_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    # Save eval_predictions.csv (for webapp)
    eval_csv_path = save_eval_predictions(
        output_dir,
        sample_ids,
        all_targets.tolist(),
        all_scores.tolist()
    )
    print(f"Evaluation predictions saved to: {eval_csv_path}")
    
    # Create scatter plot
    plot_path = output_dir / "eval_scatter_plot.png"
    create_scatter_plot(all_scores, all_targets, metrics, loss_type, plot_path)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate GAT ranker model")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest CSV")
    parser.add_argument("--pdb-dir", type=str, default=None, 
                       help="Base directory for PDB files (optional if manifest uses absolute paths)")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    parser.add_argument("--graph-cache-dir", type=str, default=None,
                       help="Directory to cache preprocessed graphs (overrides config)")
    parser.add_argument("--rebuild-cache", action="store_true",
                       help="Ignore existing cache and rebuild all graphs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    '''
    args.manifest = 'manifest.csv'
    args.model = 'checkpoints/best_model.pt'
    '''
    evaluate_model(
        model_path=args.model,
        manifest_csv=args.manifest,
        pdb_dir=args.pdb_dir,
        config_path=args.config,
        batch_size=args.batch_size,
        device=args.device,
        graph_cache_dir=args.graph_cache_dir,
        rebuild_cache=args.rebuild_cache,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
