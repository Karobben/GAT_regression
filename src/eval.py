"""
Evaluation script for GAT ranker model.
"""
import torch
from torch_geometric.loader import DataLoader
import numpy as np
from scipy.stats import spearmanr
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

from config import Config
from data.dataset import AntibodyAntigenDataset
from data.transforms import IdentityTransform
from models.gat_ranker import GATRanker
from losses.pairwise_rank_loss import compute_pairwise_accuracy
from utils import get_device, set_seed


def create_scatter_plot(
    scores: np.ndarray,
    targets: np.ndarray,
    spearman_corr: float,
    spearman_p: float,
    pairwise_acc: float,
    output_path: Path
):
    """
    Create a scatter plot showing association between predicted scores and targets.
    
    Args:
        scores: Predicted scores
        targets: Ground-truth target values
        spearman_corr: Spearman correlation coefficient
        spearman_p: P-value for Spearman correlation (can be NaN)
        pairwise_acc: Pairwise accuracy
        output_path: Path to save the plot
    """
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
    
    # Add diagonal line (perfect prediction) - use the full range that covers both axes
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
    
    # Add metrics to title
    title = f'Predicted vs. Ground-truth\n'
    title += f'Spearman ρ = {spearman_corr:.4f}'
    if not np.isnan(spearman_p):
        title += f' (p = {spearman_p:.2e})'
    title += f', Pairwise Acc = {pairwise_acc:.4f}'
    ax.set_title(title, fontsize=11)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Use auto aspect ratio (not equal) so axes can have different scales
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
    rebuild_cache: bool = False
):
    """
    Evaluate trained model on a dataset.
    
    Args:
        model_path: Path to saved model checkpoint
        manifest_csv: Path to manifest CSV for evaluation
        pdb_dir: Base directory for PDB files
        config_path: Path to config YAML (optional, will load from checkpoint if available)
        batch_size: Batch size for evaluation
        device: Device to use
    """
    device = get_device(device)
    
    # Load checkpoint
    # weights_only=False needed for PyTorch 2.6+ when checkpoint contains config objects
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Load config
    if "config" in checkpoint:
        config = checkpoint["config"]
    elif config_path:
        config = Config.from_yaml(config_path)
    else:
        print("Warning: No config found. Using defaults.")
        config = Config()
    
    # Override cache settings from command line
    if graph_cache_dir is not None:
        config.data.graph_cache_dir = graph_cache_dir
    if rebuild_cache:
        config.data.rebuild_cache = True
    
    # Load dataset
    dataset = AntibodyAntigenDataset(
        manifest_csv=manifest_csv,
        pdb_dir=pdb_dir,
        default_antibody_chains=config.data.default_antibody_chains,
        default_antigen_chains=config.data.default_antigen_chains,
        bound_cutoff=config.graph.bound_cutoff,
        unbound_cutoff=config.graph.unbound_cutoff,
        use_sequential_edges=config.graph.use_sequential_edges,
        include_residue_index=config.graph.include_residue_index,
        transform=IdentityTransform(),
        graph_cache_dir=config.data.graph_cache_dir,
        hash_pdb_contents=config.data.hash_pdb_contents,
        rebuild_cache=config.data.rebuild_cache,
        cache_stats=config.data.cache_stats
    )
    
    # Preload all graphs (will use cache if available)
    print(f"Loading {len(dataset)} graphs (will use cache if available)...")
    dataset.preload_all(verbose=True)
    dataset.print_cache_stats()
    
    # Get feature dimensions from first sample
    sample = dataset[0]
    node_feature_dim = sample.x.shape[1]
    num_edge_types = int(sample.edge_attr[:, 0].max().item()) + 1 if sample.edge_attr is not None else 2
    if config.graph.use_sequential_edges:
        num_edge_types = max(num_edge_types, 3)
    
    # Create model
    model = GATRanker(
        node_feature_dim=node_feature_dim,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout,
        use_edge_types=config.model.use_edge_types,
        num_edge_types=num_edge_types
    ).to(device)
    
    # Load model weights with compatibility handling for old checkpoint formats
    state_dict = checkpoint["model_state_dict"]
    
    # Handle compatibility: old GATConv used lin_src/lin_dst, new uses lin
    # In newer PyTorch Geometric, GATConv uses a single 'lin' layer
    # We'll map lin_src to lin (lin_dst is typically the same or can be ignored)
    new_state_dict = {}
    lin_dst_keys = {}  # Store lin_dst keys to check if we need to combine
    
    for key, value in state_dict.items():
        if "lin_src" in key:
            # Map lin_src to lin
            new_key = key.replace("lin_src", "lin")
            new_state_dict[new_key] = value
        elif "lin_dst" in key:
            # Store lin_dst - if corresponding lin_src exists, we could average/combine
            # For now, we'll just use lin_src (which we already mapped)
            lin_dst_keys[key] = value
        else:
            new_state_dict[key] = value
    
    # Try loading with mapped state dict
    try:
        model.load_state_dict(new_state_dict, strict=True)
    except RuntimeError as e:
        # If strict loading fails, try non-strict (may have some missing/unexpected keys)
        print(f"Warning: Strict loading failed: {e}")
        print("Attempting non-strict loading...")
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        if missing_keys:
            print(f"Missing keys (using defaults): {missing_keys[:5]}...")  # Show first 5
        if unexpected_keys:
            print(f"Unexpected keys (ignored): {unexpected_keys[:5]}...")  # Show first 5
    
    model.eval()
    
    # Create data loader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Evaluate
    all_scores = []
    all_targets = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            scores = model.forward_batch(batch)
            targets = batch.y.squeeze(-1)
            
            all_scores.append(scores.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Concatenate
    all_scores = np.concatenate(all_scores)
    all_targets = np.concatenate(all_targets)
    
    # Get loss type from config
    loss_type = getattr(config.loss, 'loss_type', 'pairwise_rank')
    
    # Compute metrics based on loss type
    if loss_type in ["pairwise_rank", "rank", "ranking"]:
        # Ranking metrics for ranking loss
        if len(all_scores) < 3:
            spearman_corr = spearmanr(all_scores, all_targets).correlation
            spearman_p = float('nan')
            print(f"Warning: Only {len(all_scores)} samples. P-value cannot be computed (need >= 3 samples).")
        else:
            spearman_corr, spearman_p = spearmanr(all_scores, all_targets)
        
        scores_tensor = torch.tensor(all_scores)
        targets_tensor = torch.tensor(all_targets)
        pairwise_acc = compute_pairwise_accuracy(
            scores_tensor,
            targets_tensor,
            margin_eps=config.loss.margin_eps
        )
        
        # Print results
        print("\n" + "="*50)
        print("Evaluation Results (Ranking Metrics)")
        print("="*50)
        print(f"Number of samples: {len(all_scores)}")
        if not np.isnan(spearman_p):
            print(f"Spearman correlation: {spearman_corr:.4f} (p={spearman_p:.4e})")
        else:
            print(f"Spearman correlation: {spearman_corr:.4f} (p-value not available)")
        print(f"Pairwise accuracy: {pairwise_acc:.4f}")
        print(f"Score range: [{all_scores.min():.4f}, {all_scores.max():.4f}]")
        print(f"Target range: [{all_targets.min():.4f}, {all_targets.max():.4f}]")
        print("="*50)
        
        # Save results
        results = {
            "num_samples": len(all_scores),
            "loss_type": loss_type,
            "spearman_correlation": float(spearman_corr),
            "spearman_p_value": float(spearman_p) if not np.isnan(spearman_p) else None,
            "pairwise_accuracy": float(pairwise_acc),
            "scores": all_scores.tolist(),
            "targets": all_targets.tolist()
        }
    else:
        # Regression metrics for MSE/L1/SmoothL1
        # Mean Absolute Error (MAE)
        mae = np.mean(np.abs(all_scores - all_targets))
        
        # Root Mean Squared Error (RMSE)
        rmse = np.sqrt(np.mean((all_scores - all_targets) ** 2))
        
        # R² (coefficient of determination)
        ss_res = np.sum((all_targets - all_scores) ** 2)
        ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
        if ss_tot > 0:
            r2 = 1 - (ss_res / ss_tot)
        else:
            r2 = float('nan')
        
        # Pearson correlation
        if len(all_scores) > 1:
            from scipy.stats import pearsonr
            pearson_corr, pearson_p = pearsonr(all_scores, all_targets)
        else:
            pearson_corr = 0.0
            pearson_p = float('nan')
        
        # Print results
        print("\n" + "="*50)
        print(f"Evaluation Results (Regression Metrics - {loss_type.upper()})")
        print("="*50)
        print(f"Number of samples: {len(all_scores)}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"R² (coefficient of determination): {r2:.4f}" if not np.isnan(r2) else f"R²: {r2}")
        if not np.isnan(pearson_p):
            print(f"Pearson correlation: {pearson_corr:.4f} (p={pearson_p:.4e})")
        else:
            print(f"Pearson correlation: {pearson_corr:.4f} (p-value not available)")
        print(f"Score range: [{all_scores.min():.4f}, {all_scores.max():.4f}]")
        print(f"Target range: [{all_targets.min():.4f}, {all_targets.max():.4f}]")
        print("="*50)
        
        # Save results
        results = {
            "num_samples": len(all_scores),
            "loss_type": loss_type,
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2) if not np.isnan(r2) else None,
            "pearson_correlation": float(pearson_corr),
            "pearson_p_value": float(pearson_p) if not np.isnan(pearson_p) else None,
            "scores": all_scores.tolist(),
            "targets": all_targets.tolist()
        }
    
    output_dir = Path(model_path).parent
    output_path = output_dir / "eval_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    # Create scatter plot (adapt based on loss type)
    plot_path = output_dir / "eval_scatter_plot.png"
    if loss_type in ["pairwise_rank", "rank", "ranking"]:
        # Use existing ranking plot function
        create_scatter_plot(
            all_scores, 
            all_targets, 
            spearman_corr, 
            spearman_p,
            pairwise_acc,
            plot_path
        )
    else:
        # Create regression-style scatter plot
        if HAS_MATPLOTLIB:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(all_targets, all_scores, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
            
            # Get independent ranges
            x_min, x_max = all_targets.min(), all_targets.max()
            y_min, y_max = all_scores.min(), all_scores.max()
            x_range = x_max - x_min
            y_range = y_max - y_min
            x_padding = x_range * 0.05 if x_range > 0 else 0.1
            y_padding = y_range * 0.05 if y_range > 0 else 0.1
            
            ax.set_xlim(x_min - x_padding, x_max + x_padding)
            ax.set_ylim(y_min - y_padding, y_max + y_padding)
            
            # Add diagonal line
            diag_min = min(x_min - x_padding, y_min - y_padding)
            diag_max = max(x_max + x_padding, y_max + y_padding)
            ax.plot([diag_min, diag_max], [diag_min, diag_max], 'r--', alpha=0.5, label='Perfect prediction')
            
            # Add trend line
            if len(all_scores) > 1:
                z = np.polyfit(all_targets, all_scores, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(x_min - x_padding, x_max + x_padding, 100)
                ax.plot(x_trend, p(x_trend), 'b-', alpha=0.7, linewidth=2, label='Linear fit')
            
            ax.set_xlabel('Ground-truth Target Values', fontsize=12)
            ax.set_ylabel('Predicted Scores', fontsize=12)
            
            # Add metrics to title
            title = f'Predicted vs. Ground-truth ({loss_type.upper()})\n'
            title += f'MAE = {mae:.4f}, RMSE = {rmse:.4f}, R² = {r2:.4f}'
            if not np.isnan(pearson_p):
                title += f'\nPearson r = {pearson_corr:.4f} (p = {pearson_p:.2e})'
            ax.set_title(title, fontsize=11)
            
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')
            ax.set_aspect('auto')
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Scatter plot created: {plot_path}")
        else:
            print("Skipping scatter plot generation (matplotlib not available)")
    
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
    args = parser.parse_args()
    
    set_seed(42)
    evaluate_model(
        model_path=args.model,
        manifest_csv=args.manifest,
        pdb_dir=args.pdb_dir,
        config_path=args.config,
        batch_size=args.batch_size,
        device=args.device,
        graph_cache_dir=args.graph_cache_dir,
        rebuild_cache=args.rebuild_cache
    )


if __name__ == "__main__":
    main()

