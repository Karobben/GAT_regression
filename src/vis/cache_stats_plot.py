"""
Plot cache statistics for graph conversion.
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional


def plot_cache_stats(
    stats_csv: str,
    out_dir: Optional[str] = None
):
    """
    Plot histograms of cache statistics.
    
    Args:
        stats_csv: Path to CSV file with cache statistics
        out_dir: Output directory for plots (default: same as CSV directory)
    """
    stats_path = Path(stats_csv)
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats CSV not found: {stats_csv}")
    
    df = pd.read_csv(stats_csv)
    
    if out_dir is None:
        out_dir = stats_path.parent
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Number of nodes histogram
    ax = axes[0, 0]
    ax.hist(df['num_nodes'], bins=30, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Number of Nodes', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Node Counts', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 2. Number of edges histogram
    ax = axes[0, 1]
    ax.hist(df['num_edges'], bins=30, edgecolor='black', alpha=0.7, color='orange')
    ax.set_xlabel('Number of Edges', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Edge Counts', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 3. Fraction of bound edges
    if 'edge_type_counts' in df.columns:
        # Parse edge_type_counts (assuming it's a dict string or JSON)
        import json
        import ast
        
        bound_fractions = []
        for idx, row in df.iterrows():
            try:
                if pd.notna(row['edge_type_counts']):
                    if isinstance(row['edge_type_counts'], str):
                        counts = ast.literal_eval(row['edge_type_counts'])
                    else:
                        counts = row['edge_type_counts']
                    
                    total_edges = row['num_edges']
                    bound_count = counts.get(1, 0)  # Type 1 = bound
                    if total_edges > 0:
                        bound_fractions.append(bound_count / total_edges)
            except:
                continue
        
        if bound_fractions:
            ax = axes[1, 0]
            ax.hist(bound_fractions, bins=30, edgecolor='black', alpha=0.7, color='green')
            ax.set_xlabel('Fraction of Bound Edges', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title('Distribution of Bound Edge Fractions', fontsize=12)
            ax.grid(True, alpha=0.3)
    
    # 4. Build times
    if 'build_time_sec' in df.columns:
        ax = axes[1, 1]
        ax.hist(df['build_time_sec'], bins=30, edgecolor='black', alpha=0.7, color='red')
        ax.set_xlabel('Build Time (seconds)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Graph Build Times', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    out_path = out_dir / "cache_stats_plots.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved cache statistics plots to: {out_path}")
    plt.close()

