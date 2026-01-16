"""
CLI for visualization tools.
"""
import argparse
import torch
from pathlib import Path
from typing import Optional
import pandas as pd

from src.vis.graph_vis import plot_graph_3d
from src.vis.cache_stats_plot import plot_cache_stats
from src.data.cache_utils import load_graph_from_cache
from src.data.pdb_to_graph import pdb_to_graph
from src.config import Config


def visualize_graph_from_cache(
    cache_path: str,
    out_path: str,
    only_bound_edges: bool = False,
    edge_max_dist: Optional[float] = None,
    max_edges: int = 5000,
    color_by: str = "chain_group",
    label_chains: bool = False
):
    """Visualize a graph from cache file."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")
    
    print(f"Loading graph from cache: {cache_path}")
    data, metadata = load_graph_from_cache(cache_path, device=None)
    
    # Verify required fields
    if not hasattr(data, 'pos') or data.pos is None:
        raise ValueError(f"Graph does not have 'pos' field. This cache was created before visualization support.")
    
    print(f"Graph: {data.num_nodes} nodes, {data.edge_index.shape[1]} edges")
    
    plot_graph_3d(
        data=data,
        out_path=out_path,
        max_edges=max_edges,
        show=False,
        color_by=color_by,
        edge_color_by="edge_type",
        subsample_edges=True,
        only_bound_edges=only_bound_edges,
        edge_max_dist=edge_max_dist,
        label_chains=label_chains
    )


def visualize_graph_from_pdb(
    pdb_path: str,
    out_path: str,
    antibody_chains: list = None,
    antigen_chains: list = None,
    config_path: Optional[str] = None,
    only_bound_edges: bool = False,
    edge_max_dist: Optional[float] = None,
    max_edges: int = 5000,
    color_by: str = "chain_group",
    label_chains: bool = False
):
    """Visualize a graph built from PDB file."""
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")
    
    # Load config if provided
    if config_path:
        config = Config.from_yaml(config_path)
        antibody_chains = antibody_chains or config.data.default_antibody_chains
        antigen_chains = antigen_chains or config.data.default_antigen_chains
        bound_cutoff = config.graph.bound_cutoff
        unbound_cutoff = config.graph.unbound_cutoff
        use_sequential_edges = config.graph.use_sequential_edges
    else:
        antibody_chains = antibody_chains or ["H", "L"]
        antigen_chains = antigen_chains or []
        bound_cutoff = 8.0
        unbound_cutoff = 10.0
        use_sequential_edges = False
    
    print(f"Building graph from PDB: {pdb_path}")
    print(f"Antibody chains: {antibody_chains}, Antigen chains: {antigen_chains}")
    
    data = pdb_to_graph(
        pdb_path=str(pdb_path),
        antibody_chains=antibody_chains,
        antigen_chains=antigen_chains,
        bound_cutoff=bound_cutoff,
        unbound_cutoff=unbound_cutoff,
        use_sequential_edges=use_sequential_edges,
        include_residue_index=True,
        y=None
    )
    
    print(f"Graph: {data.num_nodes} nodes, {data.edge_index.shape[1]} edges")
    
    plot_graph_3d(
        data=data,
        out_path=out_path,
        max_edges=max_edges,
        show=False,
        color_by=color_by,
        edge_color_by="edge_type",
        subsample_edges=True,
        only_bound_edges=only_bound_edges,
        edge_max_dist=edge_max_dist,
        label_chains=label_chains
    )


def visualize_graphs_from_csv(
    csv_path: str,
    out_dir: str,
    n: int = 20,
    pdb_dir: Optional[str] = None,
    config_path: Optional[str] = None,
    graph_cache_dir: Optional[str] = None,
    only_bound_edges: bool = False,
    max_edges: int = 5000
):
    """Visualize multiple graphs from a manifest CSV."""
    from src.data.dataset import AntibodyAntigenDataset
    from src.data.transforms import IdentityTransform
    
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Load config
    if config_path:
        config = Config.from_yaml(config_path)
    else:
        config = Config()
    
    if graph_cache_dir:
        config.data.graph_cache_dir = graph_cache_dir
    
    # Load dataset
    dataset = AntibodyAntigenDataset(
        manifest_csv=str(csv_path),
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
        rebuild_cache=False,
        cache_stats=False
    )
    
    # Preload graphs
    print(f"Loading {min(n, len(dataset))} graphs...")
    dataset.preload_all(verbose=True)
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize first n graphs
    for i in range(min(n, len(dataset))):
        try:
            data = dataset[i]
            
            if not hasattr(data, 'pos') or data.pos is None:
                print(f"Warning: Graph {i} does not have 'pos' field, skipping")
                continue
            
            # Get PDB path for filename
            try:
                manifest_df = pd.read_csv(csv_path)
                if i < len(manifest_df):
                    pdb_name = Path(manifest_df.iloc[i]['pdb_path']).stem
                else:
                    pdb_name = f"graph_{i}"
            except:
                pdb_name = f"graph_{i}"
            
            out_path = out_dir / f"{pdb_name}.png"
            
            print(f"Visualizing graph {i+1}/{min(n, len(dataset))}: {pdb_name}")
            plot_graph_3d(
                data=data,
                out_path=str(out_path),
                max_edges=max_edges,
                show=False,
                color_by="chain_group",
                edge_color_by="edge_type",
                subsample_edges=True,
                only_bound_edges=only_bound_edges,
                label_chains=False
            )
        except Exception as e:
            print(f"Error visualizing graph {i}: {e}")
            continue
    
    print(f"\nSaved {min(n, len(dataset))} visualizations to: {out_dir}")


def visualize_run(log_dir: str):
    """Print TensorBoard log directory information."""
    log_dir = Path(log_dir)
    
    if not log_dir.exists():
        print(f"Warning: Log directory not found: {log_dir}")
        return
    
    # Look for TensorBoard logs
    tb_dirs = list(log_dir.rglob("events.out.tfevents.*"))
    if tb_dirs:
        tb_parent = tb_dirs[0].parent
        print(f"TensorBoard logs found in: {tb_parent}")
        print(f"\nTo view, run:")
        print(f"  tensorboard --logdir {tb_parent.parent}")
    else:
        print(f"No TensorBoard logs found in: {log_dir}")
        print(f"Expected structure: {log_dir}/runs/<timestamp>/")


def main():
    parser = argparse.ArgumentParser(description="Visualization tools for GAT regression")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # graph command
    graph_parser = subparsers.add_parser('graph', help='Visualize a single graph')
    graph_group = graph_parser.add_mutually_exclusive_group(required=True)
    graph_group.add_argument('--cache-path', type=str, help='Path to cached graph (.pt file)')
    graph_group.add_argument('--pdb-path', type=str, help='Path to PDB file (will build graph)')
    graph_parser.add_argument('--out', type=str, required=True, help='Output PNG path')
    graph_parser.add_argument('--only-bound-edges', action='store_true', help='Plot only bound (interface) edges')
    graph_parser.add_argument('--edge-max-dist', type=float, help='Maximum edge distance to plot (Angstroms)')
    graph_parser.add_argument('--max-edges', type=int, default=5000, help='Maximum edges to plot (default: 5000)')
    graph_parser.add_argument('--color-by', choices=['chain_group', 'chain_id'], default='chain_group',
                             help='Node coloring scheme (default: chain_group)')
    graph_parser.add_argument('--label-chains', action='store_true', help='Add legend for chain labels')
    graph_parser.add_argument('--config', type=str, help='Config YAML (for --pdb-path)')
    graph_parser.add_argument('--antibody-chains', nargs='+', help='Antibody chain IDs (for --pdb-path)')
    graph_parser.add_argument('--antigen-chains', nargs='+', help='Antigen chain IDs (for --pdb-path)')
    
    # graphs_from_csv command
    csv_parser = subparsers.add_parser('graphs_from_csv', help='Visualize multiple graphs from CSV')
    csv_parser.add_argument('--csv', type=str, required=True, help='Path to manifest CSV')
    csv_parser.add_argument('--out-dir', type=str, required=True, help='Output directory for PNGs')
    csv_parser.add_argument('--n', type=int, default=20, help='Number of graphs to visualize (default: 20)')
    csv_parser.add_argument('--pdb-dir', type=str, help='Base directory for PDB files')
    csv_parser.add_argument('--config', type=str, help='Config YAML file')
    csv_parser.add_argument('--graph-cache-dir', type=str, help='Graph cache directory')
    csv_parser.add_argument('--only-bound-edges', action='store_true', help='Plot only bound edges')
    csv_parser.add_argument('--max-edges', type=int, default=5000, help='Maximum edges per plot (default: 5000)')
    
    # run command
    run_parser = subparsers.add_parser('run', help='Show TensorBoard log directory info')
    run_parser.add_argument('--log-dir', type=str, required=True, help='Training log directory')
    
    # cache_stats command
    stats_parser = subparsers.add_parser('cache_stats', help='Plot cache statistics')
    stats_parser.add_argument('--stats-csv', type=str, required=True, help='Path to cache stats CSV')
    stats_parser.add_argument('--out-dir', type=str, help='Output directory for plots')
    
    args = parser.parse_args()
    
    if args.command == 'graph':
        if args.cache_path:
            visualize_graph_from_cache(
                cache_path=args.cache_path,
                out_path=args.out,
                only_bound_edges=args.only_bound_edges,
                edge_max_dist=args.edge_max_dist,
                max_edges=args.max_edges,
                color_by=args.color_by,
                label_chains=args.label_chains
            )
        elif args.pdb_path:
            visualize_graph_from_pdb(
                pdb_path=args.pdb_path,
                out_path=args.out,
                antibody_chains=args.antibody_chains,
                antigen_chains=args.antigen_chains,
                config_path=args.config,
                only_bound_edges=args.only_bound_edges,
                edge_max_dist=args.edge_max_dist,
                max_edges=args.max_edges,
                color_by=args.color_by,
                label_chains=args.label_chains
            )
    
    elif args.command == 'graphs_from_csv':
        visualize_graphs_from_csv(
            csv_path=args.csv,
            out_dir=args.out_dir,
            n=args.n,
            pdb_dir=args.pdb_dir,
            config_path=args.config,
            graph_cache_dir=args.graph_cache_dir,
            only_bound_edges=args.only_bound_edges,
            max_edges=args.max_edges
        )
    
    elif args.command == 'run':
        visualize_run(args.log_dir)
    
    elif args.command == 'cache_stats':
        plot_cache_stats(
            stats_csv=args.stats_csv,
            out_dir=args.out_dir
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

