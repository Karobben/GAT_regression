"""
Utility functions for webapp to load runs, metrics, and graphs.
"""
import json
import csv
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from torch_geometric.data import Data


def list_runs(runs_dir: Path) -> List[Dict[str, str]]:
    """
    List all available runs in runs directory.
    
    Returns:
        List of dicts with 'run_id' and 'path' keys
    """
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return []
    
    runs = []
    for run_path in sorted(runs_dir.iterdir(), reverse=True):
        if run_path.is_dir():
            # Check if it has config.yaml or config.json
            if (run_path / "config.yaml").exists() or (run_path / "config.json").exists():
                runs.append({
                    "run_id": run_path.name,
                    "path": str(run_path)
                })
    
    return runs


def load_config(run_dir: Path):
    """Load config from run directory."""
    from src.config import Config
    
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        config_path = run_dir / "config.json"
    
    if config_path.exists():
        return Config.from_yaml(str(config_path))
    return None


def load_metrics(run_dir: Path, split: str = "train") -> Optional[pd.DataFrame]:
    """
    Load metrics CSV for a run.
    
    Args:
        run_dir: Run directory
        split: "train" or "val"
    
    Returns:
        DataFrame with metrics, or None if file doesn't exist
    """
    csv_path = run_dir / f"metrics_{split}.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def load_eval_predictions(run_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load eval_predictions.csv from run directory or checkpoint directory.
    
    Args:
        run_dir: Run directory (will also check checkpoint dir)
    
    Returns:
        DataFrame with columns: sample_id, y, score
    """
    # Try run_dir first
    csv_path = run_dir / "eval_predictions.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    
    # Try checkpoint directory (for backward compatibility)
    checkpoint_dir = run_dir.parent.parent / "checkpoints"
    csv_path = checkpoint_dir / "eval_predictions.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    
    return None


def get_cache_dir(run_dir: Path) -> Optional[Path]:
    """
    Get graph cache directory from cache_link.txt.
    
    Args:
        run_dir: Run directory
    
    Returns:
        Path to cache directory, or None
    """
    link_path = run_dir / "cache_link.txt"
    if link_path.exists():
        with open(link_path, "r") as f:
            cache_path = Path(f.read().strip())
            if cache_path.exists():
                return cache_path
    return None


def list_available_graphs(cache_dir: Path) -> List[Dict[str, str]]:
    """
    List all available graphs in the cache directory.
    
    Args:
        cache_dir: Graph cache directory
    
    Returns:
        List of dicts with 'sample_id', 'pdb_path', and 'cache_path' keys
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return []
    
    graphs = []
    # Search in subdirectories (cache uses first 2 chars of hash as subdir)
    for subdir in cache_dir.iterdir():
        if subdir.is_dir():
            # Look for .pt files
            for pt_file in subdir.glob("*.pt"):
                # Check metadata to get PDB path
                meta_path = pt_file.with_suffix(".meta.json")
                if meta_path.exists():
                    try:
                        with open(meta_path, "r") as f:
                            meta = json.load(f)
                            pdb_path = meta.get("pdb_path", "")
                            if pdb_path:
                                pdb_name = Path(pdb_path).name
                                graphs.append({
                                    "sample_id": pdb_name,
                                    "pdb_path": pdb_path,
                                    "cache_path": str(pt_file)
                                })
                    except Exception:
                        continue
    
    # Sort by sample_id for consistent ordering
    graphs.sort(key=lambda x: x["sample_id"])
    return graphs


def find_graph_cache_path(
    cache_dir: Path,
    sample_id: str,
    pdb_path: Optional[str] = None
) -> Optional[Path]:
    """
    Find cached graph file for a sample.
    
    Args:
        cache_dir: Graph cache directory
        sample_id: Sample ID (PDB filename)
        pdb_path: Optional full PDB path for more precise lookup
    
    Returns:
        Path to cached graph file, or None
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return None
    
    # Search in subdirectories (cache uses first 2 chars of hash as subdir)
    for subdir in cache_dir.iterdir():
        if subdir.is_dir():
            # Look for .pt files
            for pt_file in subdir.glob("*.pt"):
                # Check metadata to see if it matches
                meta_path = pt_file.with_suffix(".meta.json")
                if meta_path.exists():
                    try:
                        with open(meta_path, "r") as f:
                            meta = json.load(f)
                            meta_pdb = Path(meta.get("pdb_path", ""))
                            if meta_pdb.name == sample_id or (pdb_path and str(meta_pdb) == pdb_path):
                                return pt_file
                    except Exception:
                        continue
    
    return None


@torch.no_grad()
def load_graph(cache_path: Path) -> Tuple[Data, Dict]:
    """
    Load graph from cache file.
    
    Args:
        cache_path: Path to cached graph file
    
    Returns:
        (graph, metadata) tuple
    """
    from src.data.cache_utils import load_graph_from_cache
    
    graph, metadata = load_graph_from_cache(cache_path, device=None)
    
    # Ensure graph is on CPU and has all required fields
    graph = graph.cpu()
    
    # Verify required fields exist (support both old and new field names)
    required_fields = ['pos', 'chain_id', 'res_id', 'aa', 'edge_type', 'edge_dist']
    # Check for chain_role (new) or chain_group (old) for backward compatibility
    if not hasattr(graph, 'chain_role') and not hasattr(graph, 'chain_group'):
        raise ValueError("Graph missing required field: chain_role or chain_group")
    missing = [f for f in required_fields if not hasattr(graph, f)]
    if missing:
        raise ValueError(f"Graph missing required fields: {missing}")
    
    return graph, metadata


def get_graph_stats(graph: Data, metadata: Dict) -> Dict:
    """
    Get statistics about a graph for display.
    
    Args:
        graph: Graph Data object
        metadata: Graph metadata
    
    Returns:
        Dictionary with stats
    """
    stats = {
        "num_nodes": int(graph.num_nodes),
        "num_edges": int(graph.edge_index.shape[1]),
    }
    
    # Edge type counts
    if hasattr(graph, 'edge_type'):
        edge_types = graph.edge_type.unique().int().tolist()
        edge_type_names = {0: "COVALENT", 1: "NONCOVALENT"}
        stats["edge_type_counts"] = {
            edge_type_names.get(et, f"Type {et}"): int((graph.edge_type == et).sum())
            for et in edge_types
        }
    else:
        stats["edge_type_counts"] = metadata.get("edge_type_counts", {})
    
    # Chain info
    if hasattr(graph, 'chain_id'):
        unique_chains = graph.chain_id.unique().int().tolist()
        stats["num_chains"] = len(unique_chains)
    
    # Support both chain_role (new) and chain_group (old) for backward compatibility
    chain_role_field = 'chain_role' if hasattr(graph, 'chain_role') else 'chain_group'
    if hasattr(graph, chain_role_field):
        chain_role = getattr(graph, chain_role_field)
        antibody_count = int((chain_role == 0).sum())
        antigen_count = int((chain_role == 1).sum())
        stats["antibody_nodes"] = antibody_count
        stats["antigen_nodes"] = antigen_count
    
    # Chain mapping from metadata
    chain_mapping = metadata.get("chain_mapping", {})
    if chain_mapping:
        stats["chain_labels"] = {
            int(k): v.get("label", f"Chain{k}") for k, v in chain_mapping.items()
        }
        stats["chain_roles"] = {
            int(k): "Antibody" if v.get("role", 1) == 0 else "Antigen"
            for k, v in chain_mapping.items()
        }
    elif hasattr(graph, 'chain_labels'):
        # Fallback to chain_labels attribute
        chain_labels = graph.chain_labels if isinstance(graph.chain_labels, list) else graph.chain_labels.tolist()
        stats["chain_labels"] = {i: label for i, label in enumerate(chain_labels)}
        # Infer roles from chain_role (new) or chain_group (old)
        chain_role_field = 'chain_role' if hasattr(graph, 'chain_role') else 'chain_group'
        if hasattr(graph, chain_role_field):
            unique_chain_ids = graph.chain_id.unique().int().tolist()
            stats["chain_roles"] = {}
            chain_role = getattr(graph, chain_role_field)
            for chain_idx in unique_chain_ids:
                node_mask = graph.chain_id == chain_idx
                if node_mask.any():
                    role_val = chain_role[node_mask][0].item()
                    stats["chain_roles"][chain_idx] = "Antibody" if role_val == 0 else "Antigen"
    
    # Interface markers (if available)
    if hasattr(graph, 'is_interface'):
        stats["num_interface_nodes"] = int(graph.is_interface.sum())
    if hasattr(graph, 'min_inter_dist'):
        stats["has_interface_markers"] = True
    
    return stats

