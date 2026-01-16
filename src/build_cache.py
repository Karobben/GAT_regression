"""
Offline cache building script for PDB-to-graph preprocessing.
Uses multiprocessing to build cache in parallel.
"""
import argparse
from pathlib import Path
import multiprocessing as mp
from typing import List, Tuple
from tqdm import tqdm
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.data.dataset import CachedGraphDataset


def build_cache_worker(args: Tuple[int, str, dict]) -> Tuple[int, bool, str]:
    """
    Worker function to build cache for a single sample.
    
    Args:
        args: (index, manifest_csv, config_dict) tuple
    
    Returns:
        (index, success, error_message) tuple
    """
    idx, manifest_csv, config_dict = args
    
    try:
        # Reconstruct config from dict (simplified - just what we need)
        from src.config import Config
        config = Config()
        for key, value in config_dict.items():
            if "." in key:
                section, param = key.split(".", 1)
                setattr(getattr(config, section), param, value)
            else:
                setattr(config, key, value)
        
        # Create dataset (this will build cache if needed)
        dataset = CachedGraphDataset(
            manifest_csv=manifest_csv,
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
            graph_cache_dir=config.data.graph_cache_dir,
            hash_pdb_contents=config.data.hash_pdb_contents,
            rebuild_cache=config.data.rebuild_cache,
            cache_stats=False  # Disable stats in workers
        )
        
        # Access the item to trigger cache building
        _ = dataset[idx]
        
        return (idx, True, "")
    except Exception as e:
        return (idx, False, str(e))


def build_cache(
    manifest_csv: str,
    config: Config,
    num_workers: int = None,
    rebuild: bool = False
):
    """
    Build cache for all samples in manifest.
    
    Args:
        manifest_csv: Path to manifest CSV
        config: Config object
        num_workers: Number of parallel workers (None = auto)
        rebuild: If True, rebuild existing cache entries
    """
    # Create dataset to get length and validate
    dataset = CachedGraphDataset(
        manifest_csv=manifest_csv,
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
        graph_cache_dir=config.data.graph_cache_dir,
        hash_pdb_contents=config.data.hash_pdb_contents,
        rebuild_cache=rebuild,
        cache_stats=False
    )
    
    num_samples = len(dataset)
    print(f"Building cache for {num_samples} samples...")
    print(f"Cache directory: {config.data.graph_cache_dir}")
    print(f"Workers: {num_workers or mp.cpu_count()}")
    
    # Prepare config dict for workers
    config_dict = {
        "data.pdb_dir": config.data.pdb_dir,
        "data.default_antibody_chains": config.data.default_antibody_chains,
        "data.default_antigen_chains": config.data.default_antigen_chains,
        "data.graph_cache_dir": config.data.graph_cache_dir,
        "data.hash_pdb_contents": config.data.hash_pdb_contents,
        "data.rebuild_cache": rebuild,
        "graph.noncovalent_cutoff": config.graph.noncovalent_cutoff,
        "graph.interface_cutoff": config.graph.interface_cutoff,
        "graph.use_covalent_edges": config.graph.use_covalent_edges,
        "graph.use_noncovalent_edges": config.graph.use_noncovalent_edges,
        "graph.allow_duplicate_edges": config.graph.allow_duplicate_edges,
        "graph.include_residue_index": config.graph.include_residue_index,
        "graph.add_interface_features_to_x": config.graph.add_interface_features_to_x,
    }
    
    # Prepare arguments for workers
    worker_args = [(idx, manifest_csv, config_dict) for idx in range(num_samples)]
    
    # Build cache
    if num_workers is None or num_workers > 1:
        num_workers = num_workers or mp.cpu_count()
        print(f"Using {num_workers} workers...")
        
        with mp.Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(build_cache_worker, worker_args),
                total=num_samples,
                desc="Building cache"
            ))
    else:
        # Single-threaded
        print("Using single worker...")
        results = [build_cache_worker(args) for args in tqdm(worker_args, desc="Building cache")]
    
    # Report results
    successes = sum(1 for _, success, _ in results if success)
    failures = num_samples - successes
    
    print(f"\nCache building complete:")
    print(f"  Success: {successes}/{num_samples}")
    print(f"  Failed: {failures}/{num_samples}")
    
    if failures > 0:
        print("\nFailures:")
        for idx, success, error in results:
            if not success:
                print(f"  Sample {idx}: {error}")
    
    return successes, failures


def main():
    parser = argparse.ArgumentParser(description="Build graph cache offline")
    parser.add_argument("--csv", type=str, required=True, help="Path to manifest CSV")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--graph-cache-dir", type=str, default=None,
                       help="Directory to cache graphs (overrides config)")
    parser.add_argument("--num-workers", type=int, default=None,
                       help="Number of parallel workers (default: CPU count)")
    parser.add_argument("--rebuild", action="store_true",
                       help="Rebuild existing cache entries")
    parser.add_argument("--hash-pdb-contents", action="store_true",
                       help="Hash PDB file contents for cache key")
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()
    
    # Override from command line
    if args.graph_cache_dir:
        config.data.graph_cache_dir = args.graph_cache_dir
    if args.hash_pdb_contents:
        config.data.hash_pdb_contents = True
    
    # Build cache
    build_cache(
        manifest_csv=args.csv,
        config=config,
        num_workers=args.num_workers,
        rebuild=args.rebuild
    )


if __name__ == "__main__":
    main()


