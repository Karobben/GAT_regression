"""
Dataset class for loading PDB graphs from CSV manifest with disk caching.
"""
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from pathlib import Path
from typing import Optional, List, Callable, Dict
import warnings
import time
from datetime import datetime

from .pdb_to_graph import pdb_to_graph
from .transforms import IdentityTransform
from .cache_utils import (
    PreprocessConfig,
    make_cache_key,
    get_cache_path,
    save_graph_to_cache,
    load_graph_from_cache,
    verify_cache_metadata
)


def _preload_worker(args):
    """
    Worker function for parallel graph preloading.
    
    Args:
        args: Tuple of (idx, pdb_path, y, antibody_chains, antigen_chains, ...)
    
    Returns:
        (success, was_cache_hit, build_time) tuple
    """
    import time
    from pathlib import Path
    from .pdb_to_graph import pdb_to_graph
    from .cache_utils import (
        PreprocessConfig, make_cache_key, get_cache_path,
        save_graph_to_cache, verify_cache_metadata
    )
    from Bio.PDB import PDBParser
    import warnings
    from datetime import datetime
    
    (idx, pdb_path, y, antibody_chains, antigen_chains,
     noncovalent_cutoff, interface_cutoff, use_covalent_edges,
     use_noncovalent_edges, allow_duplicate_edges, include_residue_index,
     add_interface_features_to_x, cache_dir, hash_pdb_contents, rebuild_cache) = args
    
    pdb_path = Path(pdb_path)
    cache_dir = Path(cache_dir)
    was_cache_hit = False
    build_time = 0.0
    
    try:
        # Infer antigen chains if needed
        if antigen_chains is None:
            parser = PDBParser(QUIET=True)
            try:
                structure = parser.get_structure("complex", str(pdb_path))
                all_chain_ids = [chain.id for chain in structure[0].get_chains()]
                antigen_chains = [cid for cid in all_chain_ids if cid not in antibody_chains]
                if not antigen_chains:
                    antigen_chains = all_chain_ids[len(antibody_chains):] if len(all_chain_ids) > len(antibody_chains) else []
            except Exception:
                antigen_chains = []
        
        # Create preprocessing config
        preprocess_config = PreprocessConfig(
            pdb_path=str(pdb_path),
            antibody_chains=antibody_chains,
            antigen_chains=antigen_chains,
            noncovalent_cutoff=noncovalent_cutoff,
            interface_cutoff=interface_cutoff,
            use_covalent_edges=use_covalent_edges,
            use_noncovalent_edges=use_noncovalent_edges,
            allow_duplicate_edges=allow_duplicate_edges,
            include_residue_index=include_residue_index,
            add_interface_features_to_x=add_interface_features_to_x
        )
        
        # Generate cache key
        cache_key = make_cache_key(str(pdb_path), preprocess_config, hash_pdb_contents)
        cache_path = get_cache_path(cache_dir, cache_key)
        
        # Try to load from cache
        if not rebuild_cache and cache_path.exists():
            try:
                if verify_cache_metadata(cache_path, preprocess_config):
                    was_cache_hit = True
                    return (True, was_cache_hit, build_time)
            except Exception:
                pass  # Cache invalid, rebuild
        
        # Build graph from PDB
        build_start = time.time()
        data = pdb_to_graph(
            pdb_path=str(pdb_path),
            antibody_chains=antibody_chains,
            antigen_chains=antigen_chains,
            noncovalent_cutoff=noncovalent_cutoff,
            interface_cutoff=interface_cutoff,
            use_covalent_edges=use_covalent_edges,
            use_noncovalent_edges=use_noncovalent_edges,
            allow_duplicate_edges=allow_duplicate_edges,
            include_residue_index=include_residue_index,
            add_interface_features_to_x=add_interface_features_to_x,
            y=y
        )
        build_time = time.time() - build_start
        
        # Save to cache
        try:
            # Count edge types for metadata
            if hasattr(data, 'edge_type') and data.edge_type is not None:
                edge_types = data.edge_type.unique().int().tolist()
                edge_type_counts = {int(et): int((data.edge_type == et).sum()) for et in edge_types}
            else:
                edge_type_counts = {}
            
            # Build chain mapping
            chain_mapping = {}
            if hasattr(data, 'chain_labels') and data.chain_labels is not None:
                chain_labels = data.chain_labels if isinstance(data.chain_labels, list) else data.chain_labels.tolist()
                for chain_idx, chain_label in enumerate(chain_labels):
                    role = 0 if chain_label in antibody_chains else 1
                    chain_mapping[str(chain_idx)] = {
                        "label": chain_label,
                        "role": role
                    }
            
            metadata = {
                "pdb_path": str(pdb_path),
                "config": preprocess_config.to_dict(),
                "created": datetime.now().isoformat(),
                "num_nodes": int(data.num_nodes),
                "num_edges": int(data.edge_index.shape[1]),
                "edge_type_counts": edge_type_counts,
                "chain_mapping": chain_mapping
            }
            
            save_graph_to_cache(data, cache_path, metadata)
        except Exception as e:
            warnings.warn(f"Failed to save cache for {pdb_path}: {e}")
        
        return (True, was_cache_hit, build_time)
        
    except Exception as e:
        warnings.warn(f"Error processing {pdb_path}: {e}")
        return (False, was_cache_hit, build_time)


class CachedGraphDataset(Dataset):
    """
    Dataset for antibody-antigen complexes with disk caching.
    
    Expected CSV format:
    - pdb_path: Path to PDB file (relative or absolute)
    - y: Continuous binding property value (e.g., -log(KD))
    - antibody_chains (optional): Comma-separated chain IDs (e.g., "H,L")
    - antigen_chains (optional): Comma-separated chain IDs (e.g., "A")
    """
    
    def __init__(
        self,
        manifest_csv: str,
        pdb_dir: Optional[str] = None,
        default_antibody_chains: List[str] = None,
        default_antigen_chains: Optional[List[str]] = None,
        noncovalent_cutoff: float = 10.0,
        interface_cutoff: float = 8.0,
        use_covalent_edges: bool = True,
        use_noncovalent_edges: bool = True,
        allow_duplicate_edges: bool = False,
        include_residue_index: bool = True,
        add_interface_features_to_x: bool = True,
        transform: Optional[Callable] = None,
        graph_cache_dir: str = "cache/graphs",
        hash_pdb_contents: bool = False,
        rebuild_cache: bool = False,
        cache_stats: bool = True
    ):
        """
        Initialize dataset with disk caching.
        
        Args:
            manifest_csv: Path to CSV file with columns: pdb_path, y, [antibody_chains], [antigen_chains]
            pdb_dir: Optional base directory for PDB paths (if pdb_path in CSV is relative).
                     If pdb_path is absolute, this parameter is ignored.
            default_antibody_chains: Default antibody chains if not in CSV (default: ["H", "L"])
            default_antigen_chains: Default antigen chains if not in CSV (default: None, infer from remaining)
            noncovalent_cutoff: Distance cutoff for NONCOVALENT edges (Angstroms)
            interface_cutoff: Distance cutoff for interface definition (Angstroms)
            use_covalent_edges: Whether to include covalent edges
            use_noncovalent_edges: Whether to include noncovalent edges
            allow_duplicate_edges: If False, exclude covalent pairs from noncovalent edges
            include_residue_index: Whether to include residue index in node features
            add_interface_features_to_x: Whether to add interface features to node features
            transform: Optional transform to apply to graphs
            graph_cache_dir: Directory to cache preprocessed graphs
            hash_pdb_contents: If True, hash file contents; else use mtime+size
            rebuild_cache: If True, ignore existing cache and rebuild
            cache_stats: Print cache hit/miss statistics
        """
        self.manifest_csv = Path(manifest_csv)
        self.pdb_dir = Path(pdb_dir) if pdb_dir else None
        self.default_antibody_chains = default_antibody_chains or ["H", "L"]
        self.default_antigen_chains = default_antigen_chains
        self.noncovalent_cutoff = noncovalent_cutoff
        self.interface_cutoff = interface_cutoff
        self.use_covalent_edges = use_covalent_edges
        self.use_noncovalent_edges = use_noncovalent_edges
        self.allow_duplicate_edges = allow_duplicate_edges
        self.include_residue_index = include_residue_index
        self.add_interface_features_to_x = add_interface_features_to_x
        self.transform = transform if transform else IdentityTransform()
        self.hash_pdb_contents = hash_pdb_contents
        self.rebuild_cache = rebuild_cache
        self.cache_stats = cache_stats
        
        # Setup cache directory
        self.cache_dir = Path(graph_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache statistics
        self._cache_hits = 0
        self._cache_misses = 0
        self._build_times = []
        
        # Load manifest
        if not self.manifest_csv.exists():
            raise FileNotFoundError(f"Manifest CSV not found: {self.manifest_csv}")
        
        self.df = pd.read_csv(self.manifest_csv)
        
        # Validate required columns
        if "pdb_path" not in self.df.columns:
            raise ValueError("CSV must contain 'pdb_path' column")
        if "y" not in self.df.columns:
            raise ValueError("CSV must contain 'y' column")
        
        # Check for potential CSV parsing issues with comma-separated chain values
        expected_cols = {"pdb_path", "y"}
        optional_cols = {"antibody_chains", "antigen_chains"}
        if len(self.df.columns) > len(expected_cols | optional_cols):
            warnings.warn(
                f"CSV has {len(self.df.columns)} columns, which is more than expected. "
                f"This may indicate that comma-separated chain values were split into multiple columns. "
                f"Ensure comma-separated values like 'H,L' are quoted in CSV: '\"H,L\"'"
            )
        
        # Parse chain assignments
        self.antibody_chains_list = []
        self.antigen_chains_list = []
        
        for idx, row in self.df.iterrows():
            # Parse antibody chains (support both comma and colon separators)
            if "antibody_chains" in row and pd.notna(row["antibody_chains"]):
                ab_str = str(row["antibody_chains"]).strip()
                if ":" in ab_str:
                    ab_chains = [c.strip() for c in ab_str.split(":")]
                else:
                    ab_chains = [c.strip() for c in ab_str.split(",")]
            else:
                ab_chains = self.default_antibody_chains.copy()
            
            # Parse antigen chains (support both comma and colon separators)
            if "antigen_chains" in row and pd.notna(row["antigen_chains"]):
                ag_str = str(row["antigen_chains"]).strip()
                if ":" in ag_str:
                    ag_chains = [c.strip() for c in ag_str.split(":")]
                else:
                    ag_chains = [c.strip() for c in ag_str.split(",")]
            elif self.default_antigen_chains:
                ag_chains = self.default_antigen_chains.copy()
            else:
                ag_chains = None  # Will be inferred from PDB structure
            
            self.antibody_chains_list.append(ab_chains)
            self.antigen_chains_list.append(ag_chains)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def preload_all(self, verbose: bool = True, num_workers: int = 0):
        """
        Preload all graphs (builds cache if needed).

        With disk caching, this will:
        - Load from cache if available
        - Build and save to cache if not available

        Args:
            verbose: Whether to print progress
            num_workers: Number of parallel workers (0 = sequential, >0 = multiprocessing, None = auto)
        """
        if verbose:
            print(f"Preloading {len(self)} graphs (will use cache if available)...")

        # Pre-scan cache to identify which graphs need processing
        needs_processing = []
        cached_count = 0

        if not self.rebuild_cache:
            try:
                from tqdm import tqdm
                cache_scan_iter = tqdm(range(len(self)), disable=not verbose, desc="Scanning cache")
            except ImportError:
                cache_scan_iter = range(len(self))

            for idx in cache_scan_iter:
                row = self.df.iloc[idx]
                pdb_path = Path(row["pdb_path"])
                if pdb_path.is_absolute():
                    pass
                elif self.pdb_dir:
                    pdb_path = self.pdb_dir / pdb_path
                else:
                    if not pdb_path.exists():
                        manifest_dir = self.manifest_csv.parent
                        potential_path = manifest_dir / pdb_path
                        if potential_path.exists():
                            pdb_path = potential_path
                pdb_path = pdb_path.resolve()

                antibody_chains = self.antibody_chains_list[idx]
                antigen_chains = self.antigen_chains_list[idx]

                # Create preprocessing config
                preprocess_config = PreprocessConfig(
                    pdb_path=str(pdb_path),
                    antibody_chains=antibody_chains,
                    antigen_chains=antigen_chains,
                    noncovalent_cutoff=self.noncovalent_cutoff,
                    interface_cutoff=self.interface_cutoff,
                    use_covalent_edges=self.use_covalent_edges,
                    use_noncovalent_edges=self.use_noncovalent_edges,
                    allow_duplicate_edges=self.allow_duplicate_edges,
                    include_residue_index=self.include_residue_index,
                    add_interface_features_to_x=self.add_interface_features_to_x
                )

                # Check if cache exists
                cache_key = make_cache_key(str(pdb_path), preprocess_config, self.hash_pdb_contents)
                cache_path = get_cache_path(self.cache_dir, cache_key)

                if cache_path.exists():
                    try:
                        if verify_cache_metadata(cache_path, preprocess_config):
                            cached_count += 1
                            self._cache_hits += 1
                            continue
                    except Exception:
                        pass  # Cache invalid, needs rebuild

                needs_processing.append(idx)

        if verbose:
            total = len(self)
            to_process = len(needs_processing)
            print(f"Cache scan complete: {cached_count}/{total} graphs already cached, {to_process} need processing")

        if not needs_processing:
            if verbose:
                print("All graphs are already cached!")
            return

        # Use multiprocessing if requested and beneficial
        if num_workers is None or (num_workers > 0 and len(needs_processing) > 1):
            import multiprocessing as mp
            if num_workers is None:
                num_workers = min(mp.cpu_count(), len(needs_processing))  # Don't use more workers than tasks

            if verbose:
                print(f"Using {num_workers} workers for parallel processing of {len(needs_processing)} graphs...")

            # Prepare worker arguments only for graphs that need processing
            worker_args = []
            for idx in needs_processing:
                row = self.df.iloc[idx]
                pdb_path = Path(row["pdb_path"])
                if pdb_path.is_absolute():
                    pass
                elif self.pdb_dir:
                    pdb_path = self.pdb_dir / pdb_path
                else:
                    if not pdb_path.exists():
                        manifest_dir = self.manifest_csv.parent
                        potential_path = manifest_dir / pdb_path
                        if potential_path.exists():
                            pdb_path = potential_path
                pdb_path = pdb_path.resolve()

                antibody_chains = self.antibody_chains_list[idx]
                antigen_chains = self.antigen_chains_list[idx]

                worker_args.append((
                    idx,
                    str(pdb_path),
                    row["y"],
                    antibody_chains,
                    antigen_chains,
                    self.noncovalent_cutoff,
                    self.interface_cutoff,
                    self.use_covalent_edges,
                    self.use_noncovalent_edges,
                    self.allow_duplicate_edges,
                    self.include_residue_index,
                    self.add_interface_features_to_x,
                    str(self.cache_dir),
                    self.hash_pdb_contents,
                    self.rebuild_cache
                ))

            # Process in parallel
            try:
                from tqdm import tqdm
                with mp.Pool(num_workers) as pool:
                    results = list(tqdm(
                        pool.imap(_preload_worker, worker_args),
                        total=len(worker_args),
                        disable=not verbose,
                        desc="Building graphs"
                    ))

                # Update cache stats from results
                for success, was_cache_hit, build_time in results:
                    if was_cache_hit:
                        self._cache_hits += 1
                    else:
                        self._cache_misses += 1
                    if build_time > 0:
                        self._build_times.append(build_time)
            except Exception as e:
                if verbose:
                    print(f"Warning: Multiprocessing failed ({e}), falling back to sequential...")
                # Fall back to sequential
                num_workers = 0

        # Sequential processing
        if num_workers == 0:
            try:
                from tqdm import tqdm
                for idx in tqdm(needs_processing, disable=not verbose, desc="Building graphs"):
                    _ = self[idx]  # This will use cache or build it
            except ImportError:
                # tqdm not available, use simple loop
                for idx in needs_processing:
                    if verbose and idx % 10 == 0:
                        print(f"Building graph {idx}/{len(self)}...")
                    _ = self[idx]

        if verbose:
            self.print_cache_stats()

    def check_graph_health(self, verbose: bool = True, save_unhealthy_dir: str = None) -> Dict[str, any]:
        """
        Check the health of all graphs in the dataset.

        Performs comprehensive health checks including:
        - Missing required fields
        - Empty graphs (no nodes/edges)
        - Interface node issues
        - Structural problems

        Args:
            verbose: Whether to print detailed health report
            save_unhealthy_dir: Directory to save unhealthy graphs lists by issue type

        Returns:
            Health report dictionary with statistics and issue lists
        """
        if verbose:
            print("\n" + "="*60)
            print("GRAPH HEALTH CHECK")
            print("="*60)

        health_report = {
            "total_graphs": len(self),
            "healthy_graphs": 0,
            "issues_found": False,
            "issues": {
                "missing_is_interface": [],
                "missing_chain_role": [],
                "missing_edge_type": [],
                "no_nodes": [],
                "no_edges": [],
                "no_interface_nodes": [],
                "few_interface_nodes": [],  # Less than 3 interface nodes
                "isolated_nodes": [],  # Nodes with no edges
                "invalid_interface_markers": [],  # Interface markers don't match calculations
            },
            "statistics": {
                "avg_nodes_per_graph": 0.0,
                "avg_edges_per_graph": 0.0,
                "avg_interface_nodes_per_graph": 0.0,
                "interface_node_distribution": [],
                "edge_type_distribution": {"covalent": 0, "noncovalent": 0}
            }
        }

        total_nodes = 0
        total_edges = 0
        total_interface_nodes = 0

        try:
            from tqdm import tqdm
            iterator = tqdm(range(len(self)), disable=not verbose, desc="Health checking")
        except ImportError:
            iterator = range(len(self))

        for idx in iterator:
            try:
                # Load the graph
                data = self[idx]

                # Check required fields
                if not hasattr(data, 'is_interface') or data.is_interface is None:
                    health_report["issues"]["missing_is_interface"].append(idx)
                    continue

                if not hasattr(data, 'chain_role') or data.chain_role is None:
                    health_report["issues"]["missing_chain_role"].append(idx)
                    continue

                if not hasattr(data, 'edge_type') or data.edge_type is None:
                    health_report["issues"]["missing_edge_type"].append(idx)
                    continue

                # Basic structure checks
                num_nodes = data.num_nodes
                num_edges = data.edge_index.shape[1] if data.edge_index is not None else 0

                if num_nodes == 0:
                    health_report["issues"]["no_nodes"].append(idx)
                    continue

                if num_edges == 0:
                    health_report["issues"]["no_edges"].append(idx)
                    continue

                # Interface node checks
                interface_mask = (data.is_interface == 1)
                num_interface_nodes = interface_mask.sum().item()

                if num_interface_nodes == 0:
                    health_report["issues"]["no_interface_nodes"].append(idx)
                    continue

                if num_interface_nodes < 3:
                    health_report["issues"]["few_interface_nodes"].append((idx, num_interface_nodes))

                # Check for isolated nodes (nodes with no edges)
                row, col = data.edge_index
                node_degrees = torch.zeros(num_nodes, dtype=torch.int)
                node_degrees.scatter_add_(0, row, torch.ones_like(row, dtype=torch.int))
                node_degrees.scatter_add_(0, col, torch.ones_like(col, dtype=torch.int))
                isolated_count = (node_degrees == 0).sum().item()

                if isolated_count > 0:
                    health_report["issues"]["isolated_nodes"].append((idx, isolated_count))

                # Validate interface markers (basic sanity check)
                # For interface nodes, min_inter_dist should be finite and reasonable
                if hasattr(data, 'min_inter_dist') and data.min_inter_dist is not None:
                    interface_min_dists = data.min_inter_dist[interface_mask]
                    invalid_markers = (
                        torch.isnan(interface_min_dists) |
                        torch.isinf(interface_min_dists) |
                        (interface_min_dists <= 0) |
                        (interface_min_dists > 50)  # Unreasonably large distance
                    ).sum().item()

                    if invalid_markers > 0:
                        health_report["issues"]["invalid_interface_markers"].append((idx, invalid_markers))

                # Collect statistics
                total_nodes += num_nodes
                total_edges += num_edges
                total_interface_nodes += num_interface_nodes
                health_report["statistics"]["interface_node_distribution"].append(num_interface_nodes)

                # Edge type distribution
                if data.edge_type is not None:
                    covalent_count = (data.edge_type == 0).sum().item()
                    noncovalent_count = (data.edge_type == 1).sum().item()
                    health_report["statistics"]["edge_type_distribution"]["covalent"] += covalent_count
                    health_report["statistics"]["edge_type_distribution"]["noncovalent"] += noncovalent_count

                # Mark as healthy if we got here
                health_report["healthy_graphs"] += 1

            except Exception as e:
                if verbose:
                    print(f"Error checking graph {idx}: {e}")
                continue

        # Calculate averages
        if health_report["healthy_graphs"] > 0:
            health_report["statistics"]["avg_nodes_per_graph"] = total_nodes / health_report["healthy_graphs"]
            health_report["statistics"]["avg_edges_per_graph"] = total_edges / health_report["healthy_graphs"]
            health_report["statistics"]["avg_interface_nodes_per_graph"] = total_interface_nodes / health_report["healthy_graphs"]

        # Check if any issues were found
        health_report["issues_found"] = any(len(issues) > 0 for issues in health_report["issues"].values())

        # Save unhealthy graphs by issue type if requested
        if save_unhealthy_dir and health_report["issues_found"]:
            self._save_unhealthy_graphs_by_type(health_report, save_unhealthy_dir, verbose)

        # Print detailed report
        if verbose:
            self._print_health_report(health_report)

        return health_report

    def _print_health_report(self, report: Dict[str, any]):
        """Print a detailed health report."""
        print(f"\nDataset Health Summary:")
        print(f"Total graphs: {report['total_graphs']}")
        print(f"Healthy graphs: {report['healthy_graphs']} ({report['healthy_graphs']/report['total_graphs']*100:.1f}%)")

        if report["statistics"]["avg_nodes_per_graph"] > 0:
            print(f"\nAverage statistics per graph:")
            print(f"  Average nodes per graph: {report['statistics']['avg_nodes_per_graph']:.1f}")
            print(f"  Average edges per graph: {report['statistics']['avg_edges_per_graph']:.1f}")
            print(f"  Average interface nodes per graph: {report['statistics']['avg_interface_nodes_per_graph']:.1f}")

        if report["statistics"]["edge_type_distribution"]["covalent"] > 0:
            total_edges = (report["statistics"]["edge_type_distribution"]["covalent"] +
                          report["statistics"]["edge_type_distribution"]["noncovalent"])
            if total_edges > 0:
                print(f"\nEdge type distribution:")
                print(f"  Covalent edges: {report['statistics']['edge_type_distribution']['covalent']/total_edges*100:.1f}%")
                print(f"  Noncovalent edges: {report['statistics']['edge_type_distribution']['noncovalent']/total_edges*100:.1f}%")

        # Report issues
        issues_found = False
        for issue_type, issue_list in report["issues"].items():
            if issue_list:
                issues_found = True
                break

        if issues_found:
            print(f"\n⚠️  ISSUES FOUND:")
            for issue_type, issue_list in report["issues"].items():
                if issue_list:
                    print(f"  • {issue_type.replace('_', ' ').title()}: {len(issue_list)} graphs")
                    if len(issue_list) <= 5:  # Show details for small lists
                        for item in issue_list[:5]:
                            if isinstance(item, tuple):
                                print(f"    - Graph {item[0]}: {item[1]}")
                            else:
                                print(f"    - Graph {item}")
                    elif len(issue_list) > 5:
                        print(f"    - First 5: {issue_list[:5]}")
        else:
            print(f"\n✅ No issues found - all graphs appear healthy!")

        print("="*60)

    def _save_unhealthy_graphs_by_type(self, health_report: Dict[str, any], save_dir: str, verbose: bool = True):
        """Save unhealthy graphs to separate files by issue type."""
        import json
        from pathlib import Path
        from datetime import datetime

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Create summary file
        summary_file = save_dir / "health_check_summary.json"
        summary_data = {
            "health_check_summary": {
                "total_graphs": health_report["total_graphs"],
                "healthy_graphs": health_report["healthy_graphs"],
                "unhealthy_graphs": health_report["total_graphs"] - health_report["healthy_graphs"],
                "issues_found": health_report["issues_found"],
                "issues_breakdown": {k: len(v) for k, v in health_report["issues"].items()},
                "statistics": health_report["statistics"],
                "generated_at": datetime.now().isoformat(),
                "output_directory": str(save_dir)
            }
        }

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        # Save each issue type to separate file
        saved_files = []
        for issue_type, issue_list in health_report["issues"].items():
            if not issue_list:
                continue

            # Collect detailed information for graphs with this issue
            unhealthy_graphs = {}

            # Get unique graph indices for this issue
            graph_indices = set()
            for item in issue_list:
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    graph_indices.add(item[0])
                elif isinstance(item, int):
                    graph_indices.add(item)

            # Collect details for each unhealthy graph
            for idx in sorted(graph_indices):
                try:
                    # Get graph data
                    data = self[idx]

                    # Get manifest information
                    row = self.df.iloc[idx]
                    pdb_path = row["pdb_path"]
                    y_value = row["y"]
                    antibody_chains = self.antibody_chains_list[idx] if idx < len(self.antibody_chains_list) else []
                    antigen_chains = self.antigen_chains_list[idx] if idx < len(self.antigen_chains_list) else []

                    # Collect all issues for this graph (not just the current issue type)
                    all_issues = {}
                    for check_issue_type, check_issue_list in health_report["issues"].items():
                        for item in check_issue_list:
                            if isinstance(item, (list, tuple)) and len(item) >= 1 and item[0] == idx:
                                if check_issue_type not in all_issues:
                                    all_issues[check_issue_type] = []
                                if len(item) > 1:
                                    all_issues[check_issue_type].append(item[1])
                                else:
                                    all_issues[check_issue_type] = True
                            elif isinstance(item, int) and item == idx:
                                all_issues[check_issue_type] = True

                    # Collect graph statistics
                    graph_stats = {
                        "num_nodes": data.num_nodes,
                        "num_edges": data.edge_index.shape[1] if data.edge_index is not None else 0,
                        "has_is_interface": hasattr(data, 'is_interface') and data.is_interface is not None,
                        "has_chain_role": hasattr(data, 'chain_role') and data.chain_role is not None,
                        "has_edge_type": hasattr(data, 'edge_type') and data.edge_type is not None,
                        "has_min_inter_dist": hasattr(data, 'min_inter_dist') and data.min_inter_dist is not None,
                        "has_inter_contact_count": hasattr(data, 'inter_contact_count') and data.inter_contact_count is not None,
                    }

                    # Calculate interface statistics if available
                    if hasattr(data, 'is_interface') and data.is_interface is not None:
                        interface_mask = (data.is_interface == 1)
                        graph_stats["num_interface_nodes"] = interface_mask.sum().item()
                        graph_stats["interface_percentage"] = interface_mask.float().mean().item()

                        if hasattr(data, 'min_inter_dist') and data.min_inter_dist is not None:
                            interface_dists = data.min_inter_dist[interface_mask]
                            if len(interface_dists) > 0:
                                graph_stats["min_interface_dist"] = interface_dists.min().item()
                                graph_stats["max_interface_dist"] = interface_dists.max().item()
                                graph_stats["mean_interface_dist"] = interface_dists.mean().item()

                    # Edge type distribution
                    if hasattr(data, 'edge_type') and data.edge_type is not None:
                        edge_types = data.edge_type.unique().tolist()
                        edge_type_counts = {}
                        for et in edge_types:
                            count = (data.edge_type == et).sum().item()
                            edge_type_counts[f"type_{int(et)}"] = count
                        graph_stats["edge_type_counts"] = edge_type_counts

                    # Store unhealthy graph information
                    unhealthy_graphs[str(idx)] = {
                        "index": idx,
                        "pdb_path": pdb_path,
                        "y_value": y_value,
                        "antibody_chains": antibody_chains,
                        "antigen_chains": antigen_chains,
                        "all_issues": all_issues,
                        "statistics": graph_stats
                    }

                except Exception as e:
                    # If we can't load the graph, just record the error
                    unhealthy_graphs[str(idx)] = {
                        "index": idx,
                        "error": f"Could not load graph: {str(e)}",
                        "all_issues": {issue_type: True}
                    }

            # Save to issue-specific file
            if unhealthy_graphs:
                issue_file = save_dir / f"{issue_type}.json"
                with open(issue_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "issue_type": issue_type,
                        "description": self._get_issue_description(issue_type),
                        "num_affected_graphs": len(unhealthy_graphs),
                        "generated_at": datetime.now().isoformat(),
                        "unhealthy_graphs": unhealthy_graphs
                    }, f, indent=2, ensure_ascii=False)
                saved_files.append(str(issue_file))

        if verbose and saved_files:
            print(f"\n💾 Saved unhealthy graphs to separate files in: {save_dir}")
            print(f"   Files created: {len(saved_files)}")
            for file_path in saved_files[:5]:  # Show first 5
                file_name = Path(file_path).name
                print(f"   • {file_name}")
            if len(saved_files) > 5:
                print(f"   ... and {len(saved_files) - 5} more")
            print(f"   📊 Summary: {summary_file}")

    def _get_issue_description(self, issue_type: str) -> str:
        """Get human-readable description for an issue type."""
        descriptions = {
            "missing_is_interface": "Graphs missing the is_interface field (required for interface identification)",
            "missing_chain_role": "Graphs missing the chain_role field (required for antibody/antigen distinction)",
            "missing_edge_type": "Graphs missing the edge_type field (required for edge type information)",
            "no_nodes": "Graphs with zero nodes (empty graphs)",
            "no_edges": "Graphs with zero edges (no connectivity)",
            "no_interface_nodes": "Graphs with no interface nodes (no antibody-antigen contacts)",
            "few_interface_nodes": "Graphs with very few interface nodes (< 3, may indicate weak binding)",
            "isolated_nodes": "Graphs containing nodes with no edges (disconnected components)",
            "invalid_interface_markers": "Graphs with invalid interface distance markers (NaN, inf, or unreasonable values)"
        }
        return descriptions.get(issue_type, f"Unknown issue type: {issue_type}")

    def get_cache_stats(self) -> Dict[str, float]:
        """Get cache statistics."""
        total = self._cache_hits + self._cache_misses
        if total == 0:
            return {"hits": 0, "misses": 0, "hit_rate": 0.0, "avg_build_time": 0.0}
        
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": self._cache_hits / total,
            "avg_build_time": sum(self._build_times) / len(self._build_times) if self._build_times else 0.0
        }
    
    def print_cache_stats(self):
        """Print cache statistics."""
        if not self.cache_stats:
            return
        
        stats = self.get_cache_stats()
        total = stats["hits"] + stats["misses"]
        if total > 0:
            print(f"\nCache Statistics:")
            print(f"  Hits: {stats['hits']}/{total} ({stats['hit_rate']*100:.1f}%)")
            print(f"  Misses: {stats['misses']}/{total} ({100-stats['hit_rate']*100:.1f}%)")
            if stats["avg_build_time"] > 0:
                print(f"  Avg build time: {stats['avg_build_time']:.2f}s")
    
    def __getitem__(self, idx: int) -> Data:
        """Get graph for sample at index, using cache if available."""
        row = self.df.iloc[idx]
        
        # Resolve PDB path
        pdb_path = Path(row["pdb_path"])
        if pdb_path.is_absolute():
            pass
        elif self.pdb_dir:
            pdb_path = self.pdb_dir / pdb_path
        else:
            if not pdb_path.exists():
                manifest_dir = self.manifest_csv.parent
                potential_path = manifest_dir / pdb_path
                if potential_path.exists():
                    pdb_path = potential_path
        
        pdb_path = pdb_path.resolve()
        
        # Get target value
        y_val = row["y"]
        if pd.isna(y_val):
            raise ValueError(f"Row {idx}: Target value 'y' is NaN")
        y = float(y_val)
        
        # Get chain assignments
        antibody_chains = self.antibody_chains_list[idx]
        antigen_chains = self.antigen_chains_list[idx]
        
        # Infer antigen chains if needed
        if antigen_chains is None:
            from Bio.PDB import PDBParser
            parser = PDBParser(QUIET=True)
            try:
                structure = parser.get_structure("complex", str(pdb_path))
                all_chain_ids = [chain.id for chain in structure[0].get_chains()]
                antigen_chains = [cid for cid in all_chain_ids if cid not in antibody_chains]
                if not antigen_chains:
                    warnings.warn(
                        f"Row {idx}: Could not infer antigen chains. "
                        f"Found chains: {all_chain_ids}, antibody chains: {antibody_chains}"
                    )
                    antigen_chains = all_chain_ids[len(antibody_chains):] if len(all_chain_ids) > len(antibody_chains) else []
            except Exception as e:
                warnings.warn(f"Row {idx}: Error inferring antigen chains: {e}")
                antigen_chains = []
        
        # Create preprocessing config
        preprocess_config = PreprocessConfig(
            pdb_path=str(pdb_path),
            antibody_chains=antibody_chains,
            antigen_chains=antigen_chains,
            noncovalent_cutoff=self.noncovalent_cutoff,
            interface_cutoff=self.interface_cutoff,
            use_covalent_edges=self.use_covalent_edges,
            use_noncovalent_edges=self.use_noncovalent_edges,
            allow_duplicate_edges=self.allow_duplicate_edges,
            include_residue_index=self.include_residue_index,
            add_interface_features_to_x=self.add_interface_features_to_x
        )
        
        # Generate cache key
        cache_key = make_cache_key(str(pdb_path), preprocess_config, self.hash_pdb_contents)
        cache_path = get_cache_path(self.cache_dir, cache_key)
        
        # Try to load from cache
        # #region agent log
        import json
        import time
        from pathlib import Path as PathLib
        log_path = PathLib("/home/wenkanl2/Ken/GAT_regression/.cursor/debug.log")
        def log_debug(location, message, data, hypothesis_id=None):
            entry = {
                "sessionId": "debug-session",
                "runId": "pre-fix",
                "location": location,
                "message": message,
                "data": data,
                "timestamp": int(time.time() * 1000)
            }
            try:
                with open(log_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
            except:
                pass  # Ignore logging errors

        log_debug("dataset.py:__getitem__", "Processing sample", {
            "idx": idx,
            "pdb_path": str(pdb_path),
            "antibody_chains": antibody_chains,
            "antigen_chains": antigen_chains,
            "cache_path": str(cache_path),
            "cache_exists": cache_path.exists()
        }, "D")
        # #endregion agent log

        if not self.rebuild_cache and cache_path.exists():
            try:
                # Verify cache metadata matches config
                if verify_cache_metadata(cache_path, preprocess_config):
                    data, metadata = load_graph_from_cache(cache_path, device=None)
                    
                    # Override y value from manifest (in case it changed)
                    data.y = torch.tensor([y], dtype=torch.float32)
                    
                    # Apply transform
                    data = self.transform(data)
                    
                    self._cache_hits += 1
                    return data
                else:
                    # Cache exists but config doesn't match - rebuild
                    warnings.warn(f"Cache config mismatch for {pdb_path}, rebuilding...")
            except Exception as e:
                warnings.warn(f"Error loading cache for {pdb_path}: {e}, rebuilding...")
        
        # Cache miss or rebuild requested - build graph from PDB
        self._cache_misses += 1
        build_start = time.time()
        
        try:
            # #region agent log
            log_debug("dataset.py:__getitem__", "Before pdb_to_graph call", {
                "idx": idx,
                "pdb_path": str(pdb_path),
                "antibody_chains": antibody_chains,
                "antigen_chains": antigen_chains
            }, "B")
            # #endregion agent log

            data = pdb_to_graph(
                pdb_path=str(pdb_path),
                antibody_chains=antibody_chains,
                antigen_chains=antigen_chains,
                noncovalent_cutoff=self.noncovalent_cutoff,
                interface_cutoff=self.interface_cutoff,
                use_covalent_edges=self.use_covalent_edges,
                use_noncovalent_edges=self.use_noncovalent_edges,
                allow_duplicate_edges=self.allow_duplicate_edges,
                include_residue_index=self.include_residue_index,
                add_interface_features_to_x=self.add_interface_features_to_x,
                y=y
            )

            # #region agent log
            log_debug("dataset.py:__getitem__", "pdb_to_graph call succeeded", {
                "idx": idx,
                "pdb_path": str(pdb_path),
                "num_nodes": int(data.num_nodes),
                "num_edges": int(data.edge_index.shape[1])
            }, "B")
            # #endregion agent log

        except Exception as e:
            # #region agent log
            log_debug("dataset.py:__getitem__", "pdb_to_graph call failed", {
                "idx": idx,
                "pdb_path": str(pdb_path),
                "error_type": str(type(e).__name__),
                "error_message": str(e)
            }, "B")
            # #endregion agent log
            raise RuntimeError(f"Error processing {pdb_path}: {e}")
        
        build_time = time.time() - build_start
        self._build_times.append(build_time)
        
        # Assertions
        assert data.num_nodes > 0, f"Graph has 0 nodes: {pdb_path}"
        assert data.edge_index.shape[0] == 2, f"edge_index must be [2, E], got {data.edge_index.shape}"
        assert data.edge_index.shape[1] > 0, f"Graph has 0 edges: {pdb_path}"
        assert data.x.shape[0] == data.num_nodes, f"Node count mismatch"
        assert not torch.isnan(data.x).any(), f"NaN in node features: {pdb_path}"
        assert data.edge_attr is None or not torch.isnan(data.edge_attr).any(), f"NaN in edge attributes: {pdb_path}"
        assert data.y is not None and data.y.dtype == torch.float32, f"Invalid y: {pdb_path}"
        assert not torch.isnan(data.y), f"y is NaN: {pdb_path}"
        
        # Save to cache (atomic write)
        try:
            # Count edge types for metadata (use edge_type if available, else edge_attr)
            if hasattr(data, 'edge_type') and data.edge_type is not None:
                edge_types = data.edge_type.unique().int().tolist()
                edge_type_counts = {int(et): int((data.edge_type == et).sum()) for et in edge_types}
            elif data.edge_attr is not None and data.edge_attr.shape[1] > 0:
                edge_types = data.edge_attr[:, 0].unique().int().tolist()
                edge_type_counts = {int(et): int((data.edge_attr[:, 0] == et).sum()) for et in edge_types}
            else:
                edge_type_counts = {}
            
            # Build chain mapping for metadata (use string keys for JSON compatibility)
            chain_mapping = {}
            if hasattr(data, 'chain_labels') and data.chain_labels is not None:
                chain_labels = data.chain_labels if isinstance(data.chain_labels, list) else data.chain_labels.tolist()
                for chain_idx, chain_label in enumerate(chain_labels):
                    # Determine role (0=antibody, 1=antigen)
                    role = 0 if chain_label in antibody_chains else 1
                    chain_mapping[str(chain_idx)] = {  # String key for JSON
                        "label": chain_label,
                        "role": role
                    }
            else:
                # Fallback: infer from chain_id and chain_group if available
                if hasattr(data, 'chain_id') and hasattr(data, 'chain_group'):
                    unique_chain_ids = data.chain_id.unique().int().tolist()
                    for chain_idx in unique_chain_ids:
                        # Get role from first node with this chain_id
                        node_mask = data.chain_id == chain_idx
                        if node_mask.any():
                            role = int(data.chain_group[node_mask][0].item())
                            # Try to get label from config or use index
                            chain_label = f"Chain{chain_idx}"
                            chain_mapping[str(chain_idx)] = {
                                "label": chain_label,
                                "role": role
                            }
            
            metadata = {
                "pdb_path": str(pdb_path),
                "config": preprocess_config.to_dict(),
                "created": datetime.now().isoformat(),
                "num_nodes": int(data.num_nodes),
                "num_edges": int(data.edge_index.shape[1]),
                "edge_type_counts": edge_type_counts,
                "chain_mapping": chain_mapping
            }
            
            save_graph_to_cache(data, cache_path, metadata)
        except Exception as e:
            warnings.warn(f"Failed to save cache for {pdb_path}: {e}")
        
        # Apply transform
        data = self.transform(data)
        
        return data


# Alias for backward compatibility
AntibodyAntigenDataset = CachedGraphDataset
