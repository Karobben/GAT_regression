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
        
        # Use multiprocessing if requested
        if num_workers is None or (num_workers > 0 and len(self) > 1):
            import multiprocessing as mp
            if num_workers is None:
                num_workers = mp.cpu_count()
            
            if verbose:
                print(f"Using {num_workers} workers for parallel processing...")
            
            # Prepare worker arguments
            worker_args = []
            for idx in range(len(self)):
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
                        desc="Loading graphs"
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
                for idx in tqdm(range(len(self)), disable=not verbose, desc="Loading graphs"):
                    _ = self[idx]  # This will use cache or build it
            except ImportError:
                # tqdm not available, use simple loop
                for idx in range(len(self)):
                    if verbose and idx % 10 == 0:
                        print(f"Loading graph {idx}/{len(self)}...")
                    _ = self[idx]
        
        if verbose:
            self.print_cache_stats()
    
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
