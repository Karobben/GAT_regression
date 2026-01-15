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
        bound_cutoff: float = 8.0,
        unbound_cutoff: float = 10.0,
        use_sequential_edges: bool = False,
        include_residue_index: bool = True,
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
            bound_cutoff: Distance cutoff for BOUND edges
            unbound_cutoff: Distance cutoff for UNBOUND edges
            use_sequential_edges: Whether to include sequential edges
            include_residue_index: Whether to include residue index in node features
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
        self.bound_cutoff = bound_cutoff
        self.unbound_cutoff = unbound_cutoff
        self.use_sequential_edges = use_sequential_edges
        self.include_residue_index = include_residue_index
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
    
    def preload_all(self, verbose: bool = True):
        """
        Preload all graphs (builds cache if needed).
        
        With disk caching, this will:
        - Load from cache if available
        - Build and save to cache if not available
        
        Args:
            verbose: Whether to print progress
        """
        if verbose:
            print(f"Preloading {len(self)} graphs (will use cache if available)...")
        
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
            bound_cutoff=self.bound_cutoff,
            unbound_cutoff=self.unbound_cutoff,
            use_sequential_edges=self.use_sequential_edges,
            include_residue_index=self.include_residue_index
        )
        
        # Generate cache key
        cache_key = make_cache_key(str(pdb_path), preprocess_config, self.hash_pdb_contents)
        cache_path = get_cache_path(self.cache_dir, cache_key)
        
        # Try to load from cache
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
            data = pdb_to_graph(
                pdb_path=str(pdb_path),
                antibody_chains=antibody_chains,
                antigen_chains=antigen_chains,
                bound_cutoff=self.bound_cutoff,
                unbound_cutoff=self.unbound_cutoff,
                use_sequential_edges=self.use_sequential_edges,
                include_residue_index=self.include_residue_index,
                y=y
            )
        except Exception as e:
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
            # Count edge types for metadata
            if data.edge_attr is not None and data.edge_attr.shape[1] > 0:
                edge_types = data.edge_attr[:, 0].unique().int().tolist()
                edge_type_counts = {int(et): int((data.edge_attr[:, 0] == et).sum()) for et in edge_types}
            else:
                edge_type_counts = {}
            
            metadata = {
                "pdb_path": str(pdb_path),
                "config": preprocess_config.to_dict(),
                "created": datetime.now().isoformat(),
                "num_nodes": int(data.num_nodes),
                "num_edges": int(data.edge_index.shape[1]),
                "edge_type_counts": edge_type_counts
            }
            
            save_graph_to_cache(data, cache_path, metadata)
        except Exception as e:
            warnings.warn(f"Failed to save cache for {pdb_path}: {e}")
        
        # Apply transform
        data = self.transform(data)
        
        return data


# Alias for backward compatibility
AntibodyAntigenDataset = CachedGraphDataset
