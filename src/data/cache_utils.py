"""
Utilities for caching PDB-to-graph conversions.
"""
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import torch
from torch_geometric.data import Data


@dataclass
class PreprocessConfig:
    """Configuration for graph preprocessing (used in cache key)."""
    pdb_path: str
    antibody_chains: list
    antigen_chains: list
    bound_cutoff: float
    unbound_cutoff: float
    use_sequential_edges: bool
    include_residue_index: bool
    
    def to_dict(self) -> dict:
        """Convert to dictionary for hashing."""
        return {
            "pdb_path": str(self.pdb_path),
            "antibody_chains": sorted(self.antibody_chains) if self.antibody_chains else None,
            "antigen_chains": sorted(self.antigen_chains) if self.antigen_chains else None,
            "bound_cutoff": float(self.bound_cutoff),
            "unbound_cutoff": float(self.unbound_cutoff),
            "use_sequential_edges": bool(self.use_sequential_edges),
            "include_residue_index": bool(self.include_residue_index),
        }


def make_cache_key(
    pdb_path: str,
    config: PreprocessConfig,
    hash_pdb_contents: bool = False
) -> str:
    """
    Generate a cache key for a PDB file and preprocessing config.
    
    Args:
        pdb_path: Path to PDB file
        config: PreprocessConfig object
        hash_pdb_contents: If True, hash file contents; else use mtime+size
    
    Returns:
        Cache key (hex string)
    """
    pdb_path = Path(pdb_path).resolve()
    
    # Get file metadata
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")
    
    stat = pdb_path.stat()
    
    # Build hash input
    hash_input = {
        "config": config.to_dict(),
    }
    
    if hash_pdb_contents:
        # Hash file contents (slower but more robust)
        with open(pdb_path, "rb") as f:
            file_hash = hashlib.sha1(f.read()).hexdigest()
        hash_input["file_hash"] = file_hash
    else:
        # Use mtime + size (faster)
        hash_input["mtime"] = stat.st_mtime
        hash_input["size"] = stat.st_size
    
    # Create deterministic JSON string
    json_str = json.dumps(hash_input, sort_keys=True)
    
    # Hash the JSON string
    cache_key = hashlib.sha1(json_str.encode()).hexdigest()
    
    return cache_key


def get_cache_path(cache_dir: Path, cache_key: str) -> Path:
    """Get cache file path from cache key."""
    # Use first 2 chars for subdirectory to avoid too many files in one dir
    subdir = cache_dir / cache_key[:2]
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir / f"{cache_key}.pt"


def get_metadata_path(cache_path: Path) -> Path:
    """Get metadata JSON path for a cache file."""
    return cache_path.with_suffix(".meta.json")


def save_graph_to_cache(
    graph: Data,
    cache_path: Path,
    metadata: Dict[str, Any]
) -> None:
    """
    Save graph and metadata to cache with atomic write.
    
    Args:
        graph: PyTorch Geometric Data object
        cache_path: Path to save graph
        metadata: Metadata dictionary to save as JSON
    """
    # Atomic write: write to temp file, then rename
    temp_path = cache_path.with_suffix(".tmp")
    meta_path = get_metadata_path(cache_path)
    meta_temp_path = meta_path.with_suffix(".tmp")
    
    try:
        # Save graph
        torch.save(graph, temp_path, _use_new_zipfile_serialization=False)
        os.replace(temp_path, cache_path)
        
        # Save metadata
        with open(meta_temp_path, "w") as f:
            json.dump(metadata, f, indent=2)
        os.replace(meta_temp_path, meta_path)
    except Exception as e:
        # Clean up temp files on error
        if temp_path.exists():
            temp_path.unlink()
        if meta_temp_path.exists():
            meta_temp_path.unlink()
        raise e


def load_graph_from_cache(
    cache_path: Path,
    device: Optional[str] = None
) -> tuple:
    """
    Load graph and metadata from cache.
    
    Args:
        cache_path: Path to cache file
        device: Device to load graph on (None = CPU)
    
    Returns:
        (graph, metadata) tuple
    """
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")
    
    # Load graph (always on CPU first)
    # weights_only=False needed for PyTorch 2.6+ when loading Data objects
    graph = torch.load(cache_path, map_location="cpu", weights_only=False)
    
    # Load metadata
    meta_path = get_metadata_path(cache_path)
    if meta_path.exists():
        with open(meta_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    # Move to device if specified
    if device and device != "cpu":
        graph = graph.to(device)
    
    return graph, metadata


def verify_cache_metadata(
    cache_path: Path,
    expected_config: PreprocessConfig
) -> bool:
    """
    Verify that cached graph matches expected preprocessing config.
    
    Args:
        cache_path: Path to cache file
        expected_config: Expected preprocessing config
    
    Returns:
        True if cache matches, False otherwise
    """
    meta_path = get_metadata_path(cache_path)
    if not meta_path.exists():
        return False
    
    try:
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        
        cached_config = metadata.get("config", {})
        expected_dict = expected_config.to_dict()
        
        # Compare config fields
        for key in expected_dict:
            if key not in cached_config:
                return False
            if cached_config[key] != expected_dict[key]:
                return False
        
        return True
    except Exception:
        return False

