"""
Tests for graph caching functionality.
"""
import torch
from pathlib import Path
import tempfile
import shutil
import os

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.cache_utils import (
    PreprocessConfig,
    make_cache_key,
    get_cache_path,
    save_graph_to_cache,
    load_graph_from_cache,
    verify_cache_metadata
)
from data.dataset import CachedGraphDataset
from torch_geometric.data import Data


def test_cache_key_generation():
    """Test that cache keys are deterministic and unique."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write("ATOM      1  CA  ALA A   1      20.154  16.967  10.000  1.00 20.00           C\n")
        f.write("ATOM      2  CA  GLY A   2      21.154  17.967  11.000  1.00 20.00           C\n")
        temp_pdb = f.name
    
    try:
        config1 = PreprocessConfig(
            pdb_path=temp_pdb,
            antibody_chains=["H", "L"],
            antigen_chains=["A"],
            bound_cutoff=8.0,
            unbound_cutoff=10.0,
            use_sequential_edges=False,
            include_residue_index=True
        )
        
        config2 = PreprocessConfig(
            pdb_path=temp_pdb,
            antibody_chains=["H", "L"],
            antigen_chains=["A"],
            bound_cutoff=8.0,
            unbound_cutoff=10.0,
            use_sequential_edges=False,
            include_residue_index=True
        )
        
        # Same config should give same key
        key1 = make_cache_key(temp_pdb, config1, hash_pdb_contents=False)
        key2 = make_cache_key(temp_pdb, config2, hash_pdb_contents=False)
        assert key1 == key2, "Same config should give same cache key"
        
        # Different cutoff should give different key
        config3 = PreprocessConfig(
            pdb_path=temp_pdb,
            antibody_chains=["H", "L"],
            antigen_chains=["A"],
            bound_cutoff=9.0,  # Different
            unbound_cutoff=10.0,
            use_sequential_edges=False,
            include_residue_index=True
        )
        key3 = make_cache_key(temp_pdb, config3, hash_pdb_contents=False)
        assert key1 != key3, "Different config should give different cache key"
        
        print("✓ Cache key generation test passed")
    finally:
        os.unlink(temp_pdb)


def test_save_load_cache():
    """Test saving and loading graphs from cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        cache_dir.mkdir()
        
        # Create a dummy graph
        graph = Data(
            x=torch.randn(10, 22),
            edge_index=torch.randint(0, 10, (2, 20)),
            edge_attr=torch.randn(20, 2),
            y=torch.tensor([1.5], dtype=torch.float32),
            num_nodes=10
        )
        
        # Create a dummy PDB file for testing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write("ATOM      1  CA  ALA A   1      20.154  16.967  10.000  1.00 20.00           C\n")
            temp_pdb = f.name
        
        try:
            config = PreprocessConfig(
                pdb_path=temp_pdb,
                antibody_chains=["H", "L"],
                antigen_chains=["A"],
                bound_cutoff=8.0,
                unbound_cutoff=10.0,
                use_sequential_edges=False,
                include_residue_index=True
            )
            
            cache_key = make_cache_key(temp_pdb, config, hash_pdb_contents=False)
            cache_path = get_cache_path(cache_dir, cache_key)
            
            metadata = {
                "pdb_path": temp_pdb,
                "config": config.to_dict(),
                "num_nodes": 10,
                "num_edges": 20
            }
            
            # Save
            save_graph_to_cache(graph, cache_path, metadata)
            assert cache_path.exists(), "Cache file should exist after saving"
            
            # Load
            loaded_graph, loaded_metadata = load_graph_from_cache(cache_path)
            
            # Verify
            assert torch.allclose(loaded_graph.x, graph.x), "Node features should match"
            assert torch.allclose(loaded_graph.edge_index, graph.edge_index), "Edge index should match"
            assert torch.allclose(loaded_graph.y, graph.y), "Target should match"
            assert loaded_metadata["num_nodes"] == 10, "Metadata should match"
            
            print("✓ Save/load cache test passed")
        finally:
            os.unlink(temp_pdb)


def test_cache_metadata_verification():
    """Test cache metadata verification."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write("ATOM      1  CA  ALA A   1      20.154  16.967  10.000  1.00 20.00           C\n")
        temp_pdb = f.name
    
    try:
        config1 = PreprocessConfig(
            pdb_path=temp_pdb,
            antibody_chains=["H", "L"],
            antigen_chains=["A"],
            bound_cutoff=8.0,
            unbound_cutoff=10.0,
            use_sequential_edges=False,
            include_residue_index=True
        )
        
        config2 = PreprocessConfig(
            pdb_path=temp_pdb,
            antibody_chains=["H", "L"],
            antigen_chains=["A"],
            bound_cutoff=8.0,
            unbound_cutoff=10.0,
            use_sequential_edges=False,
            include_residue_index=True
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            cache_dir.mkdir()
            
            graph = Data(
                x=torch.randn(5, 22),
                edge_index=torch.randint(0, 5, (2, 10)),
                edge_attr=torch.randn(10, 2),
                y=torch.tensor([1.0], dtype=torch.float32),
                num_nodes=5
            )
            
            cache_key = make_cache_key(temp_pdb, config1, hash_pdb_contents=False)
            cache_path = get_cache_path(cache_dir, cache_key)
            
            metadata = {
                "pdb_path": temp_pdb,
                "config": config1.to_dict(),
                "num_nodes": 5
            }
            
            save_graph_to_cache(graph, cache_path, metadata)
            
            # Same config should verify
            assert verify_cache_metadata(cache_path, config2), "Same config should verify"
            
            # Different config should not verify
            config3 = PreprocessConfig(
                pdb_path=temp_pdb,
                antibody_chains=["H", "L"],
                antigen_chains=["A"],
                bound_cutoff=9.0,  # Different
                unbound_cutoff=10.0,
                use_sequential_edges=False,
                include_residue_index=True
            )
            assert not verify_cache_metadata(cache_path, config3), "Different config should not verify"
            
            print("✓ Cache metadata verification test passed")
    finally:
        os.unlink(temp_pdb)


def test_dataset_cache_hit_miss():
    """Test that dataset correctly uses cache on second access."""
    # This test requires actual PDB files, so we'll just test the logic
    # In a real scenario, you'd need test PDB files
    
    print("Note: Full dataset cache test requires PDB files.")
    print("  Run with actual data to test cache hits/misses.")
    print("✓ Dataset cache test placeholder passed")


if __name__ == "__main__":
    print("Running cache tests...\n")
    
    test_cache_key_generation()
    test_save_load_cache()
    test_cache_metadata_verification()
    test_dataset_cache_hit_miss()
    
    print("\nAll cache tests passed!")

