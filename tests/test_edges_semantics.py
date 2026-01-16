"""
Test edge semantics: covalent vs noncovalent edges.
"""
import numpy as np
import torch
from torch_geometric.data import Data
from src.data.pdb_to_graph import build_edges, compute_pairwise_distances


def test_covalent_edges_sequential():
    """Test that covalent edges connect sequential residues within chains."""
    # Create a simple chain of 5 residues
    N = 5
    coords = np.array([
        [0.0, 0.0, 0.0],   # Residue 0
        [3.8, 0.0, 0.0],   # Residue 1 (peptide bond distance ~3.8 Å)
        [7.6, 0.0, 0.0],   # Residue 2
        [11.4, 0.0, 0.0],  # Residue 3
        [15.2, 0.0, 0.0],  # Residue 4
    ])
    
    distances = compute_pairwise_distances(coords)
    
    # All residues in same chain
    chain_ids = ["A"] * N
    residue_info = [("A", i, " ") for i in range(1, N + 1)]  # resseq 1-5, no insertion codes
    
    # Build edges with only covalent edges
    edge_index, edge_attr, edge_type, edge_dist = build_edges(
        distances,
        chain_ids,
        residue_info,
        use_covalent_edges=True,
        use_noncovalent_edges=False,
        noncovalent_cutoff=10.0,
        allow_duplicate_edges=False
    )
    
    # Check that we have covalent edges connecting (0-1, 1-2, 2-3, 3-4)
    # Each edge is undirected, so we should have 8 edges total (4 pairs * 2 directions)
    assert edge_index.shape[1] == 8, f"Expected 8 edges (4 covalent pairs * 2 directions), got {edge_index.shape[1]}"
    assert (edge_type == 0).all(), "All edges should be covalent (type 0)"
    
    # Check that edges connect consecutive residues
    edge_pairs = set()
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        edge_pairs.add((min(src, dst), max(src, dst)))
    
    expected_pairs = {(0, 1), (1, 2), (2, 3), (3, 4)}
    assert edge_pairs == expected_pairs, f"Expected edges {expected_pairs}, got {edge_pairs}"
    
    print("✓ Covalent edges correctly connect sequential residues")


def test_noncovalent_edges_distance():
    """Test that noncovalent edges are added based on distance cutoff."""
    # Create 4 residues: 2 close pairs
    N = 4
    coords = np.array([
        [0.0, 0.0, 0.0],   # Residue 0
        [5.0, 0.0, 0.0],   # Residue 1 (5 Å away)
        [0.0, 8.0, 0.0],   # Residue 2 (8 Å away)
        [0.0, 12.0, 0.0],  # Residue 3 (12 Å away, beyond cutoff)
    ])
    
    distances = compute_pairwise_distances(coords)
    
    # All residues in same chain
    chain_ids = ["A"] * N
    residue_info = [("A", i, " ") for i in range(1, N + 1)]
    
    # Build edges with only noncovalent edges (no covalent)
    edge_index, edge_attr, edge_type, edge_dist = build_edges(
        distances,
        chain_ids,
        residue_info,
        use_covalent_edges=False,
        use_noncovalent_edges=True,
        noncovalent_cutoff=10.0,  # 10 Å cutoff
        allow_duplicate_edges=False
    )
    
    # Check edges: (0,1) at 5Å, (0,2) at 8Å should be included
    # (0,3) at 12Å should be excluded
    edge_pairs = set()
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        edge_pairs.add((min(src, dst), max(src, dst)))
    
    # Should have (0,1), (0,2), (1,2), (1,3), (2,3) within 10Å
    # But wait, let's check distances:
    # 0-1: 5.0 Å ✓
    # 0-2: 8.0 Å ✓
    # 0-3: 12.0 Å ✗
    # 1-2: sqrt(5^2 + 8^2) = sqrt(89) ≈ 9.4 Å ✓
    # 1-3: sqrt(5^2 + 12^2) = sqrt(169) = 13.0 Å ✗
    # 2-3: 4.0 Å ✓
    
    expected_pairs = {(0, 1), (0, 2), (1, 2), (2, 3)}
    assert edge_pairs == expected_pairs, f"Expected edges {expected_pairs}, got {edge_pairs}"
    assert (edge_type == 1).all(), "All edges should be noncovalent (type 1)"
    
    print("✓ Noncovalent edges correctly added based on distance cutoff")


def test_no_duplicate_edges():
    """Test that covalent pairs are excluded from noncovalent edges when allow_duplicate_edges=False."""
    # Create a chain where covalent edges are within noncovalent cutoff
    N = 3
    coords = np.array([
        [0.0, 0.0, 0.0],   # Residue 0
        [3.8, 0.0, 0.0],   # Residue 1 (covalent, 3.8 Å)
        [7.6, 0.0, 0.0],   # Residue 2 (covalent, 3.8 Å from 1)
    ])
    
    distances = compute_pairwise_distances(coords)
    
    chain_ids = ["A"] * N
    residue_info = [("A", i, " ") for i in range(1, N + 1)]
    
    # Build edges with both covalent and noncovalent
    edge_index, edge_attr, edge_type, edge_dist = build_edges(
        distances,
        chain_ids,
        residue_info,
        use_covalent_edges=True,
        use_noncovalent_edges=True,
        noncovalent_cutoff=10.0,
        allow_duplicate_edges=False  # Should exclude covalent pairs from noncovalent
    )
    
    # Count edge types
    num_covalent = (edge_type == 0).sum().item()
    num_noncovalent = (edge_type == 1).sum().item()
    
    # Covalent: (0,1) and (1,2) = 2 pairs * 2 directions = 4 edges
    # Noncovalent: (0,2) at 7.6 Å = 1 pair * 2 directions = 2 edges
    # Total: 6 edges
    assert edge_index.shape[1] == 6, f"Expected 6 edges, got {edge_index.shape[1]}"
    assert num_covalent == 4, f"Expected 4 covalent edges, got {num_covalent}"
    assert num_noncovalent == 2, f"Expected 2 noncovalent edges, got {num_noncovalent}"
    
    # Verify (0,1) and (1,2) are covalent, (0,2) is noncovalent
    edge_pairs_covalent = set()
    edge_pairs_noncovalent = set()
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        pair = (min(src, dst), max(src, dst))
        if edge_type[i] == 0:
            edge_pairs_covalent.add(pair)
        else:
            edge_pairs_noncovalent.add(pair)
    
    assert edge_pairs_covalent == {(0, 1), (1, 2)}, f"Covalent pairs should be {(0, 1), (1, 2)}, got {edge_pairs_covalent}"
    assert edge_pairs_noncovalent == {(0, 2)}, f"Noncovalent pairs should be {(0, 2)}, got {edge_pairs_noncovalent}"
    
    print("✓ Covalent pairs correctly excluded from noncovalent edges")


def test_allow_duplicate_edges():
    """Test that when allow_duplicate_edges=True, covalent pairs can also be noncovalent."""
    N = 3
    coords = np.array([
        [0.0, 0.0, 0.0],
        [3.8, 0.0, 0.0],
        [7.6, 0.0, 0.0],
    ])
    
    distances = compute_pairwise_distances(coords)
    chain_ids = ["A"] * N
    residue_info = [("A", i, " ") for i in range(1, N + 1)]
    
    # Build edges with duplicates allowed
    edge_index, edge_attr, edge_type, edge_dist = build_edges(
        distances,
        chain_ids,
        residue_info,
        use_covalent_edges=True,
        use_noncovalent_edges=True,
        noncovalent_cutoff=10.0,
        allow_duplicate_edges=True  # Allow duplicates
    )
    
    # Now (0,1) and (1,2) should appear as both covalent and noncovalent
    # Covalent: (0,1), (1,2) = 4 edges
    # Noncovalent: (0,1), (1,2), (0,2) = 6 edges
    # Total: 10 edges
    num_covalent = (edge_type == 0).sum().item()
    num_noncovalent = (edge_type == 1).sum().item()
    
    assert num_covalent == 4, f"Expected 4 covalent edges, got {num_covalent}"
    assert num_noncovalent == 6, f"Expected 6 noncovalent edges, got {num_noncovalent}"
    assert edge_index.shape[1] == 10, f"Expected 10 edges total, got {edge_index.shape[1]}"
    
    print("✓ Duplicate edges allowed when allow_duplicate_edges=True")


if __name__ == "__main__":
    print("Testing edge semantics...")
    test_covalent_edges_sequential()
    test_noncovalent_edges_distance()
    test_no_duplicate_edges()
    test_allow_duplicate_edges()
    print("\nAll edge semantics tests passed!")

