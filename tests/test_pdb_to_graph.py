"""
Unit tests for PDB to graph conversion.
"""
import numpy as np
import torch
from pathlib import Path
import tempfile
import os

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.pdb_to_graph import (
    pdb_to_graph,
    get_residue_type,
    compute_pairwise_distances,
    build_node_features,
    build_edges
)


def test_get_residue_type():
    """Test residue type encoding."""
    from Bio.PDB import Residue
    
    # Mock residue
    class MockResidue:
        def __init__(self, resname):
            self._resname = resname
        def get_resname(self):
            return self._resname
    
    assert get_residue_type(MockResidue("ALA")) == 0
    assert get_residue_type(MockResidue("GLY")) == 7
    assert get_residue_type(MockResidue("UNK")) == 20  # Unknown


def test_compute_pairwise_distances():
    """Test pairwise distance computation."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    
    distances = compute_pairwise_distances(coords)
    
    assert distances.shape == (3, 3)
    assert np.isclose(distances[0, 1], 1.0)
    assert np.isclose(distances[0, 2], 1.0)
    assert np.isclose(distances[1, 2], np.sqrt(2.0))
    assert np.allclose(distances, distances.T)  # Symmetric


def test_build_node_features():
    """Test node feature construction."""
    residue_types = [0, 7, 19]  # ALA, GLY, VAL
    chain_ids = ["H", "H", "A"]
    antibody_chains = ["H"]
    
    features = build_node_features(
        residue_types,
        chain_ids,
        antibody_chains,
        include_residue_index=True
    )
    
    assert features.shape[0] == 3
    assert features.shape[1] == 23  # 21 (AA one-hot) + 1 (chain) + 1 (index)
    
    # Check chain type
    assert features[0, 21] == 1.0  # H is antibody
    assert features[1, 21] == 1.0  # H is antibody
    assert features[2, 21] == 0.0  # A is antigen
    
    # Check residue type one-hot
    assert features[0, 0] == 1.0  # ALA
    assert features[1, 7] == 1.0  # GLY
    assert features[2, 19] == 1.0  # VAL


def test_build_edges():
    """Test edge construction."""
    # Create a simple distance matrix
    N = 4
    distances = np.array([
        [0.0, 5.0, 7.0, 12.0],  # Node 0
        [5.0, 0.0, 6.0, 11.0],  # Node 1
        [7.0, 6.0, 0.0, 9.0],   # Node 2
        [12.0, 11.0, 9.0, 0.0]  # Node 3
    ])
    
    chain_ids = ["H", "H", "A", "A"]
    antibody_chains = ["H"]
    antigen_chains = ["A"]
    
    edge_index, edge_attr = build_edges(
        distances,
        chain_ids,
        antibody_chains,
        antigen_chains,
        bound_cutoff=8.0,
        unbound_cutoff=10.0,
        use_sequential_edges=False
    )
    
    assert edge_index.shape[0] == 2
    assert edge_attr.shape[0] == edge_index.shape[1]
    assert edge_attr.shape[1] == 2  # [edge_type, normalized_distance]
    
    # Check that we have some edges
    assert edge_index.shape[1] > 0
    
    # Check edge types are valid
    edge_types = edge_attr[:, 0].long()
    assert torch.all(edge_types >= 0)
    assert torch.all(edge_types <= 1)  # BOUND (0) or UNBOUND (1)


def test_pdb_to_graph_synthetic():
    """Test PDB to graph conversion with a minimal synthetic PDB."""
    # Create a minimal PDB file
    pdb_content = """HEADER    TEST COMPLEX
ATOM      1  CA  ALA H   1      20.154  16.967  21.532  1.00 30.00           C
ATOM      2  CA  GLY H   2      21.154  17.967  22.532  1.00 30.00           C
ATOM      3  CA  VAL A   1      22.154  18.967  23.532  1.00 30.00           C
END
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write(pdb_content)
        pdb_path = f.name
    
    try:
        data = pdb_to_graph(
            pdb_path=pdb_path,
            antibody_chains=["H"],
            antigen_chains=["A"],
            bound_cutoff=8.0,
            unbound_cutoff=10.0,
            use_sequential_edges=False,
            include_residue_index=True,
            y=5.0
        )
        
        # Check basic properties
        assert data.x.shape[0] == 3  # 3 residues
        assert data.edge_index.shape[0] == 2
        assert data.y is not None
        assert data.y.item() == 5.0
        
        # Check node features
        assert data.x.shape[1] >= 22  # At least 21 (AA) + 1 (chain)
        
    finally:
        os.unlink(pdb_path)


if __name__ == "__main__":
    print("Running tests...")
    test_get_residue_type()
    print("✓ test_get_residue_type")
    
    test_compute_pairwise_distances()
    print("✓ test_compute_pairwise_distances")
    
    test_build_node_features()
    print("✓ test_build_node_features")
    
    test_build_edges()
    print("✓ test_build_edges")
    
    test_pdb_to_graph_synthetic()
    print("✓ test_pdb_to_graph_synthetic")
    
    print("\nAll tests passed!")

