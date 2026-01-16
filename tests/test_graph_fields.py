"""
Test that graph Data objects have all required fields for visualization.
"""
import torch
from torch_geometric.data import Data
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.pdb_to_graph import pdb_to_graph


def test_graph_has_required_fields():
    """Test that pdb_to_graph returns Data with all required visualization fields."""
    # Create a minimal test PDB file (or use existing test data)
    # For now, we'll test the structure of the returned Data object
    
    # This test requires an actual PDB file, so we'll make it conditional
    test_pdb = Path("tests/test_data/test.pdb")
    
    if not test_pdb.exists():
        print("Skipping test: test PDB file not found")
        return
    
    # Generate graph
    data = pdb_to_graph(
        pdb_path=str(test_pdb),
        antibody_chains=["H", "L"],
        antigen_chains=["A"],
        bound_cutoff=8.0,
        unbound_cutoff=10.0,
        use_sequential_edges=False,
        include_residue_index=True,
        y=5.0
    )
    
    # Required tensor fields
    required_tensor_fields = {
        'pos': (torch.float32, 2),  # [N, 3]
        'chain_id': (torch.int64, 1),  # [N]
        'chain_group': (torch.int64, 1),  # [N] (0=antibody, 1=antigen)
        'res_id': (torch.int64, 1),  # [N]
        'aa': (torch.int64, 1),  # [N]
        'edge_type': (torch.int64, 1),  # [E]
        'edge_dist': (torch.float32, 1),  # [E]
    }
    
    # Check all required fields exist
    for field_name, (expected_dtype, expected_dims) in required_tensor_fields.items():
        assert hasattr(data, field_name), f"Missing field: {field_name}"
        field_value = getattr(data, field_name)
        assert isinstance(field_value, torch.Tensor), f"{field_name} is not a tensor"
        assert field_value.dtype == expected_dtype, f"{field_name} has wrong dtype: {field_value.dtype} != {expected_dtype}"
        assert len(field_value.shape) == expected_dims, f"{field_name} has wrong number of dimensions"
    
    # Check pos shape
    assert data.pos.shape == (data.num_nodes, 3), f"pos shape mismatch: {data.pos.shape} != ({data.num_nodes}, 3)"
    
    # Check chain_id shape
    assert data.chain_id.shape == (data.num_nodes,), f"chain_id shape mismatch"
    
    # Check chain_group values are 0 or 1
    assert data.chain_group.min() >= 0 and data.chain_group.max() <= 1, "chain_group must be 0 or 1"
    
    # Check edge_type values are 0, 1, or 2
    assert data.edge_type.min() >= 0 and data.edge_type.max() <= 2, "edge_type must be 0, 1, or 2"
    
    # Check edge_dist is positive
    assert (data.edge_dist >= 0).all(), "edge_dist must be non-negative"
    
    # Check edge_type and edge_dist match edge_index
    num_edges = data.edge_index.shape[1]
    assert data.edge_type.shape[0] == num_edges, f"edge_type length mismatch: {data.edge_type.shape[0]} != {num_edges}"
    assert data.edge_dist.shape[0] == num_edges, f"edge_dist length mismatch: {data.edge_dist.shape[0]} != {num_edges}"
    
    # Check chain_labels exists (can be list or attribute)
    assert hasattr(data, 'chain_labels'), "Missing chain_labels"
    
    print("✓ All required graph fields are present and correctly formatted")


def test_graph_can_be_saved_and_loaded():
    """Test that graph with visualization fields can be saved and loaded."""
    test_pdb = Path("tests/test_data/test.pdb")
    
    if not test_pdb.exists():
        print("Skipping test: test PDB file not found")
        return
    
    # Generate graph
    data = pdb_to_graph(
        pdb_path=str(test_pdb),
        antibody_chains=["H", "L"],
        antigen_chains=["A"],
        y=5.0
    )
    
    # Save
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        temp_path = Path(f.name)
        torch.save(data, temp_path)
    
    try:
        # Load
        loaded_data = torch.load(temp_path, map_location="cpu", weights_only=False)
        
        # Verify fields are preserved
        assert hasattr(loaded_data, 'pos')
        assert hasattr(loaded_data, 'chain_id')
        assert hasattr(loaded_data, 'chain_group')
        assert hasattr(loaded_data, 'edge_type')
        assert hasattr(loaded_data, 'edge_dist')
        
        print("✓ Graph can be saved and loaded with all fields")
    finally:
        temp_path.unlink()


if __name__ == "__main__":
    test_graph_has_required_fields()
    test_graph_can_be_saved_and_loaded()
    print("All tests passed!")

