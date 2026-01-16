"""
Test interface marker computation.
"""
import numpy as np
import torch
from src.data.pdb_to_graph import compute_interface_markers, compute_pairwise_distances


def test_interface_markers_simple():
    """Test interface markers with simple 2 antibody + 2 antigen setup."""
    # Create 4 nodes: 2 antibody, 2 antigen
    # Antibody nodes at (0,0,0) and (5,0,0)
    # Antigen nodes at (0,7,0) and (5,7,0)
    coords = np.array([
        [0.0, 0.0, 0.0],  # Antibody 0
        [5.0, 0.0, 0.0],  # Antibody 1
        [0.0, 7.0, 0.0],  # Antigen 0 (7 Å from antibody 0)
        [5.0, 7.0, 0.0],  # Antigen 1 (7 Å from antibody 1, ~8.6 Å from antibody 0)
    ])
    
    distances = compute_pairwise_distances(coords)
    chain_role = np.array([0, 0, 1, 1])  # 0=antibody, 1=antigen
    interface_cutoff = 8.0  # 8 Å cutoff
    
    min_inter_dist, inter_contact_count, is_interface = compute_interface_markers(
        distances, chain_role, interface_cutoff
    )
    
    # Antibody 0: min distance to antigen = 7.0 Å (to antigen 0) ✓ within cutoff
    # Antibody 1: min distance to antigen = 7.0 Å (to antigen 1) ✓ within cutoff
    # Antigen 0: min distance to antibody = 7.0 Å (to antibody 0) ✓ within cutoff
    # Antigen 1: min distance to antibody = 7.0 Å (to antibody 1) ✓ within cutoff
    
    assert is_interface[0] == 1, "Antibody 0 should be interface"
    assert is_interface[1] == 1, "Antibody 1 should be interface"
    assert is_interface[2] == 1, "Antigen 0 should be interface"
    assert is_interface[3] == 1, "Antigen 1 should be interface"
    
    # Check min_inter_dist
    assert abs(min_inter_dist[0].item() - 7.0) < 0.1, f"Antibody 0 min_inter_dist should be ~7.0, got {min_inter_dist[0]}"
    assert abs(min_inter_dist[1].item() - 7.0) < 0.1, f"Antibody 1 min_inter_dist should be ~7.0, got {min_inter_dist[1]}"
    
    # Check inter_contact_count (each should have 1 contact within 8 Å)
    assert inter_contact_count[0].item() == 1.0, f"Antibody 0 should have 1 contact, got {inter_contact_count[0]}"
    assert inter_contact_count[1].item() == 1.0, f"Antibody 1 should have 1 contact, got {inter_contact_count[1]}"
    
    print("✓ Interface markers correctly computed for simple case")


def test_interface_markers_outside_cutoff():
    """Test that nodes outside cutoff are not marked as interface."""
    # Create nodes far apart
    coords = np.array([
        [0.0, 0.0, 0.0],   # Antibody 0
        [5.0, 0.0, 0.0],   # Antibody 1
        [0.0, 15.0, 0.0],  # Antigen 0 (15 Å away, beyond cutoff)
        [5.0, 15.0, 0.0],  # Antigen 1 (15 Å away, beyond cutoff)
    ])
    
    distances = compute_pairwise_distances(coords)
    chain_role = np.array([0, 0, 1, 1])
    interface_cutoff = 8.0
    
    min_inter_dist, inter_contact_count, is_interface = compute_interface_markers(
        distances, chain_role, interface_cutoff
    )
    
    # All nodes should be outside cutoff
    assert is_interface[0] == 0, "Antibody 0 should NOT be interface (too far)"
    assert is_interface[1] == 0, "Antibody 1 should NOT be interface (too far)"
    assert is_interface[2] == 0, "Antigen 0 should NOT be interface (too far)"
    assert is_interface[3] == 0, "Antigen 1 should NOT be interface (too far)"
    
    # min_inter_dist should be ~15.0 (clamped to 2 * interface_cutoff = 16.0)
    assert min_inter_dist[0].item() <= 16.0, f"min_inter_dist should be <= 16.0, got {min_inter_dist[0]}"
    assert inter_contact_count[0].item() == 0.0, f"inter_contact_count should be 0, got {inter_contact_count[0]}"
    
    print("✓ Nodes outside cutoff correctly marked as non-interface")


def test_interface_markers_multiple_contacts():
    """Test that inter_contact_count counts all contacts within cutoff."""
    # Create 1 antibody node and 3 antigen nodes
    # Antibody at (0,0,0)
    # Antigens at distances: 5 Å, 7 Å, 9 Å (only first 2 within 8 Å cutoff)
    coords = np.array([
        [0.0, 0.0, 0.0],   # Antibody 0
        [5.0, 0.0, 0.0],   # Antigen 0 (5 Å)
        [7.0, 0.0, 0.0],   # Antigen 1 (7 Å)
        [9.0, 0.0, 0.0],   # Antigen 2 (9 Å, beyond cutoff)
    ])
    
    distances = compute_pairwise_distances(coords)
    chain_role = np.array([0, 1, 1, 1])
    interface_cutoff = 8.0
    
    min_inter_dist, inter_contact_count, is_interface = compute_interface_markers(
        distances, chain_role, interface_cutoff
    )
    
    # Antibody 0 should have 2 contacts (antigen 0 and 1)
    assert inter_contact_count[0].item() == 2.0, f"Antibody 0 should have 2 contacts, got {inter_contact_count[0]}"
    assert is_interface[0] == 1, "Antibody 0 should be interface"
    assert abs(min_inter_dist[0].item() - 5.0) < 0.1, f"min_inter_dist should be ~5.0, got {min_inter_dist[0]}"
    
    # Antigen 0 and 1 should be interface (within cutoff of antibody)
    assert is_interface[1] == 1, "Antigen 0 should be interface"
    assert is_interface[2] == 1, "Antigen 1 should be interface"
    # Antigen 2 should NOT be interface (9 Å > 8 Å)
    assert is_interface[3] == 0, "Antigen 2 should NOT be interface"
    
    print("✓ Multiple contacts correctly counted")


def test_interface_markers_no_opposite_molecule():
    """Test that nodes with no opposite molecule get appropriate defaults."""
    # Only antibody nodes, no antigen
    coords = np.array([
        [0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
    ])
    
    distances = compute_pairwise_distances(coords)
    chain_role = np.array([0, 0])  # Both antibody
    interface_cutoff = 8.0
    
    min_inter_dist, inter_contact_count, is_interface = compute_interface_markers(
        distances, chain_role, interface_cutoff
    )
    
    # No interface possible
    assert (is_interface == 0).all(), "No nodes should be interface (no opposite molecule)"
    assert (inter_contact_count == 0.0).all(), "inter_contact_count should be 0"
    # min_inter_dist should be set to 2 * interface_cutoff (16.0)
    assert min_inter_dist[0].item() <= 16.0, f"min_inter_dist should be <= 16.0, got {min_inter_dist[0]}"
    
    print("✓ Nodes with no opposite molecule handled correctly")


if __name__ == "__main__":
    print("Testing interface markers...")
    test_interface_markers_simple()
    test_interface_markers_outside_cutoff()
    test_interface_markers_multiple_contacts()
    test_interface_markers_no_opposite_molecule()
    print("\nAll interface marker tests passed!")

