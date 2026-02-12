#!/usr/bin/env python3
"""
Test script to verify PDB loading functionality for initial processing.
This script tests:
1. Loading a synthetic PDB file directly
2. Loading through the dataset interface with a manifest
3. Chain parsing with both comma and colon separators
"""

import sys
import tempfile
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.pdb_to_graph import pdb_to_graph
from data.dataset import CachedGraphDataset
import pandas as pd


def create_test_pdb():
    """Create a minimal test PDB file."""
    pdb_content = """HEADER    TEST ANTIBODY-ANTIGEN COMPLEX
REMARK   This is a synthetic PDB for testing
ATOM      1  CA  ALA H   1      20.000  20.000  20.000  1.00 30.00           C
ATOM      2  CA  GLY H   2      21.000  20.000  20.000  1.00 30.00           C
ATOM      3  CA  VAL H   3      22.000  20.000  20.000  1.00 30.00           C
ATOM      4  CA  LEU L   1      20.000  21.000  20.000  1.00 30.00           C
ATOM      5  CA  ILE L   2      21.000  21.000  20.000  1.00 30.00           C
ATOM      6  CA  SER A   1      20.000  20.000  23.000  1.00 30.00           C
ATOM      7  CA  THR A   2      21.000  20.000  23.000  1.00 30.00           C
ATOM      8  CA  TRP A   3      22.000  20.000  23.000  1.00 30.00           C
END
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write(pdb_content)
        return f.name


def test_direct_pdb_loading():
    """Test 1: Direct PDB loading with pdb_to_graph function."""
    print("\n" + "="*70)
    print("TEST 1: Direct PDB Loading")
    print("="*70)
    
    pdb_path = create_test_pdb()
    
    try:
        print(f"Created test PDB: {pdb_path}")
        print(f"PDB file exists: {os.path.exists(pdb_path)}")
        print(f"PDB file size: {os.path.getsize(pdb_path)} bytes")
        
        # Test loading with comma-separated chains
        print("\nTesting with comma-separated chains: ['H', 'L'] and ['A']")
        data = pdb_to_graph(
            pdb_path=pdb_path,
            antibody_chains=["H", "L"],
            antigen_chains=["A"],
            bound_cutoff=8.0,
            unbound_cutoff=10.0,
            use_sequential_edges=False,
            include_residue_index=True,
            y=5.0
        )
        
        print(f"✓ Successfully loaded PDB file!")
        print(f"  - Number of nodes (residues): {data.x.shape[0]}")
        print(f"  - Node feature dimension: {data.x.shape[1]}")
        print(f"  - Number of edges: {data.edge_index.shape[1]}")
        print(f"  - Target value (y): {data.y.item()}")
        print(f"  - Edge attribute dimension: {data.edge_attr.shape}")
        
        # Verify expected structure
        assert data.x.shape[0] == 8, f"Expected 8 nodes, got {data.x.shape[0]}"
        assert data.edge_index.shape[0] == 2, "Edge index should have 2 rows"
        assert data.y.item() == 5.0, f"Expected y=5.0, got {data.y.item()}"
        
        print("\n✅ TEST 1 PASSED: Direct PDB loading works correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if os.path.exists(pdb_path):
            os.unlink(pdb_path)


def test_manifest_with_comma_separator():
    """Test 2: Loading via manifest CSV with comma separators."""
    print("\n" + "="*70)
    print("TEST 2: Manifest Loading with Comma Separators")
    print("="*70)
    
    pdb_path = create_test_pdb()
    
    # Create manifest with comma-separated chains
    manifest_content = f"""pdb_path,y,antibody_chains,antigen_chains
{pdb_path},5.0,"H,L",A
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(manifest_content)
        manifest_path = f.name
    
    try:
        print(f"Created test manifest: {manifest_path}")
        print("Manifest content:")
        print(manifest_content)
        
        # Test loading dataset
        dataset = CachedGraphDataset(
            manifest_csv=manifest_path,
            pdb_dir=None,
            bound_cutoff=8.0,
            unbound_cutoff=10.0,
            use_sequential_edges=False,
            include_residue_index=True,
            graph_cache_dir="/tmp/test_cache",
            rebuild_cache=True
        )
        
        print(f"\n✓ Dataset created with {len(dataset)} samples")
        
        # Load first sample
        data = dataset[0]
        
        print(f"✓ Successfully loaded graph from manifest!")
        print(f"  - Number of nodes (residues): {data.x.shape[0]}")
        print(f"  - Target value (y): {data.y.item()}")
        
        assert data.x.shape[0] == 8, f"Expected 8 nodes, got {data.x.shape[0]}"
        assert data.y.item() == 5.0, f"Expected y=5.0, got {data.y.item()}"
        
        print("\n✅ TEST 2 PASSED: Manifest loading with comma separators works!")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if os.path.exists(pdb_path):
            os.unlink(pdb_path)
        if os.path.exists(manifest_path):
            os.unlink(manifest_path)


def test_manifest_with_colon_separator():
    """Test 3: Loading via manifest CSV with colon separators."""
    print("\n" + "="*70)
    print("TEST 3: Manifest Loading with Colon Separators")
    print("="*70)
    
    pdb_path = create_test_pdb()
    
    # Create manifest with colon-separated chains (as in existing manifest.csv)
    manifest_content = f"""pdb_path,y,antibody_chains,antigen_chains
{pdb_path},5.0,"H:L",A
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(manifest_content)
        manifest_path = f.name
    
    try:
        print(f"Created test manifest: {manifest_path}")
        print("Manifest content:")
        print(manifest_content)
        
        # Test loading dataset
        dataset = CachedGraphDataset(
            manifest_csv=manifest_path,
            pdb_dir=None,
            bound_cutoff=8.0,
            unbound_cutoff=10.0,
            use_sequential_edges=False,
            include_residue_index=True,
            graph_cache_dir="/tmp/test_cache_colon",
            rebuild_cache=True
        )
        
        print(f"\n✓ Dataset created with {len(dataset)} samples")
        
        # Load first sample
        data = dataset[0]
        
        print(f"✓ Successfully loaded graph from manifest!")
        print(f"  - Number of nodes (residues): {data.x.shape[0]}")
        print(f"  - Target value (y): {data.y.item()}")
        
        assert data.x.shape[0] == 8, f"Expected 8 nodes, got {data.x.shape[0]}"
        assert data.y.item() == 5.0, f"Expected y=5.0, got {data.y.item()}"
        
        print("\n✅ TEST 3 PASSED: Manifest loading with colon separators works!")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if os.path.exists(pdb_path):
            os.unlink(pdb_path)
        if os.path.exists(manifest_path):
            os.unlink(manifest_path)


def test_chain_detection():
    """Test 4: Chain detection and parsing."""
    print("\n" + "="*70)
    print("TEST 4: Chain Detection and Parsing")
    print("="*70)
    
    pdb_path = create_test_pdb()
    manifest_content = f"""pdb_path,y,antibody_chains,antigen_chains
{pdb_path},5.0,"H:L",A
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(manifest_content)
        manifest_path = f.name
    
    try:
        dataset = CachedGraphDataset(
            manifest_csv=manifest_path,
            pdb_dir=None,
            graph_cache_dir="/tmp/test_cache_chains",
            rebuild_cache=True
        )
        
        print(f"Dataset antibody chains: {dataset.antibody_chains_list[0]}")
        print(f"Dataset antigen chains: {dataset.antigen_chains_list[0]}")
        
        assert dataset.antibody_chains_list[0] == ["H", "L"], "Antibody chains parsing failed"
        assert dataset.antigen_chains_list[0] == ["A"], "Antigen chains parsing failed"
        
        print("\n✅ TEST 4 PASSED: Chain detection and parsing works!")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if os.path.exists(pdb_path):
            os.unlink(pdb_path)
        if os.path.exists(manifest_path):
            os.unlink(manifest_path)


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("PDB LOADING VERIFICATION TEST SUITE")
    print("="*70)
    print("Testing if the repository can load PDB files for initial processing")
    print("="*70)
    
    results = []
    
    # Run all tests
    results.append(("Direct PDB Loading", test_direct_pdb_loading()))
    results.append(("Manifest with Comma Separators", test_manifest_with_comma_separator()))
    results.append(("Manifest with Colon Separators", test_manifest_with_colon_separator()))
    results.append(("Chain Detection", test_chain_detection()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "="*70)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("The repository CAN successfully load PDB files for initial processing.")
        print("\nKey capabilities verified:")
        print("  1. Direct PDB file loading via pdb_to_graph()")
        print("  2. Loading via manifest CSV files")
        print("  3. Support for comma-separated chains (H,L)")
        print("  4. Support for colon-separated chains (H:L)")
        print("  5. Proper chain detection and parsing")
    else:
        print("⚠️ SOME TESTS FAILED")
        print("There may be issues with PDB loading functionality.")
    print("="*70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
