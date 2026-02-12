#!/usr/bin/env python3
"""
Simple demonstration of PDB loading for initial processing.
This script shows the basic workflow of loading a PDB file into the system.
"""

import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.pdb_to_graph import pdb_to_graph


def create_example_pdb():
    """
    Create a minimal example PDB file for demonstration.
    
    This represents a simple antibody-antigen complex with:
    - Chain H: Heavy chain (3 residues)
    - Chain L: Light chain (2 residues)  
    - Chain A: Antigen (3 residues)
    """
    pdb_content = """HEADER    EXAMPLE ANTIBODY-ANTIGEN COMPLEX
REMARK   This is a minimal example for demonstration
REMARK   Chain H: Antibody Heavy Chain (ALA, GLY, VAL)
REMARK   Chain L: Antibody Light Chain (LEU, ILE)
REMARK   Chain A: Antigen (SER, THR, TRP)
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


def demonstrate_pdb_loading():
    """Demonstrate the PDB loading process step by step."""
    
    print("="*70)
    print("PDB LOADING DEMONSTRATION")
    print("="*70)
    print()
    
    # Step 1: Create example PDB
    print("Step 1: Creating example PDB file...")
    pdb_path = create_example_pdb()
    print(f"✓ Created PDB file: {pdb_path}")
    print(f"  File size: {Path(pdb_path).stat().st_size} bytes")
    print()
    
    # Step 2: Load PDB as graph
    print("Step 2: Loading PDB file as graph...")
    print("  Parameters:")
    print("    - Antibody chains: H (Heavy), L (Light)")
    print("    - Antigen chains: A")
    print("    - Bound cutoff: 8.0 Å (interface contacts)")
    print("    - Unbound cutoff: 10.0 Å (spatial proximity)")
    print()
    
    try:
        data = pdb_to_graph(
            pdb_path=pdb_path,
            antibody_chains=["H", "L"],
            antigen_chains=["A"],
            bound_cutoff=8.0,
            unbound_cutoff=10.0,
            use_sequential_edges=False,
            include_residue_index=True,
            y=5.0  # Example binding affinity value
        )
        
        print("✓ Successfully loaded PDB as graph!")
        print()
        
        # Step 3: Inspect the graph
        print("Step 3: Graph structure:")
        print("="*70)
        print(f"Nodes (residues):     {data.x.shape[0]}")
        print(f"Node features:        {data.x.shape[1]}")
        print(f"  - Residue type:     21 (one-hot encoding of amino acid)")
        print(f"  - Chain type:       1 (antibody=1, antigen=0)")
        print(f"  - Residue index:    1 (normalized position)")
        print()
        print(f"Edges:                {data.edge_index.shape[1]}")
        print(f"Edge features:        {data.edge_attr.shape[1]}")
        print(f"  - Edge type:        1 (BOUND=0, UNBOUND=1)")
        print(f"  - Distance:         1 (normalized)")
        print()
        print(f"Target value (y):     {data.y.item()}")
        print()
        
        # Step 4: Show node breakdown
        print("Step 4: Node breakdown by chain:")
        print("="*70)
        
        # Chain type is at index 21
        chain_types = data.x[:, 21]
        antibody_count = (chain_types == 1.0).sum().item()
        antigen_count = (chain_types == 0.0).sum().item()
        
        print(f"Antibody nodes (chains H, L): {antibody_count}")
        print(f"Antigen nodes (chain A):      {antigen_count}")
        print(f"Total nodes:                  {data.x.shape[0]}")
        print()
        
        # Step 5: Show edge breakdown
        print("Step 5: Edge breakdown by type:")
        print("="*70)
        
        # Edge type is at index 0 of edge_attr
        edge_types = data.edge_attr[:, 0]
        bound_count = (edge_types == 0.0).sum().item()
        unbound_count = (edge_types == 1.0).sum().item()
        
        print(f"BOUND edges (interface):      {bound_count}")
        print(f"UNBOUND edges (spatial):      {unbound_count}")
        print(f"Total edges:                  {data.edge_index.shape[1]}")
        print()
        
        # Summary
        print("="*70)
        print("SUMMARY")
        print("="*70)
        print()
        print("✅ PDB file successfully loaded and converted to graph!")
        print()
        print("The graph is now ready for:")
        print("  - Training with Graph Attention Networks (GAT)")
        print("  - Predicting binding affinity")
        print("  - Analyzing antibody-antigen interactions")
        print()
        print("Next steps:")
        print("  1. Create a manifest CSV with your PDB files")
        print("  2. Run training: python src/train.py --manifest manifest.csv")
        print("  3. Evaluate: python src/eval.py --model checkpoint.pt --manifest test.csv")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading PDB: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        import os
        if Path(pdb_path).exists():
            os.unlink(pdb_path)


if __name__ == "__main__":
    success = demonstrate_pdb_loading()
    sys.exit(0 if success else 1)
