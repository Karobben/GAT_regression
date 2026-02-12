# PDB Loading Guide

## Overview

This guide demonstrates how the GAT_regression repository loads PDB files for initial processing. The system is fully functional and supports multiple input methods.

## Quick Test

Run the verification test to confirm PDB loading works:

```bash
python test_pdb_loading.py
```

This test verifies:
- Direct PDB file loading
- Manifest CSV loading with comma separators (H,L)
- Manifest CSV loading with colon separators (H:L)
- Chain detection and parsing

## Method 1: Direct PDB Loading

Use the `pdb_to_graph()` function to load a single PDB file:

```python
from data.pdb_to_graph import pdb_to_graph

# Load a PDB file directly
data = pdb_to_graph(
    pdb_path="/path/to/structure.pdb",
    antibody_chains=["H", "L"],  # Heavy and Light chains
    antigen_chains=["A"],         # Antigen chain
    bound_cutoff=8.0,            # Distance cutoff for interface edges (Å)
    unbound_cutoff=10.0,         # Distance cutoff for spatial edges (Å)
    use_sequential_edges=False,  # Optional sequential edges
    include_residue_index=True,  # Include residue position features
    y=5.0                        # Optional target value (e.g., binding affinity)
)

print(f"Loaded {data.x.shape[0]} residues")
print(f"Created {data.edge_index.shape[1]} edges")
```

## Method 2: Loading via Manifest CSV

For batch processing, use a manifest CSV file:

### CSV Format

Create a CSV file with the following columns:

```csv
pdb_path,y,antibody_chains,antigen_chains
/path/to/complex1.pdb,8.5,"H,L",A
/path/to/complex2.pdb,7.2,"H:L",B
/path/to/complex3.pdb,9.1,"H,L","A,B"
```

**Important Notes:**
- Use **absolute paths** for PDB files (recommended)
- Use **comma (`,`)** OR **colon (`:`)** to separate multiple chains
- **Quote** chain values when they contain separators: `"H,L"` or `"H:L"`
- Both separator styles are supported!

### Loading the Dataset

```python
from data.dataset import CachedGraphDataset

# Create dataset from manifest
dataset = CachedGraphDataset(
    manifest_csv="manifest.csv",
    pdb_dir=None,  # Not needed if using absolute paths
    bound_cutoff=8.0,
    unbound_cutoff=10.0,
    use_sequential_edges=False,
    include_residue_index=True,
    graph_cache_dir="cache/graphs",  # Cache processed graphs
    rebuild_cache=False  # Set True to force rebuild
)

# Load a specific sample
data = dataset[0]
print(f"Sample 0: {data.x.shape[0]} residues, y={data.y.item()}")

# Iterate through all samples
for idx in range(len(dataset)):
    data = dataset[idx]
    # Process data...
```

## Method 3: Using Training Scripts

The repository provides ready-to-use training scripts:

```bash
# Train with a manifest file
cd src
python train.py --manifest manifest.csv

# With custom configuration
python train.py --config config.yaml --manifest manifest.csv

# Test with synthetic data (no PDB files needed)
python train.py --synthetic
```

## PDB File Requirements

### Format
- Standard PDB format
- Must contain C-alpha (CA) atoms
- Chain IDs must match those specified in manifest

### Example PDB Structure
```
HEADER    ANTIBODY-ANTIGEN COMPLEX
ATOM      1  CA  ALA H   1      20.000  20.000  20.000  1.00 30.00           C
ATOM      2  CA  GLY H   2      21.000  20.000  20.000  1.00 30.00           C
...
ATOM     50  CA  SER A   1      20.000  20.000  23.000  1.00 30.00           C
ATOM     51  CA  THR A   2      21.000  20.000  23.000  1.00 30.00           C
...
END
```

### What Gets Extracted
- **C-alpha atoms only** (CA) from specified chains
- **Residue types** (20 standard amino acids + unknown)
- **Coordinates** for distance calculations
- **Chain assignments** (antibody vs antigen)

## Graph Construction

The PDB file is converted to a graph where:

1. **Nodes** = C-alpha atoms (residues)
   - Features: residue type (one-hot), chain type, position index
   
2. **Edges** = Two types:
   - **BOUND edges**: Cross-interface contacts (antibody ↔ antigen within cutoff)
   - **UNBOUND edges**: Within-chain spatial proximity
   - Optional: Sequential edges (i → i+1 in sequence)

3. **Edge Features**:
   - Edge type (BOUND vs UNBOUND)
   - Normalized distance

## Caching

The system automatically caches processed graphs to disk:

```python
dataset = CachedGraphDataset(
    manifest_csv="manifest.csv",
    graph_cache_dir="cache/graphs",  # Cache directory
    hash_pdb_contents=False,  # Use file mtime+size (fast)
    rebuild_cache=False       # Use cache if available
)

# View cache statistics
dataset.print_cache_stats()
```

Benefits:
- **Fast loading**: Reuse preprocessed graphs
- **Consistency**: Same processing for train/val/test
- **Disk space**: Graphs are compressed PyTorch tensors

## Troubleshooting

### Issue: "Chain X not found in structure"
- **Solution**: Check that chain IDs in manifest match PDB file
- Use a PDB viewer to verify chain IDs

### Issue: "No C-alpha atoms found"
- **Solution**: Ensure PDB contains CA atoms for specified chains
- Check for HETATM records vs ATOM records

### Issue: CSV parsing splits comma-separated chains
- **Solution**: Quote the chain values: `"H,L"` instead of `H,L`

### Issue: File not found
- **Solution**: Use absolute paths in manifest or provide `--pdb-dir`

## Verification

To verify your setup works:

1. **Run the test suite:**
   ```bash
   python test_pdb_loading.py
   ```

2. **Run existing unit tests:**
   ```bash
   cd tests
   python test_pdb_to_graph.py
   ```

3. **Try with synthetic data:**
   ```bash
   cd src
   python train.py --synthetic --epoch 2
   ```

## Summary

✅ **The repository FULLY supports PDB loading for initial processing**

Key features:
- ✅ Direct PDB file loading with `pdb_to_graph()`
- ✅ Batch loading via manifest CSV files
- ✅ Support for both comma (`,`) and colon (`:`) chain separators
- ✅ Automatic graph construction with edge types
- ✅ Disk caching for efficiency
- ✅ Integration with training pipeline

The system is production-ready and handles PDB files correctly!
