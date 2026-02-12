# Quick Start: PDB Loading

## TL;DR - Can the script load PDB files?

**YES! ✅** The repository fully supports loading PDB files for initial processing.

## Quickest Test (30 seconds)

```bash
# Run the verification test
python test_pdb_loading.py

# Or see a demonstration
python demo_pdb_loading.py
```

## Common Use Cases

### Use Case 1: Load a Single PDB File

```python
from data.pdb_to_graph import pdb_to_graph

data = pdb_to_graph(
    pdb_path="structure.pdb",
    antibody_chains=["H", "L"],
    antigen_chains=["A"]
)
print(f"Loaded {data.x.shape[0]} residues, {data.edge_index.shape[1]} edges")
```

### Use Case 2: Load Multiple PDB Files from CSV

Create `manifest.csv`:
```csv
pdb_path,y,antibody_chains,antigen_chains
/path/to/complex1.pdb,8.5,"H,L",A
/path/to/complex2.pdb,7.2,"H:L",B
```

Load:
```python
from data.dataset import CachedGraphDataset

dataset = CachedGraphDataset(manifest_csv="manifest.csv")
for idx in range(len(dataset)):
    data = dataset[idx]
    # Use data...
```

### Use Case 3: Train a Model

```bash
cd src
python train.py --manifest manifest.csv --epoch 100
```

## What Gets Extracted from PDB?

- **C-alpha atoms** from specified chains
- **Residue types** (20 amino acids + unknown)
- **Coordinates** for distance-based edges
- **Chain assignments** (antibody vs antigen)

Graph structure:
- **Nodes**: Residues with features (type, chain, position)
- **Edges**: BOUND (interface) and UNBOUND (spatial proximity)

## Supported Features

✅ Direct PDB file loading  
✅ Batch loading via CSV manifest  
✅ Comma-separated chains: `"H,L"`  
✅ Colon-separated chains: `"H:L"`  
✅ Absolute and relative paths  
✅ Automatic caching  
✅ BioPython-based parsing  

## Files to Use

- **Test**: `test_pdb_loading.py` - Verify everything works
- **Demo**: `demo_pdb_loading.py` - See step-by-step loading
- **Guide**: `PDB_LOADING_GUIDE.md` - Full documentation
- **Train**: `src/train.py` - Train models with PDB data

## Example Output

```
✓ Successfully loaded PDB file!
  - Number of nodes (residues): 8
  - Node feature dimension: 23
  - Number of edges: 44
  - Target value (y): 5.0
```

## Need Help?

See `PDB_LOADING_GUIDE.md` for:
- Detailed examples
- Troubleshooting
- PDB requirements
- Caching options
- Graph construction details

---

**Conclusion**: The repository is fully functional for PDB loading! 🎉
