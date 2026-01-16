# Web Dashboard Guide

## Quick Start

1. **Train a model** (creates run directory with metrics):
   ```bash
   python train.py --config config.yaml --manifest data/train.csv
   ```

2. **Evaluate the model** (creates eval_predictions.csv):
   ```bash
   python eval.py --model checkpoints/best_model.pt --manifest data/test.csv
   ```

3. **Launch the dashboard**:
   ```bash
   streamlit run src/webapp/app.py
   ```

4. **View results**:
   - Open browser to `http://localhost:8501`
   - Select a run in the "Runs" tab
   - Explore training curves, evaluation results, and 3D graph visualizations

## Dashboard Features

### Tab 1: Runs
- Lists all training runs in `runs/` directory
- Shows configuration summary
- Displays dataset statistics

### Tab 2: Training
- Interactive Plotly charts:
  - Loss over epochs (train/val)
  - Spearman correlation or R² (depending on loss type)
  - Pairwise accuracy (for ranking)
  - Score mean/std
  - Gradient norms
- Select multiple metrics to view simultaneously
- Overlay train/val curves

### Tab 3: Evaluation
- Scatter plot: Predicted vs Ground Truth
- Metrics: Spearman, Pearson, MAE, RMSE, R²
- Residual plot
- Distribution comparison

### Tab 4: Graph Viewer (3D)
- Select any sample from evaluation predictions
- 3D interactive visualization:
  - **Nodes**: Colored by chain role (blue=antibody, red=antigen)
  - **Edges**: 
    - BOUND (orange, thick) - antibody-antigen contacts
    - UNBOUND (gray, thin) - within-chain proximity
    - SEQUENTIAL (green, optional) - (i, i+1) connections
- Controls:
  - Toggle edge types
  - Filter by max distance
  - Limit unbound edges (for performance)
- Hover tooltips with chain label, residue index, coordinates

## File Structure

After training and evaluation, you'll have:

```
runs/
└── YYYYMMDD_HHMMSS_<hash>/
    ├── config.yaml              # Training configuration
    ├── metrics_train.csv        # Training metrics per epoch
    ├── metrics_val.csv          # Validation metrics per epoch
    ├── eval_predictions.csv     # Evaluation predictions (after eval.py)
    └── cache_link.txt           # Path to graph cache

cache/graphs/
└── <hash_prefix>/
    ├── <hash>.pt                # Cached graph (Data object)
    └── <hash>.meta.json         # Graph metadata (chain mapping, etc.)
```

## Graph Data Fields

Each cached graph includes:
- `pos`: [N, 3] Cα coordinates (Angstroms)
- `chain_id`: [N] integer chain index
- `chain_group`: [N] 0=antibody, 1=antigen
- `res_id`: [N] residue number
- `aa`: [N] amino acid type (0-19)
- `edge_type`: [E] 0=BOUND, 1=UNBOUND, 2=SEQUENTIAL
- `edge_dist`: [E] edge distance (Angstroms)

Metadata JSON includes:
- `chain_mapping`: {chain_idx: {label, role}}
- `edge_type_counts`: Counts per edge type
- `pdb_path`: Source PDB file path

## Performance Notes

- Large graphs may be slow to render
- Unbound edges are automatically sampled if > 5000 (configurable)
- All bound edges are always shown
- Graphs are cached in Streamlit for faster reloading

## Troubleshooting

**No runs found**: Train a model first to create run directories.

**Graph not found**: Ensure graph cache directory is correct and graphs were cached during training.

**Missing fields**: Old cached graphs may not have visualization fields. Rebuild cache with `--rebuild-cache`.

