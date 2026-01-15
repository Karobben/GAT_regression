# GNN Ranker for Antibody-Antigen Binding

A PyTorch Geometric-based Graph Neural Network (GNN) model for ranking antibody-antigen complexes by binding affinity. The model uses a **pairwise ranking loss** to learn relative ordering within batches, rather than predicting absolute binding values.

## Overview

This project implements a Graph Attention Network (GAT) that:
- Converts PDB structures of antibody-antigen complexes into graphs using C-alpha atoms
- Learns graph representations with edge-type aware attention (BOUND vs UNBOUND edges)
- Produces scalar scores for ranking complexes by binding property
- Trains using pairwise ranking loss (RankNet-style) within batches

## Key Features

- **Graph Construction**: C-alpha atoms only, with configurable distance cutoffs
- **Edge Types**: 
  - BOUND edges: Cross-interface contacts (antibody ↔ antigen) within 8.0 Å
  - UNBOUND edges: Within-chain spatial proximity within 10.0 Å
  - Optional sequential edges (i, i+1) within chains
- **Node Features**: Residue type (20 AA one-hot), chain type (antibody/antigen), optional residue index
- **Model**: GAT with 4 attention heads, edge-type aware message passing
- **Loss**: Pairwise ranking loss with configurable margin and weighting

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (optional, but recommended)

### Setup

1. Create and activate conda environment:
```bash
conda create -n gnn-ranker python=3.9
conda activate gnn-ranker
```

2. Install dependencies:
```bash
pip install torch torch-geometric biopython numpy pandas scipy pyyaml tqdm
```

Or install from requirements file (if provided):
```bash
pip install -r requirements.txt
```

## Project Structure

```
GNNBind/
├── src/
│   ├── data/
│   │   ├── pdb_to_graph.py      # PDB → PyTorch Geometric Data
│   │   ├── dataset.py            # Dataset class for CSV manifest
│   │   └── transforms.py         # Data augmentation transforms
│   ├── models/
│   │   └── gat_ranker.py         # GAT model for ranking
│   ├── losses/
│   │   └── pairwise_rank_loss.py # Pairwise ranking loss
│   ├── config.py                 # Configuration dataclasses
│   ├── train.py                  # Training script
│   ├── eval.py                   # Evaluation script
│   └── utils.py                  # Utility functions
├── tests/
│   ├── test_pdb_to_graph.py      # Tests for graph construction
│   └── test_loss.py              # Tests for ranking loss
└── README.md
```

## Data Format

### Manifest CSV

The training data should be provided as a CSV file with the following columns:

- `pdb_path`: **Absolute path** to PDB file (recommended) or relative path
- `y`: Continuous binding property value (e.g., -log(KD), IC50, etc.)
- `antibody_chains` (optional): Comma-separated chain IDs (e.g., "H,L")
- `antigen_chains` (optional): Comma-separated chain IDs (e.g., "A")

Example `manifest.csv` (using absolute paths - recommended):
```csv
pdb_path,y,antibody_chains,antigen_chains
/home/user/data/pdbs/complex1.pdb,8.5,"H,L",A
/home/user/data/pdbs/complex2.pdb,7.2,"H,L",B
/home/user/data/pdbs/complex3.pdb,9.1,"H,L",A
```

**Note**: Using absolute paths eliminates the need for the `--pdb-dir` parameter. If you use relative paths, you can either:
- Provide `--pdb-dir` to specify the base directory, or
- Place the manifest CSV in the same directory as your PDB files

**Important**: When multiple chains are specified, they must be comma-separated within a **single column**. If the values are not quoted, CSV parsers will split them into separate columns, causing data misalignment.

- **Correct**: `antibody_chains` = `"H,L"` (quoted, parsed as single column containing "H,L")
- **Incorrect**: `antibody_chains` = `H,L` (unquoted, parsed as two separate columns: H and L)

For multiple antigen chains: `antigen_chains` = `"A,B"` (quoted) or `A` (single chain, no quotes needed)

**Note**: If `antibody_chains` and `antigen_chains` are not provided, the code will:
- Default to `["H", "L"]` for antibodies
- Attempt to infer antigen chains from remaining chains (with a warning)

## Usage

### Training

Basic training with default configuration (using absolute paths in manifest):
```bash
cd src
python train.py --manifest ../data/manifest.csv
```

If using relative paths in manifest, specify the base directory:
```bash
python train.py --manifest ../data/manifest.csv --pdb-dir ../data/pdbs/
```

With custom configuration:
```bash
python train.py --config config.yaml --manifest ../data/manifest.csv
```

Override specific settings from command line:
```bash
# Override number of epochs
python train.py --config config.yaml --manifest ../data/manifest.csv --epoch 200

# Override loss function
python train.py --config config.yaml --manifest ../data/manifest.csv --loss mse

# Override both
python train.py --config config.yaml --manifest ../data/manifest.csv --epoch 50 --loss l1
```

Synthetic dataset for testing (no PDB files needed):
```bash
python train.py --synthetic
```

### Evaluation

Evaluate a trained model (using absolute paths in manifest):
```bash
python eval.py --model checkpoints/best_model.pt --manifest ../data/test_manifest.csv
```

If using relative paths in manifest, specify the base directory:
```bash
python eval.py --model checkpoints/best_model.pt --manifest ../data/test_manifest.csv --pdb-dir ../data/pdbs/
```

### Configuration

You can create a YAML configuration file to customize training:

```yaml
graph:
  bound_cutoff: 8.0
  unbound_cutoff: 10.0
  use_sequential_edges: false
  include_residue_index: true

model:
  hidden_dim: 128
  num_layers: 3
  num_heads: 4
  dropout: 0.1
  use_edge_types: true

loss:
  loss_type: "pairwise_rank"  # Options: "pairwise_rank", "mse", "l1", "smooth_l1"
  margin_eps: 0.0             # For pairwise_rank only
  weight_by_diff: true        # For pairwise_rank only
  reduction: "mean"

training:
  batch_size: 16
  num_epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001
  optimizer: "adamw"
  scheduler: "cosine"
  val_split: 0.2
  device: "auto"
```

## How Ranking Loss Works

The pairwise ranking loss is designed for learning relative orderings rather than absolute values:

1. **Within each batch**, for every pair of samples (i, j) where `y_i > y_j + margin_eps`:
   - We enforce that the predicted score for sample i should be greater than sample j
   - Loss: `-log(sigmoid(score_i - score_j))`

2. **Optional weighting**: Loss can be weighted by `|y_i - y_j|` to emphasize confident orderings

3. **Benefits**:
   - Robust to scale differences in labels
   - Focuses on relative comparisons (what we care about for ranking)
   - Works well when absolute values are noisy but relative orderings are reliable

## Model Architecture

1. **Input**: Node features (residue type + chain type + optional index)
2. **GAT Layers**: Multiple layers of graph attention with edge-type awareness
3. **Global Pooling**: Mean + Max pooling over nodes
4. **MLP Head**: Produces scalar score per graph

## Evaluation Metrics

- **Spearman Correlation**: Rank correlation between predicted scores and targets
- **Pairwise Accuracy**: Fraction of correctly ordered pairs within batches

## Testing

Run unit tests:
```bash
cd tests
python test_pdb_to_graph.py
python test_loss.py
```

## Example: Tiny Run

To test the pipeline with synthetic data:

1. Create a minimal manifest:
```bash
echo "pdb_path,y,antibody_chains,antigen_chains" > test_manifest.csv
echo "test1.pdb,5.0,H,L,A" >> test_manifest.csv
echo "test2.pdb,7.0,H,L,A" >> test_manifest.csv
```

2. (You'll need actual PDB files or modify the dataset to handle missing files)

3. Train:
```bash
python train.py --manifest test_manifest.csv --pdb-dir ./
```

## Notes

- **PDB Parsing**: Uses BioPython's PDBParser. Handles missing residues gracefully.
- **Chain Assignment**: If chains are not specified in CSV, defaults to H/L for antibodies.
- **Edge Construction**: Efficiently computed using vectorized numpy operations.
- **Batching**: Uses PyTorch Geometric's DataLoader for efficient batching of graphs.

## Troubleshooting

1. **No edges found**: Check distance cutoffs and chain assignments
2. **CUDA out of memory**: Reduce batch size or model hidden dimension
3. **PDB parsing errors**: Ensure PDB files are valid and contain C-alpha atoms

## License

[Specify your license]

## Citation

If you use this code, please cite:
[Add citation information]

