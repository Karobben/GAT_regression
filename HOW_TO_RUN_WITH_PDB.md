# How to Run with PDB Files - Complete Guide

This guide shows you **exactly** how to run the GAT_regression code with your PDB files, step by step.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Quick Start (5 minutes)](#quick-start-5-minutes)
3. [Complete Workflow](#complete-workflow)
4. [Using Your Own PDB Files](#using-your-own-pdb-files)
5. [Training Options](#training-options)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### 1. Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt
```

Required packages:
- torch >= 1.12.0
- torch-geometric >= 2.0.0
- biopython >= 1.79
- pandas, numpy, scipy, pyyaml, tqdm

### 2. Verify Installation

```bash
# Test that PDB loading works
python test_pdb_loading.py
```

You should see: `🎉 ALL TESTS PASSED!`

---

## Quick Start (5 minutes)

### Option A: Test with Synthetic Data (No PDB files needed)

```bash
cd src
python train.py --synthetic --epoch 10
```

This will:
- Generate synthetic graph data
- Train for 10 epochs
- Show training progress
- Save model to `checkpoints/best_model.pt`

### Option B: Run Demo with Example PDB

```bash
# See how PDB loading works
python demo_pdb_loading.py
```

This shows:
- Creating a test PDB file
- Loading it as a graph
- Graph structure details
- Node and edge breakdown

---

## Complete Workflow

### Step 1: Prepare Your PDB Files

You need PDB files of antibody-antigen complexes. Each PDB should contain:
- **Antibody chains** (typically H for heavy, L for light)
- **Antigen chains** (one or more chains)
- **C-alpha atoms** (standard PDB format)

Example PDB structure:
```
HEADER    ANTIBODY-ANTIGEN COMPLEX
ATOM      1  CA  ALA H   1      20.000  20.000  20.000  1.00 30.00           C
ATOM      2  CA  GLY H   2      21.000  20.000  20.000  1.00 30.00           C
...
```

**Where to get PDB files:**
- Download from RCSB PDB: https://www.rcsb.org/
- Use your own structure predictions
- Use homology models

### Step 2: Create a Manifest CSV

Create a file called `my_manifest.csv` with this format:

```csv
pdb_path,y,antibody_chains,antigen_chains
/absolute/path/to/complex1.pdb,8.5,"H,L",A
/absolute/path/to/complex2.pdb,7.2,"H,L",B
/absolute/path/to/complex3.pdb,9.1,"H,L","A,B"
```

**Column descriptions:**
- `pdb_path`: Full path to PDB file (use absolute paths!)
- `y`: Binding affinity value (e.g., -log(KD), IC50)
- `antibody_chains`: Comma or colon separated (e.g., `"H,L"` or `"H:L"`)
- `antigen_chains`: Chain ID(s) of antigen (e.g., `A` or `"A,B"`)

**Important:**
- Use quotes around multi-chain values: `"H,L"` not `H,L`
- Use absolute paths for PDB files
- Y values are your target predictions (binding strength, etc.)

### Step 3: Run Training

```bash
cd src
python train.py --manifest ../my_manifest.csv --epoch 100
```

**What happens:**
1. Loads PDB files and converts to graphs
2. Caches processed graphs (faster next time)
3. Splits data into train/validation
4. Trains GAT model for 100 epochs
5. Saves best model to `checkpoints/best_model.pt`
6. Prints metrics every epoch

**Training output:**
```
Loading manifest: ../my_manifest.csv
Found 3 samples
Building graphs from PDB files...
Loading graphs: 100%|██████████| 3/3 [00:05<00:00]

Starting training...
Epoch 1/100 - Loss: 2.341, Val Spearman: 0.123
Epoch 2/100 - Loss: 2.104, Val Spearman: 0.245
...
Epoch 100/100 - Loss: 0.543, Val Spearman: 0.876

Best model saved to: checkpoints/best_model.pt
```

### Step 4: Evaluate Your Model

```bash
# Evaluate on test set
python eval.py --model checkpoints/best_model.pt --manifest ../test_manifest.csv
```

**Evaluation output:**
```
==================================================
Evaluation Results (Ranking Metrics)
==================================================
Number of samples: 10
Spearman correlation: 0.8234 (p=1.23e-04)
Pairwise accuracy: 0.9100
Score range: [-2.1234, 3.4567]
Target range: [5.0, 10.0]

Results saved to: checkpoints/eval_results.json
Scatter plot created: checkpoints/eval_scatter.png
```

---

## Using Your Own PDB Files

### Example: Training with 3 PDB Files

Let's say you have three PDB files:
```
/home/user/pdbs/1abc.pdb  (binding score: 8.5)
/home/user/pdbs/2def.pdb  (binding score: 7.2)
/home/user/pdbs/3ghi.pdb  (binding score: 9.1)
```

#### 1. Create manifest.csv

```bash
cat > manifest.csv << 'EOF'
pdb_path,y,antibody_chains,antigen_chains
/home/user/pdbs/1abc.pdb,8.5,"H,L",A
/home/user/pdbs/2def.pdb,7.2,"H,L",B
/home/user/pdbs/3ghi.pdb,9.1,"H,L",A
EOF
```

#### 2. Train the model

```bash
cd src
python train.py --manifest ../manifest.csv --epoch 50
```

#### 3. Check results

```bash
# Model saved to:
ls -lh checkpoints/best_model.pt

# Training history saved to:
cat checkpoints/history.json
```

---

## Training Options

### Basic Options

```bash
# Train with custom epochs
python train.py --manifest manifest.csv --epoch 200

# Use different loss function
python train.py --manifest manifest.csv --loss mse

# Set random seed for reproducibility
python train.py --manifest manifest.csv --seed 42
```

### Advanced Options

```bash
# Use custom config file
python train.py --config my_config.yaml --manifest manifest.csv

# Overfit test (debug mode)
python train.py --manifest manifest.csv --overfit-n 32

# Rebuild cache (if PDB files changed)
python train.py --manifest manifest.csv --rebuild-cache
```

### Configuration File

Create `my_config.yaml`:

```yaml
graph:
  bound_cutoff: 8.0        # Interface contact distance (Å)
  unbound_cutoff: 10.0     # Spatial proximity distance (Å)
  use_sequential_edges: false
  include_residue_index: true

model:
  hidden_dim: 128
  num_layers: 3
  num_heads: 4
  dropout: 0.1

training:
  batch_size: 16
  num_epochs: 100
  learning_rate: 0.001
  val_split: 0.2
```

Then run:
```bash
python train.py --config my_config.yaml --manifest manifest.csv
```

---

## Complete Example Script

Save this as `run_training.sh`:

```bash
#!/bin/bash

# Complete example: Training with PDB files

echo "================================"
echo "GAT Training with PDB Files"
echo "================================"

# Step 1: Check dependencies
echo "Step 1: Checking dependencies..."
python -c "import torch; import torch_geometric; import Bio; print('✓ All dependencies installed')"

# Step 2: Verify PDB loading works
echo "Step 2: Verifying PDB loading..."
python test_pdb_loading.py

# Step 3: Create example manifest (update paths to your PDB files!)
echo "Step 3: Creating manifest..."
cat > my_manifest.csv << 'EOF'
pdb_path,y,antibody_chains,antigen_chains
/path/to/your/complex1.pdb,8.5,"H,L",A
/path/to/your/complex2.pdb,7.2,"H,L",B
EOF

# Step 4: Train model
echo "Step 4: Training model..."
cd src
python train.py --manifest ../my_manifest.csv --epoch 50

# Step 5: Check results
echo "Step 5: Results saved to:"
ls -lh checkpoints/best_model.pt
ls -lh checkpoints/history.json

echo "================================"
echo "Training complete!"
echo "================================"
```

Make executable and run:
```bash
chmod +x run_training.sh
./run_training.sh
```

---

## Troubleshooting

### Problem: "FileNotFoundError: PDB file not found"

**Solution:** Use absolute paths in your manifest:
```csv
# ✗ Wrong - relative path
pdb_path,y,antibody_chains,antigen_chains
complex1.pdb,8.5,"H,L",A

# ✓ Correct - absolute path
pdb_path,y,antibody_chains,antigen_chains
/home/user/pdbs/complex1.pdb,8.5,"H,L",A
```

Or use the `--pdb-dir` option:
```bash
python train.py --manifest manifest.csv --pdb-dir /home/user/pdbs/
```

### Problem: "Chain X not found in structure"

**Solution:** Check your PDB file for correct chain IDs:
```bash
# View chains in PDB file
grep "^ATOM" your_file.pdb | awk '{print $5}' | sort -u
```

Update your manifest with correct chain IDs.

### Problem: "No C-alpha atoms found"

**Solution:** Ensure your PDB contains CA atoms:
```bash
# Check for CA atoms
grep "CA" your_file.pdb | head -5
```

### Problem: CSV parsing errors

**Solution:** Quote multi-chain values:
```csv
# ✗ Wrong - unquoted
antibody_chains
H,L

# ✓ Correct - quoted
antibody_chains
"H,L"
```

### Problem: Out of memory during training

**Solution:** Reduce batch size in config:
```yaml
training:
  batch_size: 8  # Reduce from 16
```

Or use command line:
```bash
python train.py --manifest manifest.csv --batch-size 8
```

### Problem: Want to see detailed progress

**Solution:** Use debug logging:
```bash
# Enable detailed logging
python train.py --manifest manifest.csv --overfit-n 32
```

---

## Next Steps

1. **Start Simple**: Train with synthetic data first
   ```bash
   cd src && python train.py --synthetic --epoch 10
   ```

2. **Test with One PDB**: Create a manifest with one PDB file
   ```bash
   python train.py --manifest single_pdb_manifest.csv --epoch 20
   ```

3. **Scale Up**: Add more PDB files to your manifest
   ```bash
   python train.py --manifest full_manifest.csv --epoch 100
   ```

4. **Evaluate**: Test your trained model
   ```bash
   python eval.py --model checkpoints/best_model.pt --manifest test.csv
   ```

5. **Optimize**: Tune hyperparameters in config file
   ```bash
   python train.py --config optimized_config.yaml --manifest manifest.csv
   ```

---

## Summary

**To run with PDB files, you need:**

1. ✅ PDB files (antibody-antigen complexes)
2. ✅ Manifest CSV (paths + binding values)
3. ✅ Run training: `python train.py --manifest manifest.csv`
4. ✅ Evaluate: `python eval.py --model checkpoint.pt --manifest test.csv`

**Quick test (no PDB needed):**
```bash
cd src && python train.py --synthetic --epoch 10
```

**Full workflow:**
```bash
# 1. Create manifest
echo 'pdb_path,y,antibody_chains,antigen_chains' > manifest.csv
echo '/path/to/file.pdb,8.5,"H,L",A' >> manifest.csv

# 2. Train
cd src && python train.py --manifest ../manifest.csv

# 3. Evaluate
python eval.py --model checkpoints/best_model.pt --manifest ../test.csv
```

For more details, see:
- `QUICKSTART_PDB.md` - Quick reference
- `PDB_LOADING_GUIDE.md` - Complete documentation
- `demo_pdb_loading.py` - Interactive demo

**Need help?** Run `python demo_pdb_loading.py` to see how it works!
