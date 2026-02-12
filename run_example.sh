#!/bin/bash

# ==============================================================================
# Complete Example: Running GAT Training with PDB Files
# ==============================================================================
# This script shows the complete workflow from start to finish
# ==============================================================================

set -e  # Exit on error

echo ""
echo "========================================================================"
echo "  GAT Training with PDB Files - Complete Example"
echo "========================================================================"
echo ""

# ------------------------------------------------------------------------------
# Step 1: Check dependencies
# ------------------------------------------------------------------------------
echo "Step 1: Checking dependencies..."
echo "------------------------------------------------------------------------"

if python -c "import torch, torch_geometric, Bio, pandas, numpy" 2>/dev/null; then
    echo "✓ All required packages are installed"
else
    echo "✗ Missing packages. Installing..."
    pip install -r requirements.txt
fi

echo ""

# ------------------------------------------------------------------------------
# Step 2: Test PDB loading
# ------------------------------------------------------------------------------
echo "Step 2: Testing PDB loading functionality..."
echo "------------------------------------------------------------------------"

if python test_pdb_loading.py > /tmp/test_output.txt 2>&1; then
    echo "✓ PDB loading test passed"
    grep "ALL TESTS PASSED" /tmp/test_output.txt || true
else
    echo "✗ PDB loading test failed. See /tmp/test_output.txt for details"
    exit 1
fi

echo ""

# ------------------------------------------------------------------------------
# Step 3: Run demonstration
# ------------------------------------------------------------------------------
echo "Step 3: Running PDB loading demonstration..."
echo "------------------------------------------------------------------------"

echo "This shows how a PDB file is converted to a graph..."
python demo_pdb_loading.py 2>&1 | head -30

echo ""
echo "(Output truncated - see above for graph structure details)"
echo ""

# ------------------------------------------------------------------------------
# Step 4: Option A - Train with synthetic data (no PDB files needed)
# ------------------------------------------------------------------------------
echo "Step 4: Training with synthetic data (no PDB files needed)..."
echo "------------------------------------------------------------------------"
echo "This is useful for testing without having actual PDB files."
echo ""

cd src

echo "Running: python train.py --synthetic --epoch 5"
echo ""

if python train.py --synthetic --epoch 5 2>&1 | tee /tmp/train_output.txt; then
    echo ""
    echo "✓ Training completed successfully!"
    echo ""
    echo "Model saved to: src/checkpoints/best_model.pt"
    echo ""
else
    echo "✗ Training failed. See /tmp/train_output.txt"
    exit 1
fi

cd ..

# ------------------------------------------------------------------------------
# Step 5: Show results
# ------------------------------------------------------------------------------
echo "========================================================================"
echo "  Training Complete - Summary"
echo "========================================================================"
echo ""

if [ -f "src/checkpoints/best_model.pt" ]; then
    echo "✓ Model saved:"
    ls -lh src/checkpoints/best_model.pt
    echo ""
fi

if [ -f "src/checkpoints/history.json" ]; then
    echo "✓ Training history saved:"
    ls -lh src/checkpoints/history.json
    echo ""
    echo "Last few lines of training history:"
    tail -10 src/checkpoints/history.json
    echo ""
fi

# ------------------------------------------------------------------------------
# Next steps
# ------------------------------------------------------------------------------
echo "========================================================================"
echo "  Next Steps"
echo "========================================================================"
echo ""
echo "To train with your own PDB files:"
echo ""
echo "1. Create a manifest CSV file:"
echo "   cat > my_manifest.csv << 'EOF'"
echo "   pdb_path,y,antibody_chains,antigen_chains"
echo "   /path/to/complex1.pdb,8.5,\"H,L\",A"
echo "   /path/to/complex2.pdb,7.2,\"H,L\",B"
echo "   EOF"
echo ""
echo "2. Run training:"
echo "   cd src"
echo "   python train.py --manifest ../my_manifest.csv --epoch 100"
echo ""
echo "3. Evaluate the model:"
echo "   python eval.py --model checkpoints/best_model.pt --manifest ../test.csv"
echo ""
echo "For detailed instructions, see: HOW_TO_RUN_WITH_PDB.md"
echo ""
echo "========================================================================"
echo "  Example Complete!"
echo "========================================================================"
echo ""
