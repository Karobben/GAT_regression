# Documentation Index

## Quick Answer: "How to run with PDB files?"

👉 **See [HOW_TO_RUN_WITH_PDB.md](HOW_TO_RUN_WITH_PDB.md)** for complete step-by-step instructions!

Or try the quick reference: `cat QUICK_REFERENCE.txt`

---

## Documentation Files

### 🚀 Getting Started

| File | Description | Best For |
|------|-------------|----------|
| **[HOW_TO_RUN_WITH_PDB.md](HOW_TO_RUN_WITH_PDB.md)** | Complete step-by-step running guide | First-time users |
| **[QUICK_REFERENCE.txt](QUICK_REFERENCE.txt)** | Visual quick reference card | Quick lookups |
| **[run_example.sh](run_example.sh)** | Executable example script | Hands-on learning |
| **[README.md](README.md)** | Project overview | Understanding the system |

### 📚 PDB Loading Documentation

| File | Description | Best For |
|------|-------------|----------|
| **[PDB_LOADING_STATUS.md](PDB_LOADING_STATUS.md)** | Quick status overview | Checking capabilities |
| **[QUICKSTART_PDB.md](QUICKSTART_PDB.md)** | Quick reference for PDB loading | Common use cases |
| **[PDB_LOADING_GUIDE.md](PDB_LOADING_GUIDE.md)** | Complete PDB documentation | Detailed information |

### 🧪 Testing & Demos

| File | Description | Command |
|------|-------------|---------|
| **[test_pdb_loading.py](test_pdb_loading.py)** | Test PDB loading functionality | `python test_pdb_loading.py` |
| **[demo_pdb_loading.py](demo_pdb_loading.py)** | Interactive demonstration | `python demo_pdb_loading.py` |

### 🔧 Debugging

| File | Description | Best For |
|------|-------------|----------|
| **[DEBUGGING_RUNBOOK.md](DEBUGGING_RUNBOOK.md)** | Debugging guide | Troubleshooting issues |

---

## Quick Commands

### Test Everything Works
```bash
python test_pdb_loading.py
```

### Quick Training Test (No PDB files needed)
```bash
cd src && python train.py --synthetic --epoch 10
```

### Run Complete Example
```bash
./run_example.sh
```

### Train with Your PDB Files
```bash
# 1. Create manifest.csv
cat > manifest.csv << 'EOF'
pdb_path,y,antibody_chains,antigen_chains
/path/to/complex1.pdb,8.5,"H,L",A
EOF

# 2. Train
cd src && python train.py --manifest ../manifest.csv --epoch 100
```

---

## Learning Path

### Beginner
1. Read [QUICK_REFERENCE.txt](QUICK_REFERENCE.txt) (2 min)
2. Run `./run_example.sh` (5 min)
3. Try synthetic training: `cd src && python train.py --synthetic --epoch 10`

### Intermediate
1. Read [HOW_TO_RUN_WITH_PDB.md](HOW_TO_RUN_WITH_PDB.md) (10 min)
2. Run `python test_pdb_loading.py`
3. Create manifest with your PDB files
4. Train your first model

### Advanced
1. Read [PDB_LOADING_GUIDE.md](PDB_LOADING_GUIDE.md)
2. Customize config file
3. Optimize hyperparameters
4. Use advanced training options

---

## Common Questions

### Q: Can the script load PDB files?
**A:** Yes! See [PDB_LOADING_STATUS.md](PDB_LOADING_STATUS.md)

### Q: How do I run with PDB files?
**A:** See [HOW_TO_RUN_WITH_PDB.md](HOW_TO_RUN_WITH_PDB.md)

### Q: What's the quickest way to test?
**A:** Run `cd src && python train.py --synthetic --epoch 10`

### Q: I have PDB files, what now?
**A:** Create a manifest CSV, then run training. See [HOW_TO_RUN_WITH_PDB.md](HOW_TO_RUN_WITH_PDB.md) Section "Complete Workflow"

### Q: What format should my manifest be?
**A:** See [QUICK_REFERENCE.txt](QUICK_REFERENCE.txt) or [HOW_TO_RUN_WITH_PDB.md](HOW_TO_RUN_WITH_PDB.md) Section "Manifest CSV Format"

### Q: Something's not working!
**A:** See [HOW_TO_RUN_WITH_PDB.md](HOW_TO_RUN_WITH_PDB.md) Section "Troubleshooting" or [DEBUGGING_RUNBOOK.md](DEBUGGING_RUNBOOK.md)

---

## File Tree

```
GAT_regression/
├── Documentation
│   ├── HOW_TO_RUN_WITH_PDB.md         ← START HERE
│   ├── QUICK_REFERENCE.txt            ← Quick lookup
│   ├── README.md                       ← Project overview
│   ├── PDB_LOADING_STATUS.md          ← Capabilities
│   ├── PDB_LOADING_GUIDE.md           ← Detailed docs
│   ├── QUICKSTART_PDB.md              ← Quick reference
│   ├── DEBUGGING_RUNBOOK.md           ← Troubleshooting
│   └── DOC_INDEX.md                   ← This file
│
├── Examples & Tests
│   ├── run_example.sh                 ← Run this!
│   ├── test_pdb_loading.py            ← Test loading
│   └── demo_pdb_loading.py            ← See demo
│
├── Source Code
│   └── src/
│       ├── train.py                    ← Training script
│       ├── eval.py                     ← Evaluation script
│       └── data/
│           ├── pdb_to_graph.py        ← PDB loading
│           └── dataset.py             ← Dataset loader
│
└── Configuration
    ├── config_example.yaml            ← Example config
    └── requirements.txt               ← Dependencies
```

---

## Next Steps

1. **First time?** → Read [HOW_TO_RUN_WITH_PDB.md](HOW_TO_RUN_WITH_PDB.md)
2. **Quick test?** → Run `./run_example.sh`
3. **Have PDB files?** → Follow the 3-step workflow in [QUICK_REFERENCE.txt](QUICK_REFERENCE.txt)
4. **Need help?** → Check troubleshooting in [HOW_TO_RUN_WITH_PDB.md](HOW_TO_RUN_WITH_PDB.md)

**Most important:** You can test everything without PDB files using synthetic data:
```bash
cd src && python train.py --synthetic --epoch 10
```

This verifies your installation and shows you how training works!
