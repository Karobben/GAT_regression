"""
Configuration dataclasses for GNN ranking model.
"""
from dataclasses import dataclass, field
from typing import Optional, List
import yaml
from pathlib import Path


@dataclass
class GraphConfig:
    """Configuration for graph construction."""
    # Distance cutoffs (Angstroms)
    noncovalent_cutoff: float = 10.0  # Cα-Cα distance for noncovalent edges
    interface_cutoff: float = 8.0  # Cα-Cα distance for interface definition
    
    # Edge construction
    use_covalent_edges: bool = True  # Whether to include covalent (peptide backbone) edges
    use_noncovalent_edges: bool = True  # Whether to include noncovalent (distance-based) edges
    allow_duplicate_edges: bool = False  # If False, exclude covalent pairs from noncovalent edges
    
    # Node features
    include_residue_index: bool = True  # Include normalized residue index as feature
    add_interface_features_to_x: bool = True  # Add interface markers to node features


@dataclass
class ModelConfig:
    """Configuration for GAT model architecture."""
    hidden_dim: int = 128
    num_layers: int = 3
    num_heads: int = 4  # Number of attention heads
    dropout: float = 0.1
    use_edge_types: bool = True  # Whether to use edge type information in model
    num_edge_types: int = 2  # COVALENT (0) and NONCOVALENT (1)

    # New relational GAT parameters
    use_residual: bool = True  # Use residual connections
    combine_edge_types: str = "sum"  # "sum" or "concat" for combining edge type messages
    interface_pool_mode: str = "all"  # "all" (pool all interface nodes) or "split_roles" (separate antibody/antigen interface pooling)
    
    # Input feature dimensions (will be set automatically)
    node_feature_dim: int = 24  # 21 AA types + 1 chain type + 1 optional residue index + 2 optional interface features


@dataclass
class LossConfig:
    """Configuration for loss function."""
    loss_type: str = "pairwise_rank"  # Loss type: "pairwise_rank", "mse", "l1", "smooth_l1"
    # Parameters for pairwise ranking loss
    margin_eps: float = 0.0  # Margin for pairwise comparisons
    tie_eps: float = 1e-6  # Ignore pairs with |y_i - y_j| <= tie_eps (ties)
    weight_by_diff: bool = True  # Weight loss by |y_i - y_j|
    reduction: str = "mean"  # "mean" or "sum"
    temperature: float = 1.0  # Temperature scaling for score differences
    # Parameters for smooth L1 loss
    beta: float = 1.0  # Beta parameter for smooth L1 loss (Huber loss threshold)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    batch_size: int = 16
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Optimizer
    optimizer: str = "adamw"  # "adamw" or "adam"
    
    # Learning rate scheduler
    scheduler: str = "cosine"  # "cosine", "step", or None
    scheduler_params: dict = field(default_factory=lambda: {
        "cosine": {"T_max": 100},
        "step": {"step_size": 30, "gamma": 0.1}
    })
    
    # Validation
    val_split: float = 0.2
    val_metric: str = "spearman"  # For ranking: "spearman" or "pairwise_acc"; For regression: "r2", "rmse", or "mae"
    
    # Logging
    log_interval: int = 10
    save_dir: str = "checkpoints"
    save_best: bool = True
    save_all_epochs: bool = False  # If True, save model checkpoint at end of each epoch
    
    # Device
    device: str = "auto"  # "auto", "cuda", or "cpu"
    num_workers: int = 4
    
    # Training stabilizers
    grad_clip: float = 1.0  # Gradient clipping norm (0 = disabled)
    enable_debug_logs: bool = False  # Print detailed debug info
    
    # Overfit sanity test
    overfit_n: int = 0  # If >0, train only on first N samples (no val split)
    
    # Graph preloading
    preload_workers: int = None  # Number of workers for parallel graph preloading (None = auto, 0 = sequential)


@dataclass
class DataConfig:
    """Configuration for data loading."""
    manifest_csv: str = "data/manifest.csv"  # CSV with columns: pdb_path, y, [antibody_chains], [antigen_chains]
    pdb_dir: Optional[str] = None  # Optional base directory for PDB paths (ignored if pdb_path in CSV is absolute)
    
    # Chain assignment heuristics (if not in CSV)
    default_antibody_chains: List[str] = field(default_factory=lambda: ["H", "L"])
    default_antigen_chains: Optional[List[str]] = None  # If None, infer from remaining chains
    
    # Data transforms
    transforms: List[str] = field(default_factory=list)  # e.g., ["normalize"]
    
    # Graph caching
    graph_cache_dir: str = "cache/graphs"  # Directory to cache preprocessed graphs
    hash_pdb_contents: bool = False  # If True, hash file contents; else use mtime+size
    rebuild_cache: bool = False  # If True, ignore existing cache and rebuild
    cache_stats: bool = True  # Print cache hit/miss statistics


@dataclass
class Config:
    """Main configuration class."""
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        
        return cls(
            graph=GraphConfig(**data.get("graph", {})),
            model=ModelConfig(**data.get("model", {})),
            loss=LossConfig(**data.get("loss", {})),
            training=TrainingConfig(**data.get("training", {})),
            data=DataConfig(**data.get("data", {}))
        )
    
    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        data = {
            "graph": self.graph.__dict__,
            "model": self.model.__dict__,
            "loss": self.loss.__dict__,
            "training": self.training.__dict__,
            "data": self.data.__dict__
        }
        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def update_model_dims(self, node_feature_dim: int, num_edge_types: int):
        """Update model dimensions based on actual graph construction."""
        self.model.node_feature_dim = node_feature_dim
        self.model.num_edge_types = num_edge_types

