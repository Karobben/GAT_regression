"""
Graph Attention Network (GAT) for ranking antibody-antigen complexes.
Produces a scalar score per graph for pairwise ranking.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Batch
from typing import Optional


class GATRanker(nn.Module):
    """
    GAT model for ranking antibody-antigen complexes.
    
    Architecture:
    - Multiple GAT layers with attention heads
    - Edge type information incorporated via separate attention or edge embeddings
    - Global pooling (mean + max) for graph-level representation
    - MLP head for scalar score prediction
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_edge_types: bool = True,
        num_edge_types: int = 2
    ):
        """
        Initialize GAT ranker.
        
        Args:
            node_feature_dim: Dimension of node features
            hidden_dim: Hidden dimension for GAT layers
            num_layers: Number of GAT layers
            num_heads: Number of attention heads per layer
            dropout: Dropout probability
            use_edge_types: Whether to use edge type information
            num_edge_types: Number of edge types (BOUND, UNBOUND, [SEQUENTIAL])
        """
        super().__init__()
        
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_edge_types = use_edge_types
        self.num_edge_types = num_edge_types
        
        # Input projection
        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim * num_heads
            self.gat_layers.append(
                GATConv(
                    in_channels=in_dim,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True  # Concatenate heads
                )
            )
        
        # Edge type embedding (if using edge types)
        # Note: We'll update embedding dim based on current layer's output dim
        if use_edge_types:
            self.edge_type_embeddings = nn.ModuleList()
            for i in range(num_layers):
                # Embedding dim matches the input dim for each layer
                in_dim = hidden_dim if i == 0 else hidden_dim * num_heads
                self.edge_type_embeddings.append(nn.Embedding(num_edge_types, in_dim))
        
        # Final layer dimension after concatenating heads
        final_dim = hidden_dim * num_heads
        
        # Global pooling: mean + max
        self.pool_dim = final_dim * 2
        
        # MLP head for scalar score with LayerNorm for stability
        self.score_head = nn.Sequential(
            nn.Linear(self.pool_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # LayerNorm for stability
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),  # LayerNorm for stability
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Scalar score
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features (N, node_feature_dim)
            edge_index: Edge indices (2, E)
            edge_attr: Edge attributes (E, 2) [edge_type, normalized_distance]
            batch: Batch assignment vector (N,) for batching multiple graphs
            
        Returns:
            scores: Scalar scores per graph (B, 1) where B is batch size
        """
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Apply GAT layers
        for i, gat_layer in enumerate(self.gat_layers):
            # Incorporate edge type information if available
            if self.use_edge_types and edge_attr is not None:
                # Extract edge type from edge_attr (first column)
                edge_type = edge_attr[:, 0].long()  # (E,)
                # Get embedding for current layer (dimension matches current x)
                edge_type_emb = self.edge_type_embeddings[i](edge_type)  # (E, x_dim)
                
                # Aggregate edge type embeddings per node (mean pooling)
                # This handles nodes with multiple edges correctly
                row, col = edge_index
                num_nodes = x.shape[0]
                x_dim = x.shape[1]
                
                # Aggregate incoming edge embeddings for each node
                edge_emb_agg = torch.zeros(num_nodes, x_dim, 
                                          device=x.device, dtype=x.dtype)
                edge_emb_agg.scatter_add_(0, row.unsqueeze(1).expand(-1, x_dim), 
                                         edge_type_emb)
                edge_emb_agg.scatter_add_(0, col.unsqueeze(1).expand(-1, x_dim), 
                                         edge_type_emb)
                
                # Count edges per node for normalization
                edge_count = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
                edge_count.scatter_add_(0, row, torch.ones_like(row, dtype=x.dtype))
                edge_count.scatter_add_(0, col, torch.ones_like(col, dtype=x.dtype))
                edge_count = torch.clamp(edge_count, min=1.0)  # Avoid division by zero
                
                # Normalize and add to node features
                edge_emb_agg = edge_emb_agg / edge_count.unsqueeze(1)
                x = x + edge_emb_agg  # Additive incorporation
            
            # Apply GAT layer
            x = gat_layer(x, edge_index)
            if i < len(self.gat_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling: mean + max
        if batch is None:
            # Single graph
            x_mean = x.mean(dim=0, keepdim=True)
            x_max = x.max(dim=0, keepdim=True)[0]
        else:
            # Batched graphs
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
        
        x_pooled = torch.cat([x_mean, x_max], dim=1)  # (B, pool_dim)
        
        # Predict scalar score
        score = self.score_head(x_pooled)  # (B, 1)
        
        return score.squeeze(-1)  # (B,)
    
    def forward_batch(self, data: Batch) -> torch.Tensor:
        """Convenience method for batched data."""
        return self.forward(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            batch=data.batch
        )

