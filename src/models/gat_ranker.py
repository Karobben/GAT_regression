"""
Graph Attention Network (GAT) for ranking antibody-antigen complexes.
Produces a scalar score per graph for pairwise ranking.

New architecture:
- Relational GAT: separate convolutions for each edge type (covalent vs noncovalent)
- Interface-only pooling: final graph representation uses only interface nodes
- Full-graph message passing provides structural context, interface pooling focuses on binding
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Batch
from typing import Optional, List


class GATRanker(nn.Module):
    """
    Relational GAT model for ranking antibody-antigen complexes.
    
    Architecture:
    - Full-graph message passing captures structural context
    - Relational GAT: separate convolutions for covalent vs noncovalent edges
    - Interface-only pooling focuses final decision on binding interface
    - MLP head produces scalar score for ranking

    Key features:
    - Edge types handled via separate message passing (not just embeddings)
    - Interface nodes identified by is_interface marker
    - Optional role-aware pooling (antibody vs antigen interface)
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_edge_types: bool = True,
        num_edge_types: int = 2,
        use_residual: bool = True,
        combine_edge_types: str = "sum",
        interface_pool_mode: str = "all"
    ):
        """
        Initialize relational GAT ranker.
        
        Args:
            node_feature_dim: Dimension of node features
            hidden_dim: Hidden dimension for GAT layers
            num_layers: Number of GAT layers
            num_heads: Number of attention heads per layer
            dropout: Dropout probability
            use_edge_types: Whether to use edge type information
            num_edge_types: Number of edge types (COVALENT=0, NONCOVALENT=1)
            use_residual: Whether to use residual connections
            combine_edge_types: How to combine messages from different edge types ("sum" or "concat")
            interface_pool_mode: Interface pooling mode ("all" or "split_roles")
        """
        super().__init__()
        
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_edge_types = use_edge_types
        self.num_edge_types = num_edge_types
        self.use_residual = use_residual
        self.combine_edge_types = combine_edge_types
        self.interface_pool_mode = interface_pool_mode
        
        # Input projection
        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)
        
        # Relational GAT layers: separate convs for each edge type
        self.gat_layers_cov = nn.ModuleList()  # For covalent edges (type 0)
        self.gat_layers_non = nn.ModuleList()  # For noncovalent edges (type 1)

        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim * num_heads

            self.gat_layers_cov.append(
                GATConv(
                    in_channels=in_dim,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True
                )
            )

            self.gat_layers_non.append(
                GATConv(
                    in_channels=in_dim,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True
                )
            )
        
        # Layer normalization for residual connections
        if use_residual:
            self.layer_norms = nn.ModuleList()
            for i in range(num_layers):
                layer_dim = hidden_dim * num_heads
                self.layer_norms.append(nn.LayerNorm(layer_dim))

        # Combination layer if using concat mode
        if combine_edge_types == "concat":
            concat_dim = hidden_dim * num_heads * 2  # Double because we concat two edge type outputs
            self.combine_proj = nn.Linear(concat_dim, hidden_dim * num_heads)
        
        # Final layer dimension after concatenating heads
        final_dim = hidden_dim * num_heads
        
        # Interface pooling dimensions depend on pooling mode
        if interface_pool_mode == "all":
            # Pool all interface nodes: mean + max
            self.pool_dim = final_dim * 2
        elif interface_pool_mode == "split_roles":
            # Split by antibody/antigen interface: ab_mean + ab_max + ag_mean + ag_max
            self.pool_dim = final_dim * 4
        else:
            raise ValueError(f"Unknown interface_pool_mode: {interface_pool_mode}")
        
        # MLP head for scalar score with LayerNorm for stability
        self.score_head = nn.Sequential(
            nn.Linear(self.pool_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Scalar score
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        edge_type: Optional[torch.Tensor] = None,
        is_interface: Optional[torch.Tensor] = None,
        chain_role: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with relational GAT and interface-only pooling.
        
        Args:
            x: Node features (N, node_feature_dim)
            edge_index: Edge indices (2, E)
            edge_attr: Edge attributes (E, 2) [edge_type, normalized_distance] - legacy
            batch: Batch assignment vector (N,) for batching multiple graphs
            edge_type: Edge type tensor (E,) with 0=COVALENT, 1=NONCOVALENT
            is_interface: Interface marker (N,) with 1=interface node, 0=non-interface
            chain_role: Chain role (N,) with 0=antibody, 1=antigen
            
        Returns:
            scores: Scalar scores per graph (B,) where B is batch size
        """
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Get edge type information
        num_edges = edge_index.shape[1] if edge_index is not None else 0
        
        if self.use_edge_types:
            # Try multiple sources for edge_type
            edge_type_tensor = None
            
            if edge_type is not None:
                edge_type_tensor = edge_type.long()
            elif hasattr(self, '_edge_type') and self._edge_type is not None:
                edge_type_tensor = self._edge_type.long()
            elif edge_attr is not None and edge_attr.shape[1] > 0:
                edge_type_tensor = edge_attr[:, 0].long()
            
            # Validate edge_type shape matches number of edges
            if edge_type_tensor is not None:
                if edge_type_tensor.shape[0] != num_edges:
                    # Shape mismatch - try to fix or warn
                    if edge_type_tensor.numel() == num_edges:
                        # Reshape if it's just a different shape
                        edge_type_tensor = edge_type_tensor.view(-1)[:num_edges]
                    elif num_edges > 0 and edge_type_tensor.numel() > num_edges:
                        # If edge_type is larger, take first num_edges elements
                        # This can happen if edge_type includes padding or extra data
                        edge_type_tensor = edge_type_tensor[:num_edges]
                    else:
                        # If shapes don't match at all, fall back to None
                        import warnings
                        warnings.warn(
                            f"edge_type shape {edge_type_tensor.shape} doesn't match "
                            f"number of edges {num_edges}. Disabling edge type filtering."
                        )
                        edge_type_tensor = None
        else:
            edge_type_tensor = None

        # Apply relational GAT layers
        for i in range(self.num_layers):
            x_input = x  # Store input for residual connection

            # Separate edges by type
            if edge_type_tensor is not None and num_edges > 0:
                mask_cov = (edge_type_tensor == 0)  # Covalent edges
                mask_non = (edge_type_tensor == 1)  # Noncovalent edges

                edge_index_cov = edge_index[:, mask_cov] if mask_cov.any() else torch.empty(2, 0, dtype=edge_index.dtype, device=edge_index.device)
                edge_index_non = edge_index[:, mask_non] if mask_non.any() else torch.empty(2, 0, dtype=edge_index.dtype, device=edge_index.device)

                # Apply separate convolutions for each edge type
                h_cov = self.gat_layers_cov[i](x, edge_index_cov) if edge_index_cov.shape[1] > 0 else torch.zeros_like(x)
                h_non = self.gat_layers_non[i](x, edge_index_non) if edge_index_non.shape[1] > 0 else torch.zeros_like(x)

                # Combine messages from different edge types
                if self.combine_edge_types == "sum":
                    x_new = h_cov + h_non
                elif self.combine_edge_types == "concat":
                    x_combined = torch.cat([h_cov, h_non], dim=1)
                    x_new = self.combine_proj(x_combined)
                else:
                    raise ValueError(f"Unknown combine_edge_types: {self.combine_edge_types}")
            else:
                # Fallback: use single GAT layer on all edges
                x_new = self.gat_layers_cov[i](x, edge_index)

            # Residual connection and normalization
            if self.use_residual and x_input.shape == x_new.shape:
                x = self.layer_norms[i](x_input + F.dropout(x_new, p=self.dropout, training=self.training))
            else:
                x = x_new

            # Activation for intermediate layers
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Interface-only pooling
        x_pooled = self._interface_pooling(x, batch, is_interface, chain_role)
        
        # Predict scalar score
        score = self.score_head(x_pooled)
        
        return score.squeeze(-1)  # (B,)
    
    def _interface_pooling(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor],
        is_interface: Optional[torch.Tensor],
        chain_role: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Perform interface-only pooling.

        Args:
            x: Node features after GAT layers (N, final_dim)
            batch: Batch assignment (N,)
            is_interface: Interface marker (N,) - 1 for interface nodes
            chain_role: Chain role (N,) - 0 for antibody, 1 for antigen

        Returns:
            Pooled representation (B, pool_dim)
        """
        if batch is None:
            # Single graph case
            if is_interface is None or not is_interface.any():
                # No interface nodes, fallback to all nodes
                x_mean = x.mean(dim=0, keepdim=True)
                x_max = x.max(dim=0, keepdim=True)[0]
                return torch.cat([x_mean, x_max], dim=1)

            # Pool only interface nodes
            interface_mask = is_interface == 1
            x_interface = x[interface_mask]

            if self.interface_pool_mode == "all":
                x_mean = x_interface.mean(dim=0, keepdim=True)
                x_max = x_interface.max(dim=0, keepdim=True)[0]
                return torch.cat([x_mean, x_max], dim=1)

            elif self.interface_pool_mode == "split_roles":
                if chain_role is None:
                    # Fallback to all interface pooling
                    x_mean = x_interface.mean(dim=0, keepdim=True)
                    x_max = x_interface.max(dim=0, keepdim=True)[0]
                    return torch.cat([x_mean, x_max, x_mean, x_max], dim=1)  # Duplicate for consistency

                # Split by antibody/antigen interface
                ab_interface_mask = interface_mask & (chain_role == 0)
                ag_interface_mask = interface_mask & (chain_role == 1)

                # Pool antibody interface (fallback to zeros if empty)
                if ab_interface_mask.any():
                    x_ab = x[ab_interface_mask]
                    ab_mean = x_ab.mean(dim=0, keepdim=True)
                    ab_max = x_ab.max(dim=0, keepdim=True)[0]
                else:
                    ab_mean = ab_max = torch.zeros_like(x[:1])

                # Pool antigen interface (fallback to zeros if empty)
                if ag_interface_mask.any():
                    x_ag = x[ag_interface_mask]
                    ag_mean = x_ag.mean(dim=0, keepdim=True)
                    ag_max = x_ag.max(dim=0, keepdim=True)[0]
                else:
                    ag_mean = ag_max = torch.zeros_like(x[:1])

                return torch.cat([ab_mean, ab_max, ag_mean, ag_max], dim=1)

        else:
            # Batched graphs case
            batch_size = batch.max().item() + 1
            pooled_features = []

            for b in range(batch_size):
                # Get nodes for this graph
                graph_mask = batch == b
                x_graph = x[graph_mask]

                # Get interface information for this graph
                interface_mask = None
                if is_interface is not None:
                    graph_interface = is_interface[graph_mask]
                    interface_mask = graph_interface == 1

                chain_role_graph = None
                if chain_role is not None and interface_mask is not None:
                    chain_role_graph = chain_role[graph_mask]

                # Pool this graph
                if interface_mask is None or not interface_mask.any():
                    # No interface nodes, fallback to all nodes
                    x_mean = x_graph.mean(dim=0, keepdim=True)
                    x_max = x_graph.max(dim=0, keepdim=True)[0]
                    graph_pooled = torch.cat([x_mean, x_max], dim=1)
                else:
                    # Pool interface nodes
                    x_interface = x_graph[interface_mask]

                    if self.interface_pool_mode == "all":
                        x_mean = x_interface.mean(dim=0, keepdim=True)
                        x_max = x_interface.max(dim=0, keepdim=True)[0]
                        graph_pooled = torch.cat([x_mean, x_max], dim=1)

                    elif self.interface_pool_mode == "split_roles":
                        if chain_role_graph is None:
                            # Fallback to all interface pooling
                            x_mean = x_interface.mean(dim=0, keepdim=True)
                            x_max = x_interface.max(dim=0, keepdim=True)[0]
                            graph_pooled = torch.cat([x_mean, x_max, x_mean, x_max], dim=1)
                        else:
                            # Split by antibody/antigen interface
                            ab_interface_mask = chain_role_graph == 0
                            ag_interface_mask = chain_role_graph == 1

                            # Pool antibody interface
                            if ab_interface_mask.any():
                                x_ab = x_interface[ab_interface_mask]
                                ab_mean = x_ab.mean(dim=0, keepdim=True)
                                ab_max = x_ab.max(dim=0, keepdim=True)[0]
                            else:
                                ab_mean = ab_max = torch.zeros_like(x_graph[:1])

                            # Pool antigen interface
                            if ag_interface_mask.any():
                                x_ag = x_interface[ag_interface_mask]
                                ag_mean = x_ag.mean(dim=0, keepdim=True)
                                ag_max = x_ag.max(dim=0, keepdim=True)[0]
                            else:
                                ag_mean = ag_max = torch.zeros_like(x_graph[:1])

                            graph_pooled = torch.cat([ab_mean, ab_max, ag_mean, ag_max], dim=1)

                pooled_features.append(graph_pooled)

            return torch.cat(pooled_features, dim=0)

    def forward_batch(self, data: Batch) -> torch.Tensor:
        """
        Convenience method for batched data with interface markers.

        Args:
            data: Batch containing x, edge_index, edge_type, batch, is_interface, chain_role

        Returns:
            scores: Scalar scores per graph (B,)
        """
        return self.forward(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=getattr(data, 'edge_attr', None),
            batch=data.batch,
            edge_type=getattr(data, 'edge_type', None),
            is_interface=getattr(data, 'is_interface', None),
            chain_role=getattr(data, 'chain_role', None)
        )

