"""
3D graph visualization for protein structures.
"""
import torch
import numpy as np
from torch_geometric.data import Data
from pathlib import Path
from typing import Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings


def plot_graph_3d(
    data: Data,
    out_path: str,
    max_edges: int = 5000,
    show: bool = False,
    color_by: str = "chain_group",
    edge_color_by: str = "edge_type",
    subsample_edges: bool = True,
    node_size: int = 10,
    linewidth: float = 0.5,
    only_bound_edges: bool = False,
    edge_max_dist: Optional[float] = None,
    label_chains: bool = False
):
    """
    Plot 3D graph visualization of protein structure.
    
    Args:
        data: PyTorch Geometric Data object with pos, chain_group, edge_type, etc.
        out_path: Path to save PNG file
        max_edges: Maximum number of edges to plot (for performance)
        show: Whether to display plot (requires GUI)
        color_by: Node coloring scheme ("chain_group" or "chain_id")
        edge_color_by: Edge coloring scheme ("edge_type")
        subsample_edges: Whether to subsample edges if exceeding max_edges
        node_size: Size of node markers
        linewidth: Width of edge lines
        only_bound_edges: If True, plot only bound (interface) edges
        edge_max_dist: Maximum edge distance to plot (Angstroms)
        label_chains: Whether to add legend for chain labels
    """
    # Check required fields
    if not hasattr(data, 'pos') or data.pos is None:
        raise ValueError("Data object must have 'pos' field (Cα coordinates)")
    
    pos = data.pos.cpu().numpy() if torch.is_tensor(data.pos) else data.pos
    num_nodes = pos.shape[0]
    
    # Get node colors
    if color_by == "chain_group":
        if not hasattr(data, 'chain_group') or data.chain_group is None:
            warnings.warn("chain_group not found, using chain_id instead")
            color_by = "chain_id"
    
    if color_by == "chain_group":
        node_colors = data.chain_group.cpu().numpy() if torch.is_tensor(data.chain_group) else data.chain_group
        # Map: 0=antibody (blue), 1=antigen (red)
        color_map = {0: 'blue', 1: 'red'}
        node_colors_list = [color_map.get(int(c), 'gray') for c in node_colors]
        legend_labels = {'blue': 'Antibody', 'red': 'Antigen'}
    elif color_by == "chain_id":
        if not hasattr(data, 'chain_id') or data.chain_id is None:
            raise ValueError("chain_id not found in data")
        chain_ids = data.chain_id.cpu().numpy() if torch.is_tensor(data.chain_id) else data.chain_id
        # Use different colors for different chains
        unique_chains = np.unique(chain_ids)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_chains)))
        chain_color_map = {int(cid): colors[i] for i, cid in enumerate(unique_chains)}
        node_colors_list = [chain_color_map[int(cid)] for cid in chain_ids]
        legend_labels = {}
        if label_chains and hasattr(data, 'chain_labels'):
            for cid, color in chain_color_map.items():
                if cid < len(data.chain_labels):
                    legend_labels[color] = f"Chain {data.chain_labels[cid]}"
    else:
        raise ValueError(f"Unknown color_by: {color_by}")
    
    # Get edges
    edge_index = data.edge_index.cpu().numpy() if torch.is_tensor(data.edge_index) else data.edge_index
    num_edges = edge_index.shape[1]
    
    # Filter edges if needed
    if only_bound_edges:
        if not hasattr(data, 'edge_type') or data.edge_type is None:
            warnings.warn("edge_type not found, cannot filter bound edges")
        else:
            edge_type = data.edge_type.cpu().numpy() if torch.is_tensor(data.edge_type) else data.edge_type
            # edge_type: 0=unbound, 1=bound, 2=sequential
            bound_mask = edge_type == 1
            edge_index = edge_index[:, bound_mask]
            if hasattr(data, 'edge_type'):
                edge_type = edge_type[bound_mask]
            num_edges = edge_index.shape[1]
            print(f"Filtered to {num_edges} bound edges")
    
    # Filter by distance if specified
    if edge_max_dist is not None:
        if not hasattr(data, 'edge_dist') or data.edge_dist is None:
            # Compute distances on the fly
            edge_dist = np.linalg.norm(pos[edge_index[0]] - pos[edge_index[1]], axis=1)
        else:
            edge_dist = data.edge_dist.cpu().numpy() if torch.is_tensor(data.edge_dist) else data.edge_dist
        
        dist_mask = edge_dist <= edge_max_dist
        edge_index = edge_index[:, dist_mask]
        if hasattr(data, 'edge_type'):
            edge_type = edge_type[dist_mask] if only_bound_edges or 'edge_type' in locals() else None
        num_edges = edge_index.shape[1]
        print(f"Filtered to {num_edges} edges within {edge_max_dist} Å")
    
    # Subsample edges if needed
    if subsample_edges and num_edges > max_edges:
        # Prefer keeping bound edges if available
        if hasattr(data, 'edge_type') and data.edge_type is not None:
            edge_type_full = data.edge_type.cpu().numpy() if torch.is_tensor(data.edge_type) else data.edge_type
            if only_bound_edges or 'edge_type' not in locals():
                # Re-filter if we lost edge_type
                edge_type_full = edge_type_full[bound_mask] if only_bound_edges else edge_type_full
                if edge_max_dist is not None:
                    edge_type_full = edge_type_full[dist_mask]
            
            # Prioritize bound edges (type 0)
            bound_indices = np.where(edge_type_full == 0)[0]
            other_indices = np.where(edge_type_full != 0)[0]
            
            # Keep all bound edges if possible, then sample others
            if len(bound_indices) < max_edges:
                n_other = max_edges - len(bound_indices)
                if len(other_indices) > n_other:
                    sampled_other = np.random.choice(other_indices, n_other, replace=False)
                    selected = np.concatenate([bound_indices, sampled_other])
                else:
                    selected = np.concatenate([bound_indices, other_indices])
            else:
                selected = np.random.choice(bound_indices, max_edges, replace=False)
        else:
            # Random sample
            selected = np.random.choice(num_edges, max_edges, replace=False)
        
        edge_index = edge_index[:, selected]
        num_edges = edge_index.shape[1]
        print(f"Subsampled to {num_edges} edges")
    
    # Get edge colors
    if edge_color_by == "edge_type":
        if not hasattr(data, 'edge_type') or data.edge_type is None:
            warnings.warn("edge_type not found, using default edge color")
            edge_colors_list = ['gray'] * num_edges
        else:
            edge_type_plot = data.edge_type.cpu().numpy() if torch.is_tensor(data.edge_type) else data.edge_type
            if only_bound_edges or edge_max_dist is not None or (subsample_edges and num_edges < len(edge_type_plot)):
                # Need to use filtered edge_type
                if 'edge_type' in locals():
                    edge_type_plot = edge_type
                else:
                    edge_type_full = data.edge_type.cpu().numpy() if torch.is_tensor(data.edge_type) else data.edge_type
                    if only_bound_edges:
                        bound_mask = edge_type_full == 0  # 0 = BOUND
                        edge_type_plot = edge_type_full[bound_mask]
                    else:
                        edge_type_plot = edge_type_full
                    if edge_max_dist is not None:
                        edge_type_plot = edge_type_plot[dist_mask] if 'dist_mask' in locals() else edge_type_plot
                    if subsample_edges and num_edges < len(edge_type_plot):
                        edge_type_plot = edge_type_plot[selected] if 'selected' in locals() else edge_type_plot
            
            # Map: 0=bound (orange), 1=unbound (gray), 2=sequential (green)
            # Note: edge_type 0=BOUND, 1=UNBOUND, 2=SEQUENTIAL
            edge_color_map = {0: 'orange', 1: 'gray', 2: 'green'}
            edge_colors_list = [edge_color_map.get(int(et), 'black') for et in edge_type_plot[:num_edges]]
    else:
        edge_colors_list = ['gray'] * num_edges
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot edges
    for i in range(num_edges):
        u, v = edge_index[0, i], edge_index[1, i]
        ax.plot(
            [pos[u, 0], pos[v, 0]],
            [pos[u, 1], pos[v, 1]],
            [pos[u, 2], pos[v, 2]],
            color=edge_colors_list[i],
            linewidth=linewidth,
            alpha=0.6
        )
    
    # Plot nodes
    scatter = ax.scatter(
        pos[:, 0], pos[:, 1], pos[:, 2],
        c=node_colors_list,
        s=node_size,
        alpha=0.8,
        edgecolors='black',
        linewidths=0.5
    )
    
    # Labels and title
    ax.set_xlabel('X (Å)', fontsize=12)
    ax.set_ylabel('Y (Å)', fontsize=12)
    ax.set_zlabel('Z (Å)', fontsize=12)
    
    title = f"Protein Graph (N={num_nodes}, E={num_edges})"
    if only_bound_edges:
        title += " - Bound Edges Only"
    ax.set_title(title, fontsize=14)
    
    # Add legend if requested
    if legend_labels:
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=label) for color, label in legend_labels.items()]
        ax.legend(handles=legend_elements, loc='upper left')
    
    # Set equal aspect ratio
    max_range = np.array([
        pos[:, 0].max() - pos[:, 0].min(),
        pos[:, 1].max() - pos[:, 1].min(),
        pos[:, 2].max() - pos[:, 2].min()
    ]).max() / 2.0
    mid_x = (pos[:, 0].max() + pos[:, 0].min()) * 0.5
    mid_y = (pos[:, 1].max() + pos[:, 1].min()) * 0.5
    mid_z = (pos[:, 2].max() + pos[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    
    # Save
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved 3D graph plot to: {out_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

