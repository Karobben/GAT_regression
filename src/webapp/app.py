"""
Interactive web dashboard for GAT ranking model.
Built with Streamlit for training progress, evaluation results, and 3D graph visualization.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import argparse
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.webapp.utils import (
    list_runs, load_config, load_metrics, load_eval_predictions,
    get_cache_dir, find_graph_cache_path, load_graph, get_graph_stats,
    list_available_graphs
)


# Page config
st.set_page_config(
    page_title="GAT Ranking Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if "selected_run" not in st.session_state:
    st.session_state.selected_run = None
if "graph_cache_dir" not in st.session_state:
    st.session_state.graph_cache_dir = None


def parse_args():
    """Parse command line arguments."""
    # Streamlit passes args differently - check sys.argv
    import sys
    runs_dir = "runs"
    graph_cache_dir = "cache/graphs"
    
    # Parse from sys.argv if present
    if "--runs-dir" in sys.argv:
        idx = sys.argv.index("--runs-dir")
        if idx + 1 < len(sys.argv):
            runs_dir = sys.argv[idx + 1]
    
    if "--graph-cache-dir" in sys.argv:
        idx = sys.argv.index("--graph-cache-dir")
        if idx + 1 < len(sys.argv):
            graph_cache_dir = sys.argv[idx + 1]
    
    return runs_dir, graph_cache_dir


@st.cache_data
def load_metrics_cached(run_dir: Path, split: str):
    """Cached version of load_metrics."""
    return load_metrics(run_dir, split)


@st.cache_data
def load_eval_predictions_cached(run_dir: Path):
    """Cached version of load_eval_predictions."""
    return load_eval_predictions(run_dir)


@st.cache_resource
def load_graph_cached(cache_path: Path):
    """Cached version of load_graph."""
    graph, metadata = load_graph(cache_path)
    return graph, metadata


def render_runs_tab(runs_dir: Path):
    """Render the Runs selection tab."""
    st.header("📁 Runs")
    
    runs = list_runs(runs_dir)
    
    if not runs:
        st.warning(f"No runs found in {runs_dir}. Train a model first!")
        return None
    
    # Run selector
    run_options = {f"{r['run_id']}": r['path'] for r in runs}
    selected_run_id = st.selectbox(
        "Select Run",
        options=list(run_options.keys()),
        index=0
    )
    
    run_dir = Path(run_options[selected_run_id])
    st.session_state.selected_run = run_dir
    
    # Load and display config
    config = load_config(run_dir)
    if config:
        st.subheader("Configuration")
        with st.expander("View Config", expanded=False):
            config_dict = {
                "graph": config.graph.__dict__,
                "model": config.model.__dict__,
                "loss": config.loss.__dict__,
                "training": config.training.__dict__,
                "data": config.data.__dict__
            }
            st.json(config_dict)
    
    # Dataset stats
    st.subheader("Dataset Statistics")
    train_metrics = load_metrics_cached(run_dir, "train")
    val_metrics = load_metrics_cached(run_dir, "val")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if train_metrics is not None:
            st.metric("Training Epochs", len(train_metrics))
    with col2:
        if val_metrics is not None:
            st.metric("Validation Epochs", len(val_metrics))
    with col3:
        eval_preds = load_eval_predictions_cached(run_dir)
        if eval_preds is not None:
            st.metric("Evaluation Samples", len(eval_preds))
    
    # Cache directory
    cache_dir = get_cache_dir(run_dir)
    if cache_dir:
        st.session_state.graph_cache_dir = cache_dir
        st.info(f"Graph cache: {cache_dir}")
    else:
        # Try default
        default_cache = Path("cache/graphs")
        if default_cache.exists():
            st.session_state.graph_cache_dir = default_cache
            st.info(f"Using default graph cache: {default_cache}")
    
    return run_dir


def render_training_tab(run_dir: Path):
    """Render the Training progress tab."""
    st.header("📈 Training Progress")
    
    train_metrics = load_metrics_cached(run_dir, "train")
    val_metrics = load_metrics_cached(run_dir, "val")
    
    if train_metrics is None:
        st.warning("No training metrics found.")
        return
    
    # Determine loss type from config
    config = load_config(run_dir)
    loss_type = "pairwise_rank"
    if config:
        loss_type = getattr(config.loss, 'loss_type', 'pairwise_rank')
    
    # Metric selector
    metric_options = ["loss", "score_mean", "score_std", "grad_norm"]
    if loss_type in ["pairwise_rank", "rank", "ranking"]:
        metric_options.extend(["spearman", "pairwise_acc"])
    else:
        metric_options.extend(["mae", "rmse", "r2", "pearson"])
    
    selected_metrics = st.multiselect(
        "Select Metrics to Plot",
        options=metric_options,
        default=["loss", "spearman" if loss_type == "pairwise_rank" else "r2"]
    )
    
    if not selected_metrics:
        return
    
    # Create plots
    for metric in selected_metrics:
        if metric not in train_metrics.columns:
            continue
        
        fig = go.Figure()
        
        # Training curve
        fig.add_trace(go.Scatter(
            x=train_metrics["epoch"],
            y=train_metrics[metric],
            mode="lines",
            name="Train",
            line=dict(color="blue", width=2)
        ))
        
        # Validation curve
        if val_metrics is not None and metric in val_metrics.columns:
            fig.add_trace(go.Scatter(
                x=val_metrics["epoch"],
                y=val_metrics[metric],
                mode="lines",
                name="Validation",
                line=dict(color="red", width=2)
            ))
        
        fig.update_layout(
            title=f"{metric.upper()} vs Epoch",
            xaxis_title="Epoch",
            yaxis_title=metric.upper(),
            height=400,
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)


def render_evaluation_tab(run_dir: Path):
    """Render the Evaluation results tab."""
    st.header("📊 Evaluation Results")
    
    eval_preds = load_eval_predictions_cached(run_dir)
    
    if eval_preds is None:
        st.warning("No evaluation predictions found. Run evaluation first!")
        return
    
    # Scatter plot
    st.subheader("Predicted vs Ground Truth")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=eval_preds["y"],
        y=eval_preds["score"],
        mode="markers",
        marker=dict(size=8, opacity=0.6, color="blue"),
        name="Predictions",
        text=eval_preds["sample_id"],
        hovertemplate="<b>%{text}</b><br>y: %{x:.4f}<br>score: %{y:.4f}<extra></extra>"
    ))
    
    # Add diagonal line
    min_val = min(eval_preds["y"].min(), eval_preds["score"].min())
    max_val = max(eval_preds["y"].max(), eval_preds["score"].max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        name="Perfect prediction",
        line=dict(color="red", dash="dash", width=2)
    ))
    
    # Add trend line
    if len(eval_preds) > 1:
        z = np.polyfit(eval_preds["y"], eval_preds["score"], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min_val, max_val, 100)
        fig.add_trace(go.Scatter(
            x=x_trend,
            y=p(x_trend),
            mode="lines",
            name="Linear fit",
            line=dict(color="green", width=2)
        ))
    
    # Compute metrics
    from scipy.stats import spearmanr, pearsonr
    
    spearman_corr, spearman_p = spearmanr(eval_preds["y"], eval_preds["score"])
    pearson_corr, pearson_p = pearsonr(eval_preds["y"], eval_preds["score"])
    mae = np.mean(np.abs(eval_preds["score"] - eval_preds["y"]))
    rmse = np.sqrt(np.mean((eval_preds["score"] - eval_preds["y"]) ** 2))
    ss_res = np.sum((eval_preds["y"] - eval_preds["score"]) ** 2)
    ss_tot = np.sum((eval_preds["y"] - eval_preds["y"].mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    fig.update_layout(
        title=f"Predicted vs Ground Truth (n={len(eval_preds)})",
        xaxis_title="Ground Truth (y)",
        yaxis_title="Predicted Score",
        height=600,
        hovermode="closest"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics display
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Spearman ρ", f"{spearman_corr:.4f}")
    with col2:
        st.metric("Pearson r", f"{pearson_corr:.4f}")
    with col3:
        st.metric("MAE", f"{mae:.4f}")
    with col4:
        st.metric("RMSE", f"{rmse:.4f}")
    
    col5, col6 = st.columns(2)
    with col5:
        st.metric("R²", f"{r2:.4f}")
    with col6:
        st.metric("Samples", len(eval_preds))
    
    # Residual plot
    st.subheader("Residual Plot")
    residuals = eval_preds["score"] - eval_preds["y"]
    
    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(
        x=eval_preds["y"],
        y=residuals,
        mode="markers",
        marker=dict(size=8, opacity=0.6, color="purple"),
        name="Residuals"
    ))
    fig_res.add_hline(y=0, line_dash="dash", line_color="red")
    
    fig_res.update_layout(
        title="Residuals vs Ground Truth",
        xaxis_title="Ground Truth (y)",
        yaxis_title="Residuals (Predicted - Ground Truth)",
        height=400
    )
    
    st.plotly_chart(fig_res, use_container_width=True)
    
    # Distribution comparison
    st.subheader("Distribution Comparison")
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=eval_preds["y"],
        name="Ground Truth",
        opacity=0.7,
        nbinsx=20
    ))
    fig_dist.add_trace(go.Histogram(
        x=eval_preds["score"],
        name="Predictions",
        opacity=0.7,
        nbinsx=20
    ))
    fig_dist.update_layout(
        title="Distribution of Ground Truth vs Predictions",
        xaxis_title="Value",
        yaxis_title="Frequency",
        height=400,
        barmode="overlay"
    )
    st.plotly_chart(fig_dist, use_container_width=True)


@st.cache_data
def list_available_graphs_cached(cache_dir: Path):
    """Cached version of list_available_graphs."""
    return list_available_graphs(cache_dir)


def render_graph_viewer_tab(run_dir: Path):
    """Render the 3D Graph Viewer tab."""
    st.header("🔬 3D Graph Viewer")
    
    # Get cache directory
    cache_dir = st.session_state.graph_cache_dir
    if cache_dir is None:
        cache_dir = get_cache_dir(run_dir)
        if cache_dir is None:
            cache_dir = Path("cache/graphs")
    
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        st.error(f"Graph cache directory not found: {cache_dir}")
        st.info("Please specify the graph cache directory in the sidebar or ensure it exists.")
        return
    
    # List all available graphs in cache
    available_graphs = list_available_graphs_cached(cache_dir)
    
    if not available_graphs:
        st.warning(f"No cached graphs found in: {cache_dir}")
        st.info("Please ensure graphs have been cached. You can run the preprocessing step to cache graphs.")
        return
    
    # Sample selector - use sample_id from the graph list
    sample_options = [g["sample_id"] for g in available_graphs]
    selected_sample = st.selectbox(
        f"Select Sample (found {len(available_graphs)} graphs)",
        options=sample_options,
        index=0
    )
    
    # Find the selected graph's cache path
    selected_graph = next((g for g in available_graphs if g["sample_id"] == selected_sample), None)
    
    if selected_graph is None:
        st.error(f"Could not find graph for sample: {selected_sample}")
        return
    
    cache_path = Path(selected_graph["cache_path"])
    
    if not cache_path.exists():
        st.error(f"Cached graph file not found: {cache_path}")
        return
    
    # Load graph
    try:
        graph, metadata = load_graph_cached(cache_path)
        stats = get_graph_stats(graph, metadata)
    except Exception as e:
        st.error(f"Error loading graph: {e}")
        return
    
    # Display stats
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Nodes", stats["num_nodes"])
    with col2:
        st.metric("Edges", stats["num_edges"])
    with col3:
        st.metric("Chains", stats.get("num_chains", "N/A"))
    with col4:
        if "antibody_nodes" in stats:
            st.metric("Antibody Nodes", stats["antibody_nodes"])
    with col5:
        if "num_interface_nodes" in stats:
            st.metric("Interface Nodes", stats["num_interface_nodes"])
    
    # Edge type controls
    st.subheader("Edge Display Controls")
    col1, col2, col3 = st.columns(3)
    with col1:
        show_covalent = st.checkbox("Show COVALENT edges", value=True)
    with col2:
        show_noncovalent = st.checkbox("Show NONCOVALENT edges", value=True)
    with col3:
        show_interface_edges = st.checkbox("Show interface edges", value=False,
                                          help="Highlight edges connecting interface nodes across antibody-antigen boundary")
    
    max_noncovalent_edges = st.slider(
        "Max NONCOVALENT edges to display",
        min_value=100,
        max_value=10000,
        value=5000,
        step=100,
        help="Randomly sample noncovalent edges if there are too many"
    )
    
    # Interface highlighting
    highlight_interface = st.checkbox("Highlight interface nodes", value=False, 
                                     help="Highlight nodes that are part of the antibody-antigen interface")
    
    max_edge_dist = st.slider(
        "Max edge distance (Å)",
        min_value=0.0,
        max_value=20.0,
        value=15.0,
        step=0.5
    )
    
    # Create 3D plot
    st.subheader("3D Graph Visualization")
    
    # Extract node positions
    pos = graph.pos.numpy()
    # Support both chain_role (new) and chain_group (old) for backward compatibility
    chain_role_field = 'chain_role' if hasattr(graph, 'chain_role') else 'chain_group'
    chain_role = getattr(graph, chain_role_field).numpy()
    chain_id = graph.chain_id.numpy()
    
    # Get interface markers if available
    has_interface = hasattr(graph, 'is_interface')
    if has_interface:
        is_interface = graph.is_interface.numpy()
        min_inter_dist = graph.min_inter_dist.numpy() if hasattr(graph, 'min_inter_dist') else None
        inter_contact_count = graph.inter_contact_count.numpy() if hasattr(graph, 'inter_contact_count') else None
    else:
        is_interface = None
        min_inter_dist = None
        inter_contact_count = None
    
    # Get chain labels
    chain_labels = stats.get("chain_labels", {})
    chain_roles = stats.get("chain_roles", {})
    
    # Create node traces by chain role
    fig = go.Figure()
    
    # Antibody nodes
    antibody_mask = chain_role == 0
    if antibody_mask.any():
        antibody_indices = np.where(antibody_mask)[0]
        # Build hover text with interface info if available
        antibody_text = []
        for i in antibody_indices:
            text = f"Chain: {chain_labels.get(int(chain_id[i]), '?')}<br>Residue: {graph.res_id[i].item()}"
            if has_interface and is_interface is not None:
                if is_interface[i]:
                    text += "<br>Interface: Yes"
                    if min_inter_dist is not None:
                        text += f"<br>Min inter dist: {min_inter_dist[i]:.2f} Å"
                    if inter_contact_count is not None:
                        text += f"<br>Inter contacts: {int(inter_contact_count[i])}"
                else:
                    text += "<br>Interface: No"
            antibody_text.append(text)
        
        # Color by interface if highlighting
        if highlight_interface and has_interface and is_interface is not None:
            # Split into interface and non-interface
            ab_interface_mask = antibody_mask & (is_interface == 1)
            ab_noninterface_mask = antibody_mask & (is_interface == 0)
            
            if ab_interface_mask.any():
                ab_interface_indices = np.where(ab_interface_mask)[0]
                ab_interface_text = [antibody_text[i] for i in ab_interface_indices]
                fig.add_trace(go.Scatter3d(
                    x=pos[ab_interface_mask, 0],
                    y=pos[ab_interface_mask, 1],
                    z=pos[ab_interface_mask, 2],
                    mode="markers",
                    name="Antibody (Interface)",
                    marker=dict(size=7, color="cyan", opacity=0.9, line=dict(width=1, color="darkblue")),
                    text=ab_interface_text,
                    hovertemplate="%{text}<br>Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>"
                ))
            
            if ab_noninterface_mask.any():
                ab_noninterface_indices = np.where(ab_noninterface_mask)[0]
                ab_noninterface_text = [antibody_text[i] for i in ab_noninterface_indices]
                fig.add_trace(go.Scatter3d(
                    x=pos[ab_noninterface_mask, 0],
                    y=pos[ab_noninterface_mask, 1],
                    z=pos[ab_noninterface_mask, 2],
                    mode="markers",
                    name="Antibody (Non-interface)",
                    marker=dict(size=5, color="blue", opacity=0.6),
                    text=ab_noninterface_text,
                    hovertemplate="%{text}<br>Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>"
                ))
        else:
            fig.add_trace(go.Scatter3d(
                x=pos[antibody_mask, 0],
                y=pos[antibody_mask, 1],
                z=pos[antibody_mask, 2],
                mode="markers",
                name="Antibody",
                marker=dict(size=5, color="blue", opacity=0.8),
                text=antibody_text,
                hovertemplate="%{text}<br>Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>"
            ))
    
    # Antigen nodes
    antigen_mask = chain_role == 1
    if antigen_mask.any():
        antigen_indices = np.where(antigen_mask)[0]
        # Build hover text with interface info if available
        antigen_text = []
        for i in antigen_indices:
            text = f"Chain: {chain_labels.get(int(chain_id[i]), '?')}<br>Residue: {graph.res_id[i].item()}"
            if has_interface and is_interface is not None:
                if is_interface[i]:
                    text += "<br>Interface: Yes"
                    if min_inter_dist is not None:
                        text += f"<br>Min inter dist: {min_inter_dist[i]:.2f} Å"
                    if inter_contact_count is not None:
                        text += f"<br>Inter contacts: {int(inter_contact_count[i])}"
                else:
                    text += "<br>Interface: No"
            antigen_text.append(text)
        
        # Color by interface if highlighting
        if highlight_interface and has_interface and is_interface is not None:
            # Split into interface and non-interface
            ag_interface_mask = antigen_mask & (is_interface == 1)
            ag_noninterface_mask = antigen_mask & (is_interface == 0)
            
            if ag_interface_mask.any():
                ag_interface_indices = np.where(ag_interface_mask)[0]
                ag_interface_text = [antigen_text[i] for i in ag_interface_indices]
                fig.add_trace(go.Scatter3d(
                    x=pos[ag_interface_mask, 0],
                    y=pos[ag_interface_mask, 1],
                    z=pos[ag_interface_mask, 2],
                    mode="markers",
                    name="Antigen (Interface)",
                    marker=dict(size=7, color="yellow", opacity=0.9, line=dict(width=1, color="darkred")),
                    text=ag_interface_text,
                    hovertemplate="%{text}<br>Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>"
                ))
            
            if ag_noninterface_mask.any():
                ag_noninterface_indices = np.where(ag_noninterface_mask)[0]
                ag_noninterface_text = [antigen_text[i] for i in ag_noninterface_indices]
                fig.add_trace(go.Scatter3d(
                    x=pos[ag_noninterface_mask, 0],
                    y=pos[ag_noninterface_mask, 1],
                    z=pos[ag_noninterface_mask, 2],
                    mode="markers",
                    name="Antigen (Non-interface)",
                    marker=dict(size=5, color="red", opacity=0.6),
                    text=ag_noninterface_text,
                    hovertemplate="%{text}<br>Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>"
                ))
        else:
            fig.add_trace(go.Scatter3d(
                x=pos[antigen_mask, 0],
                y=pos[antigen_mask, 1],
                z=pos[antigen_mask, 2],
                mode="markers",
                name="Antigen",
                marker=dict(size=5, color="red", opacity=0.8),
                text=antigen_text,
                hovertemplate="%{text}<br>Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>"
            ))
    
    # Add edges
    edge_index = graph.edge_index.numpy()
    edge_type = graph.edge_type.numpy()
    edge_dist = graph.edge_dist.numpy()
    
    # Filter edges by distance
    dist_mask = edge_dist <= max_edge_dist
    
    # Identify interface edges (edges connecting interface nodes across antibody-antigen boundary)
    interface_edge_mask = None
    if show_interface_edges and has_interface and is_interface is not None:
        # Interface edges: both nodes are interface nodes AND they cross antibody-antigen boundary
        interface_edge_mask = np.zeros(edge_index.shape[1], dtype=bool)
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            # Both nodes must be interface nodes
            if is_interface[src] == 1 and is_interface[dst] == 1:
                # Must cross antibody-antigen boundary
                if chain_role[src] != chain_role[dst]:
                    interface_edge_mask[i] = True
    
    # Interface edges (highlighted separately if requested)
    if show_interface_edges and interface_edge_mask is not None and interface_edge_mask.any():
        interface_edge_mask_filtered = interface_edge_mask & dist_mask
        if interface_edge_mask_filtered.any():
            x_edges, y_edges, z_edges = [], [], []
            for i in range(edge_index.shape[1]):
                if interface_edge_mask_filtered[i]:
                    src, dst = edge_index[0, i], edge_index[1, i]
                    x_edges.extend([pos[src, 0], pos[dst, 0], None])
                    y_edges.extend([pos[src, 1], pos[dst, 1], None])
                    z_edges.extend([pos[src, 2], pos[dst, 2], None])
            
            if x_edges:
                fig.add_trace(go.Scatter3d(
                    x=x_edges,
                    y=y_edges,
                    z=z_edges,
                    mode="lines",
                    name="Interface edges",
                    line=dict(color="purple", width=4),
                    hoverinfo="skip"
                ))
    
    # COVALENT edges (type 0) - exclude interface edges if showing them separately
    if show_covalent:
        covalent_mask = (edge_type == 0) & dist_mask
        # Exclude interface edges if they're being shown separately
        if show_interface_edges and interface_edge_mask is not None:
            covalent_mask = covalent_mask & (~interface_edge_mask)
        if covalent_mask.any():
            x_edges, y_edges, z_edges = [], [], []
            for i in range(edge_index.shape[1]):
                if covalent_mask[i]:
                    src, dst = edge_index[0, i], edge_index[1, i]
                    x_edges.extend([pos[src, 0], pos[dst, 0], None])
                    y_edges.extend([pos[src, 1], pos[dst, 1], None])
                    z_edges.extend([pos[src, 2], pos[dst, 2], None])
            
            if x_edges:
                fig.add_trace(go.Scatter3d(
                    x=x_edges,
                    y=y_edges,
                    z=z_edges,
                    mode="lines",
                    name="COVALENT edges",
                    line=dict(color="green", width=3),
                    hoverinfo="skip"
                ))
    
    # NONCOVALENT edges (type 1, with sampling) - exclude interface edges if showing them separately
    if show_noncovalent:
        noncovalent_mask = (edge_type == 1) & dist_mask
        # Exclude interface edges if they're being shown separately
        if show_interface_edges and interface_edge_mask is not None:
            noncovalent_mask = noncovalent_mask & (~interface_edge_mask)
        if noncovalent_mask.any():
            noncovalent_indices = np.where(noncovalent_mask)[0]
            if len(noncovalent_indices) > max_noncovalent_edges:
                np.random.seed(42)  # Deterministic sampling
                noncovalent_indices = np.random.choice(noncovalent_indices, max_noncovalent_edges, replace=False)
            
            x_edges, y_edges, z_edges = [], [], []
            for idx in noncovalent_indices:
                src, dst = edge_index[0, idx], edge_index[1, idx]
                x_edges.extend([pos[src, 0], pos[dst, 0], None])
                y_edges.extend([pos[src, 1], pos[dst, 1], None])
                z_edges.extend([pos[src, 2], pos[dst, 2], None])
            
            if x_edges:
                fig.add_trace(go.Scatter3d(
                    x=x_edges,
                    y=y_edges,
                    z=z_edges,
                    mode="lines",
                    name="NONCOVALENT edges",
                    line=dict(color="gray", width=1),
                    hoverinfo="skip"
                ))
    
    fig.update_layout(
        title=f"3D Graph: {selected_sample}",
        scene=dict(
            xaxis_title="X (Å)",
            yaxis_title="Y (Å)",
            zaxis_title="Z (Å)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        height=800
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Edge type stats
    st.subheader("Edge Statistics")
    if "edge_type_counts" in stats:
        edge_df = pd.DataFrame([
            {"Type": k, "Count": v} for k, v in stats["edge_type_counts"].items()
        ])
        st.dataframe(edge_df, use_container_width=True)


def main():
    """Main app entry point."""
    st.title("🔬 GAT Ranking Model Dashboard")
    
    # Parse args (from sys.argv for streamlit)
    default_runs_dir, default_graph_cache_dir = parse_args()
    runs_dir = Path(default_runs_dir)
    graph_cache_dir = Path(default_graph_cache_dir)
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        runs_dir_input = st.text_input("Runs Directory", value=str(runs_dir))
        graph_cache_dir_input = st.text_input("Graph Cache Directory", value=str(graph_cache_dir))
        
        runs_dir = Path(runs_dir_input)
        st.session_state.graph_cache_dir = Path(graph_cache_dir_input) if graph_cache_dir_input else None
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Runs", "📈 Training", "📊 Evaluation", "🔬 Graph Viewer"])
    
    with tab1:
        run_dir = render_runs_tab(runs_dir)
    
    with tab2:
        if st.session_state.selected_run:
            render_training_tab(st.session_state.selected_run)
        else:
            st.info("Please select a run in the 'Runs' tab first.")
    
    with tab3:
        if st.session_state.selected_run:
            render_evaluation_tab(st.session_state.selected_run)
        else:
            st.info("Please select a run in the 'Runs' tab first.")
    
    with tab4:
        if st.session_state.selected_run:
            render_graph_viewer_tab(st.session_state.selected_run)
        else:
            st.info("Please select a run in the 'Runs' tab first.")


if __name__ == "__main__":
    main()

