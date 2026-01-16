"""
Run logging utilities for saving training/evaluation artifacts.
"""
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import subprocess


def get_run_id(git_short: bool = True) -> str:
    """
    Generate a run ID: YYYYMMDD_HHMMSS_<gitshort(optional)>
    
    Args:
        git_short: If True, append git short hash if available
    
    Returns:
        Run ID string
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if git_short:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                git_hash = result.stdout.strip()
                return f"{timestamp}_{git_hash}"
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
    
    return timestamp


def create_run_dir(runs_dir: Path, run_id: Optional[str] = None) -> Path:
    """
    Create a run directory.
    
    Args:
        runs_dir: Base directory for runs
        run_id: Optional run ID (generated if None)
    
    Returns:
        Path to run directory
    """
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    if run_id is None:
        run_id = get_run_id()
    
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    return run_dir


def save_config(run_dir: Path, config: Any, format: str = "yaml") -> Path:
    """
    Save config to run directory.
    
    Args:
        run_dir: Run directory
        config: Config object (must have to_yaml method)
        format: "yaml" or "json"
    
    Returns:
        Path to saved config file
    """
    if format == "yaml":
        config_path = run_dir / "config.yaml"
        config.to_yaml(str(config_path))
    else:
        config_path = run_dir / "config.json"
        # Convert config to dict and save as JSON
        config_dict = {
            "graph": config.graph.__dict__,
            "model": config.model.__dict__,
            "loss": config.loss.__dict__,
            "training": config.training.__dict__,
            "data": config.data.__dict__
        }
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    return config_path


def init_metrics_csv(run_dir: Path, split: str, loss_type: str) -> Path:
    """
    Initialize metrics CSV file with headers.
    
    Args:
        run_dir: Run directory
        split: "train" or "val"
        loss_type: Loss type for determining columns
    
    Returns:
        Path to CSV file
    """
    csv_path = run_dir / f"metrics_{split}.csv"
    
    # Base columns
    columns = ["epoch", "split", "loss", "score_mean", "score_std", "grad_norm", "lr"]
    
    # Add loss-specific columns
    if loss_type in ["pairwise_rank", "rank", "ranking"]:
        columns.extend(["spearman", "pairwise_acc", "num_pairs"])
    else:
        columns.extend(["mae", "rmse", "r2", "pearson"])
    
    # Write header
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
    
    return csv_path


def append_metrics(
    csv_path: Path,
    epoch: int,
    split: str,
    metrics: Dict[str, Any],
    loss_type: str,
    lr: Optional[float] = None
):
    """
    Append metrics for one epoch to CSV.
    
    Args:
        csv_path: Path to metrics CSV
        epoch: Epoch number
        split: "train" or "val"
        metrics: Metrics dictionary
        loss_type: Loss type
        lr: Learning rate (optional)
    """
    row = [
        epoch,
        split,
        metrics.get("loss", 0.0),
        metrics.get("score_mean", 0.0),
        metrics.get("score_std", 0.0),
        metrics.get("grad_norm_mean", 0.0),
        lr if lr is not None else ""
    ]
    
    if loss_type in ["pairwise_rank", "rank", "ranking"]:
        row.extend([
            metrics.get("spearman", 0.0),
            metrics.get("pairwise_acc", 0.0),
            ""  # num_pairs (could be computed if needed)
        ])
    else:
        row.extend([
            metrics.get("mae", 0.0),
            metrics.get("rmse", 0.0),
            metrics.get("r2", 0.0),
            metrics.get("pearson", 0.0)
        ])
    
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def save_cache_link(run_dir: Path, graph_cache_dir: Path):
    """
    Save a link/reference to the graph cache directory.
    
    Args:
        run_dir: Run directory
        graph_cache_dir: Path to graph cache directory
    """
    link_path = run_dir / "cache_link.txt"
    with open(link_path, "w") as f:
        f.write(str(Path(graph_cache_dir).resolve()))


def save_eval_predictions(
    run_dir: Path,
    sample_ids: list,
    targets: list,
    predictions: list
) -> Path:
    """
    Save evaluation predictions to CSV.
    
    Args:
        run_dir: Run directory
        sample_ids: List of sample IDs (pdb_path or hashed id)
        targets: List of target values
        predictions: List of predicted scores
    
    Returns:
        Path to saved CSV
    """
    csv_path = run_dir / "eval_predictions.csv"
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "y", "score"])
        for sid, y, score in zip(sample_ids, targets, predictions):
            writer.writerow([sid, y, score])
    
    return csv_path

