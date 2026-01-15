"""
Convert PDB files to PyTorch Geometric Data objects.
Uses C-alpha atoms only for graph construction.
"""
import numpy as np
import torch
from torch_geometric.data import Data
from Bio.PDB import PDBParser, Selection
from typing import List, Optional, Tuple, Dict
from pathlib import Path
import warnings

# Amino acid to integer mapping (20 standard AAs)
AA_TO_INT = {
    "ALA": 0, "ARG": 1, "ASN": 2, "ASP": 3, "CYS": 4,
    "GLN": 5, "GLU": 6, "GLY": 7, "HIS": 8, "ILE": 9,
    "LEU": 10, "LYS": 11, "MET": 12, "PHE": 13, "PRO": 14,
    "SER": 15, "THR": 16, "TRP": 17, "TYR": 18, "VAL": 19
}
UNKNOWN_AA = 20  # For non-standard residues


def get_residue_type(residue) -> int:
    """Get integer encoding for residue type."""
    resname = residue.get_resname().strip()
    return AA_TO_INT.get(resname, UNKNOWN_AA)


def extract_ca_coordinates(
    structure,
    chain_ids: List[str]
) -> Tuple[np.ndarray, List[Tuple[str, int]], List[int]]:
    """
    Extract C-alpha coordinates for specified chains.
    
    Args:
        structure: Bio.PDB Structure object
        chain_ids: List of chain IDs to extract
        
    Returns:
        coords: (N, 3) array of C-alpha coordinates
        residue_info: List of (chain_id, residue_number) tuples
        residue_types: List of residue type integers
    """
    coords = []
    residue_info = []
    residue_types = []
    
    for chain_id in chain_ids:
        try:
            chain = structure[0][chain_id]
        except KeyError:
            warnings.warn(f"Chain {chain_id} not found in structure")
            continue
        
        for residue in chain:
            # Skip heteroatoms (water, ions, etc.)
            if residue.id[0] != " ":
                continue
            
            try:
                ca = residue["CA"]
                coords.append(ca.get_coord())
                residue_info.append((chain_id, residue.id[1]))
                residue_types.append(get_residue_type(residue))
            except KeyError:
                # No C-alpha atom (e.g., missing residue)
                continue
    
    if len(coords) == 0:
        raise ValueError("No C-alpha atoms found in specified chains")
    
    return np.array(coords), residue_info, residue_types


def compute_pairwise_distances(coords: np.ndarray) -> np.ndarray:
    """Compute pairwise C-alpha distances efficiently."""
    # coords: (N, 3)
    # Returns: (N, N) distance matrix
    diff = coords[:, None, :] - coords[None, :, :]  # (N, N, 3)
    distances = np.linalg.norm(diff, axis=2)  # (N, N)
    return distances


def build_node_features(
    residue_types: List[int],
    chain_ids: List[str],
    antibody_chains: List[str],
    include_residue_index: bool = True
) -> torch.Tensor:
    """
    Build node features: [residue_type (20-dim one-hot), chain_type (1-dim), residue_index (1-dim, optional)].
    
    Args:
        residue_types: List of residue type integers
        chain_ids: List of chain IDs for each residue
        antibody_chains: List of chain IDs that are antibodies
        include_residue_index: Whether to include normalized residue index
        
    Returns:
        node_features: (N, feature_dim) tensor
    """
    N = len(residue_types)
    
    # Residue type: one-hot encoding (20 dims) + unknown (1 dim) = 21 dims
    residue_onehot = torch.zeros(N, 21)
    for i, res_type in enumerate(residue_types):
        residue_onehot[i, res_type] = 1.0
    
    # Chain type: 1 if antibody, 0 if antigen (1 dim)
    chain_type = torch.tensor([
        1.0 if chain_id in antibody_chains else 0.0
        for chain_id in chain_ids
    ], dtype=torch.float32).unsqueeze(1)
    
    features = [residue_onehot, chain_type]
    
    # Optional: normalized residue index
    if include_residue_index:
        residue_indices = torch.arange(N, dtype=torch.float32) / max(N - 1, 1)
        features.append(residue_indices.unsqueeze(1))
    
    return torch.cat(features, dim=1)


def build_edges(
    distances: np.ndarray,
    chain_ids: List[str],
    antibody_chains: List[str],
    antigen_chains: List[str],
    bound_cutoff: float = 8.0,
    unbound_cutoff: float = 10.0,
    use_sequential_edges: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build edges and edge attributes.
    
    Edge types:
    - 0: BOUND (antibody-antigen contact)
    - 1: UNBOUND (within-chain spatial proximity)
    - 2: SEQUENTIAL (i, i+1) if enabled
    
    Args:
        distances: (N, N) distance matrix
        chain_ids: List of chain IDs for each residue
        antibody_chains: List of antibody chain IDs
        antigen_chains: List of antigen chain IDs
        bound_cutoff: Distance cutoff for BOUND edges
        unbound_cutoff: Distance cutoff for UNBOUND edges
        use_sequential_edges: Whether to include sequential edges
        
    Returns:
        edge_index: (2, E) tensor of edge indices
        edge_attr: (E, 2) tensor [edge_type, normalized_distance]
    """
    N = distances.shape[0]
    edges = []
    edge_attrs = []
    
    # Determine chain membership for each node
    is_antibody = np.array([cid in antibody_chains for cid in chain_ids])
    is_antigen = np.array([cid in antigen_chains for cid in chain_ids])
    
    # Maximum distance for normalization (use unbound_cutoff as reference)
    max_dist = max(bound_cutoff, unbound_cutoff) * 1.5
    
    for i in range(N):
        for j in range(i + 1, N):  # Upper triangle only
            dist = distances[i, j]
            chain_i = chain_ids[i]
            chain_j = chain_ids[j]
            
            # BOUND edges: cross-interface (antibody <-> antigen) within cutoff
            if (is_antibody[i] and is_antigen[j]) or (is_antigen[i] and is_antibody[j]):
                if dist <= bound_cutoff:
                    edges.append([i, j])
                    edges.append([j, i])  # Undirected
                    edge_type = 0  # BOUND
                    edge_attrs.append([edge_type, dist / max_dist])
                    edge_attrs.append([edge_type, dist / max_dist])
            
            # UNBOUND edges: within-chain spatial proximity
            elif chain_i == chain_j and dist <= unbound_cutoff:
                edges.append([i, j])
                edges.append([j, i])  # Undirected
                edge_type = 1  # UNBOUND
                edge_attrs.append([edge_type, dist / max_dist])
                edge_attrs.append([edge_type, dist / max_dist])
    
    # Sequential edges: (i, i+1) within same chain
    if use_sequential_edges:
        for i in range(N - 1):
            if chain_ids[i] == chain_ids[i + 1]:
                dist = distances[i, i + 1]
                edges.append([i, i + 1])
                edges.append([i + 1, i])  # Undirected
                edge_type = 2  # SEQUENTIAL
                edge_attrs.append([edge_type, dist / max_dist])
                edge_attrs.append([edge_type, dist / max_dist])
    
    if len(edges) == 0:
        warnings.warn("No edges found in graph!")
        # Create self-loops to avoid empty graph
        edges = [[i, i] for i in range(N)]
        edge_attrs = [[0, 0.0] for _ in range(N)]
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    
    return edge_index, edge_attr


def pdb_to_graph(
    pdb_path: str,
    antibody_chains: List[str],
    antigen_chains: List[str],
    bound_cutoff: float = 8.0,
    unbound_cutoff: float = 10.0,
    use_sequential_edges: bool = False,
    include_residue_index: bool = True,
    y: Optional[float] = None
) -> Data:
    """
    Convert PDB file to PyTorch Geometric Data object.
    
    Args:
        pdb_path: Path to PDB file
        antibody_chains: List of antibody chain IDs (e.g., ["H", "L"])
        antigen_chains: List of antigen chain IDs (e.g., ["A"])
        bound_cutoff: Distance cutoff for BOUND edges (Angstroms)
        unbound_cutoff: Distance cutoff for UNBOUND edges (Angstroms)
        use_sequential_edges: Whether to include sequential edges
        include_residue_index: Whether to include residue index in node features
        y: Optional target value (binding property)
        
    Returns:
        torch_geometric.data.Data object
    """
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")
    
    # Parse PDB
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", str(pdb_path))
    
    # Extract all specified chains
    all_chains = antibody_chains + antigen_chains
    
    # Extract C-alpha coordinates
    coords, residue_info, residue_types = extract_ca_coordinates(structure, all_chains)
    
    # Extract chain IDs for each residue
    chain_ids = [info[0] for info in residue_info]
    
    # Compute pairwise distances
    distances = compute_pairwise_distances(coords)
    
    # Build node features
    node_features = build_node_features(
        residue_types,
        chain_ids,
        antibody_chains,
        include_residue_index=include_residue_index
    )
    
    # Build edges
    edge_index, edge_attr = build_edges(
        distances,
        chain_ids,
        antibody_chains,
        antigen_chains,
        bound_cutoff=bound_cutoff,
        unbound_cutoff=unbound_cutoff,
        use_sequential_edges=use_sequential_edges
    )
    
    # Create Data object
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([y], dtype=torch.float32) if y is not None else None,
        num_nodes=len(coords)
    )
    
    return data

