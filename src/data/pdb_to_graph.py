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
) -> Tuple[np.ndarray, List[Tuple[str, int, str]], List[int]]:
    """
    Extract C-alpha coordinates for specified chains.
    
    Args:
        structure: Bio.PDB Structure object
        chain_ids: List of chain IDs to extract
        
    Returns:
        coords: (N, 3) array of C-alpha coordinates
        residue_info: List of (chain_id, resseq, insertion_code) tuples
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
                # residue.id is (hetero_flag, resseq, insertion_code)
                resseq = residue.id[1]
                insertion_code = residue.id[2] if len(residue.id) > 2 else " "
                residue_info.append((chain_id, resseq, insertion_code))
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
    include_residue_index: bool = True,
    interface_features: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Build node features: [residue_type (21-dim one-hot), chain_type (1-dim), residue_index (1-dim, optional), interface_features (optional)].
    
    Args:
        residue_types: List of residue type integers
        chain_ids: List of chain IDs for each residue
        antibody_chains: List of chain IDs that are antibodies
        include_residue_index: Whether to include normalized residue index
        interface_features: Optional tensor of interface features to concatenate (N, interface_dim)
        
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
    
    # Optional: interface features
    if interface_features is not None:
        features.append(interface_features)
    
    return torch.cat(features, dim=1)


def sort_residues_by_chain(
    chain_ids: List[str],
    residue_info: List[Tuple[str, int, str]]
) -> List[int]:
    """
    Sort residues within each chain by (resseq, insertion_code) and return sorted indices.
    
    Args:
        chain_ids: List of chain IDs for each residue (in original order)
        residue_info: List of (chain_id, resseq, insertion_code) tuples
        
    Returns:
        sorted_indices: List of indices that sort residues within each chain
    """
    N = len(chain_ids)
    indices = list(range(N))
    
    # Group by chain
    chain_groups = {}
    for i, chain_id in enumerate(chain_ids):
        if chain_id not in chain_groups:
            chain_groups[chain_id] = []
        chain_groups[chain_id].append(i)
    
    # Sort within each chain by (resseq, insertion_code)
    sorted_indices = []
    for chain_id in sorted(chain_groups.keys()):
        chain_indices = chain_groups[chain_id]
        # Sort by (resseq, insertion_code)
        chain_indices_sorted = sorted(
            chain_indices,
            key=lambda i: (residue_info[i][1], residue_info[i][2] or " ")
        )
        sorted_indices.extend(chain_indices_sorted)
    
    # Create mapping from original index to sorted index
    index_map = {sorted_idx: orig_idx for orig_idx, sorted_idx in enumerate(sorted_indices)}
    # Return indices that map original -> sorted
    return [index_map[i] for i in range(N)]


def build_edges(
    distances: np.ndarray,
    chain_ids: List[str],
    residue_info: List[Tuple[str, int, str]],
    use_covalent_edges: bool = True,
    use_noncovalent_edges: bool = True,
    noncovalent_cutoff: float = 10.0,
    allow_duplicate_edges: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build edges and edge attributes.
    
    Edge types:
    - 0: COVALENT (peptide backbone connectivity, sequential residues within chain)
    - 1: NONCOVALENT (distance-based contacts)
    
    Args:
        distances: (N, N) distance matrix
        chain_ids: List of chain IDs for each residue
        residue_info: List of (chain_id, resseq, insertion_code) tuples
        use_covalent_edges: Whether to include covalent edges
        use_noncovalent_edges: Whether to include noncovalent edges
        noncovalent_cutoff: Distance cutoff for NONCOVALENT edges (Angstroms)
        allow_duplicate_edges: If False, exclude covalent pairs from noncovalent edges
        
    Returns:
        edge_index: (2, E) tensor of edge indices
        edge_attr: (E, 1) tensor [normalized_distance]
        edge_type: (E,) tensor of edge types (0=covalent, 1=noncovalent)
        edge_dist: (E,) tensor of edge distances in Angstroms
    """
    N = distances.shape[0]
    edges = []
    edge_attrs = []
    edge_types_list = []
    edge_dists_list = []
    
    # Maximum distance for normalization
    max_dist = noncovalent_cutoff * 1.5
    
    # Build covalent edges first (within each chain, sequential residues)
    covalent_pairs = set()
    
    if use_covalent_edges:
        # Group residues by chain
        chain_groups = {}
        for i, chain_id in enumerate(chain_ids):
            if chain_id not in chain_groups:
                chain_groups[chain_id] = []
            chain_groups[chain_id].append(i)
        
        # For each chain, sort by (resseq, insertion_code) and connect consecutive residues
        for chain_id, chain_indices in chain_groups.items():
            # Sort indices by residue order
            chain_indices_sorted = sorted(
                chain_indices,
                key=lambda i: (residue_info[i][1], residue_info[i][2] or " ")
            )
            
            # Connect consecutive residues
            for k in range(len(chain_indices_sorted) - 1):
                i = chain_indices_sorted[k]
                j = chain_indices_sorted[k + 1]
                dist = distances[i, j]
                
                # Add both directions (undirected)
                edges.append([i, j])
                edges.append([j, i])
                edge_type = 0  # COVALENT
                edge_attrs.append([dist / max_dist])
                edge_attrs.append([dist / max_dist])
                edge_types_list.append(edge_type)
                edge_types_list.append(edge_type)
                edge_dists_list.append(dist)
                edge_dists_list.append(dist)
    
                # Track covalent pairs
                covalent_pairs.add((min(i, j), max(i, j)))
    
    # Build noncovalent edges (distance-based, all pairs within cutoff)
    if use_noncovalent_edges:
        for i in range(N):
            for j in range(i + 1, N):  # Upper triangle only
                # Skip if already covalent and duplicates not allowed
                if not allow_duplicate_edges and (i, j) in covalent_pairs:
                    continue
                
                dist = distances[i, j]
                if dist <= noncovalent_cutoff:
                    # Add both directions (undirected)
                    edges.append([i, j])
                    edges.append([j, i])
                    edge_type = 1  # NONCOVALENT
                    edge_attrs.append([dist / max_dist])
                    edge_attrs.append([dist / max_dist])
                edge_types_list.append(edge_type)
                edge_types_list.append(edge_type)
                edge_dists_list.append(dist)
                edge_dists_list.append(dist)
    
    if len(edges) == 0:
        warnings.warn("No edges found in graph!")
        # Create self-loops to avoid empty graph
        edges = [[i, i] for i in range(N)]
        edge_attrs = [[0.0] for _ in range(N)]
        edge_types_list = [0] * N
        edge_dists_list = [0.0] * N
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    edge_type = torch.tensor(edge_types_list, dtype=torch.int64)
    edge_dist = torch.tensor(edge_dists_list, dtype=torch.float32)
    
    return edge_index, edge_attr, edge_type, edge_dist


def compute_interface_markers(
    distances: np.ndarray,
    chain_role: np.ndarray,
    interface_cutoff: float = 8.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute interface markers for each node based on geometry.
    
    Args:
        distances: (N, N) distance matrix
        chain_role: (N,) array where 0=antibody, 1=antigen
        interface_cutoff: Distance cutoff for interface definition (Angstroms)
        
    Returns:
        min_inter_dist: (N,) tensor of minimum distance to opposite molecule
        inter_contact_count: (N,) tensor of count of opposite-molecule nodes within cutoff
        is_interface: (N,) tensor of binary interface markers (0/1)
    """
    N = distances.shape[0]
    
    # Create masks for antibody and antigen nodes
    is_antibody = (chain_role == 0)
    is_antigen = (chain_role == 1)
    
    # Check if we have both antibody and antigen nodes
    if not is_antibody.any() or not is_antigen.any():
        # No interface possible
        min_inter_dist = torch.full((N,), float('inf'), dtype=torch.float32)
        inter_contact_count = torch.zeros(N, dtype=torch.float32)
        is_interface = torch.zeros(N, dtype=torch.int64)
        return min_inter_dist, inter_contact_count, is_interface
    
    # Compute minimum distance to opposite molecule for each node
    min_inter_dist = torch.full((N,), float('inf'), dtype=torch.float32)
    inter_contact_count = torch.zeros(N, dtype=torch.float32)
    
    distances_tensor = torch.from_numpy(distances).float()
    
    for i in range(N):
        if is_antibody[i]:
            # Find distances to all antigen nodes
            antigen_dists = distances_tensor[i, is_antigen]
            if len(antigen_dists) > 0:
                min_inter_dist[i] = antigen_dists.min()
                inter_contact_count[i] = (antigen_dists <= interface_cutoff).sum().float()
        elif is_antigen[i]:
            # Find distances to all antibody nodes
            antibody_dists = distances_tensor[i, is_antibody]
            if len(antibody_dists) > 0:
                min_inter_dist[i] = antibody_dists.min()
                inter_contact_count[i] = (antibody_dists <= interface_cutoff).sum().float()
    
    # Replace inf with a large value (e.g., 2 * interface_cutoff)
    min_inter_dist = torch.where(
        torch.isinf(min_inter_dist),
        torch.full_like(min_inter_dist, 2.0 * interface_cutoff),
        min_inter_dist
    )
    
    # Binary interface marker
    is_interface = (inter_contact_count > 0).long()
    
    return min_inter_dist, inter_contact_count, is_interface


def pdb_to_graph(
    pdb_path: str,
    antibody_chains: List[str],
    antigen_chains: List[str],
    noncovalent_cutoff: float = 10.0,
    interface_cutoff: float = 8.0,
    use_covalent_edges: bool = True,
    use_noncovalent_edges: bool = True,
    allow_duplicate_edges: bool = False,
    include_residue_index: bool = True,
    add_interface_features_to_x: bool = True,
    y: Optional[float] = None
) -> Data:
    """
    Convert PDB file to PyTorch Geometric Data object.
    
    Args:
        pdb_path: Path to PDB file
        antibody_chains: List of antibody chain IDs (e.g., ["H", "L"])
        antigen_chains: List of antigen chain IDs (e.g., ["A"])
        noncovalent_cutoff: Distance cutoff for NONCOVALENT edges (Angstroms)
        interface_cutoff: Distance cutoff for interface definition (Angstroms)
        use_covalent_edges: Whether to include covalent edges
        use_noncovalent_edges: Whether to include noncovalent edges
        allow_duplicate_edges: If False, exclude covalent pairs from noncovalent edges
        include_residue_index: Whether to include residue index in node features
        add_interface_features_to_x: Whether to add interface features to node features
        y: Optional target value (binding property)
        
    Returns:
        torch_geometric.data.Data object
    """
    # #region agent log
    import json
    import time
    from pathlib import Path as PathLib
    log_path = PathLib("/home/wenkanl2/Ken/GAT_regression/.cursor/debug.log")
    def log_debug(location, message, data, hypothesis_id=None):
        entry = {
            "sessionId": "debug-session",
            "runId": "pre-fix",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000)
        }
        try:
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except:
            pass  # Ignore logging errors

    log_debug("pdb_to_graph.py:pdb_to_graph", "Function entry", {
        "pdb_path": str(pdb_path),
        "antibody_chains": antibody_chains,
        "antigen_chains": antigen_chains,
        "noncovalent_cutoff": noncovalent_cutoff,
        "interface_cutoff": interface_cutoff,
        "use_covalent_edges": use_covalent_edges,
        "use_noncovalent_edges": use_noncovalent_edges,
        "allow_duplicate_edges": allow_duplicate_edges
    }, "A")
    # #endregion agent log

    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")
    
    # Parse PDB
    # #region agent log
    log_debug("pdb_to_graph.py:pdb_to_graph", "Before PDB parsing", {
        "pdb_path": str(pdb_path),
        "pdb_exists": pdb_path.exists(),
        "pdb_size": pdb_path.stat().st_size if pdb_path.exists() else None
    }, "A")
    # #endregion agent log

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", str(pdb_path))
    
    # #region agent log
    log_debug("pdb_to_graph.py:pdb_to_graph", "PDB parsing completed", {
        "pdb_path": str(pdb_path),
        "num_models": len(structure),
        "chains_in_model_0": [chain.id for chain in structure[0].get_chains()] if len(structure) > 0 else [],
        "antibody_chains": antibody_chains,
        "antigen_chains": antigen_chains
    }, "A")
    # #endregion agent log
    
    # Check which chains actually exist in the PDB
    available_chains = [chain.id for chain in structure[0].get_chains()]
    missing_ab_chains = [c for c in antibody_chains if c not in available_chains]
    missing_ag_chains = [c for c in antigen_chains if c not in available_chains]

    # Handle missing chains by issuing warnings and trying to infer
    corrected_antibody_chains = [c for c in antibody_chains if c in available_chains]
    corrected_antigen_chains = [c for c in antigen_chains if c in available_chains]

    if missing_ab_chains:
        warnings.warn(f"Antibody chains {missing_ab_chains} not found in {pdb_path}, available: {available_chains}")
    if missing_ag_chains:
        warnings.warn(f"Antigen chains {missing_ag_chains} not found in {pdb_path}, available: {available_chains}")
        # Try to infer antigen chains as remaining chains not in antibody
        if corrected_antibody_chains:
            potential_antigen = [c for c in available_chains if c not in corrected_antibody_chains]
            if potential_antigen:
                corrected_antigen_chains = potential_antigen
                warnings.warn(f"Inferred antigen chains as {corrected_antigen_chains} for {pdb_path}")

    # Extract all specified chains (using corrected lists)
    all_chains = corrected_antibody_chains + corrected_antigen_chains

    # #region agent log
    log_debug("pdb_to_graph.py:pdb_to_graph", "Before coordinate extraction", {
        "original_antibody_chains": antibody_chains,
        "original_antigen_chains": antigen_chains,
        "corrected_antibody_chains": corrected_antibody_chains,
        "corrected_antigen_chains": corrected_antigen_chains,
        "all_chains": all_chains,
        "available_chains": available_chains
    }, "B")
    # #endregion agent log
    
    # Extract C-alpha coordinates
    coords, residue_info, residue_types = extract_ca_coordinates(structure, all_chains)
    
    # #region agent log
    log_debug("pdb_to_graph.py:pdb_to_graph", "Coordinate extraction completed", {
        "num_coords": len(coords),
        "num_residue_info": len(residue_info),
        "num_residue_types": len(residue_types),
        "coords_shape": coords.shape if hasattr(coords, 'shape') else None
    }, "B")
    # #endregion agent log
    
    # Extract chain IDs, residue numbers, and insertion codes
    chain_ids = [info[0] for info in residue_info]
    residue_numbers = [info[1] for info in residue_info]
    insertion_codes = [info[2] if info[2] and info[2] != " " else " " for info in residue_info]
    
    # Create chain ID to integer mapping
    unique_chains = sorted(set(chain_ids))
    chain_id_map = {chain: idx for idx, chain in enumerate(unique_chains)}
    chain_ids_int = [chain_id_map[cid] for cid in chain_ids]
    
    # Create chain role mapping (0=antibody, 1=antigen) using corrected chains
    chain_roles = [0 if cid in corrected_antibody_chains else 1 for cid in chain_ids]
    
    # Encode insertion codes as integers (0 for blank, ord value otherwise)
    insertion_code_ints = [0 if ic == " " else ord(ic) for ic in insertion_codes]
    
    # Compute pairwise distances
    distances = compute_pairwise_distances(coords)
    
    # Compute interface markers
    # #region agent log
    log_debug("pdb_to_graph.py:pdb_to_graph", "Before interface markers computation", {
        "distances_shape": distances.shape,
        "chain_roles": chain_roles,
        "interface_cutoff": interface_cutoff
    }, "C")
    # #endregion agent log

    chain_role_array = np.array(chain_roles)
    min_inter_dist, inter_contact_count, is_interface = compute_interface_markers(
        distances, chain_role_array, interface_cutoff
    )

    # #region agent log
    log_debug("pdb_to_graph.py:pdb_to_graph", "Interface markers computed", {
        "min_inter_dist_shape": min_inter_dist.shape,
        "inter_contact_count_shape": inter_contact_count.shape,
        "is_interface_shape": is_interface.shape,
        "num_interface_nodes": is_interface.sum().item()
    }, "C")
    # #endregion agent log
    
    # Prepare interface features for node features (if requested)
    interface_features = None
    if add_interface_features_to_x:
        # Normalize min_inter_dist by interface_cutoff and clip to [0, 2]
        min_inter_dist_norm = torch.clamp(min_inter_dist / interface_cutoff, 0.0, 2.0)
        # Normalize inter_contact_count by log1p
        inter_contact_count_norm = torch.log1p(inter_contact_count)
        interface_features = torch.stack([min_inter_dist_norm, inter_contact_count_norm], dim=1)
    
    # Build node features
    node_features = build_node_features(
        residue_types,
        chain_ids,
        antibody_chains,
        include_residue_index=include_residue_index,
        interface_features=interface_features
    )
    
    # Build edges
    # #region agent log
    log_debug("pdb_to_graph.py:pdb_to_graph", "Before edge building", {
        "distances_shape": distances.shape,
        "num_chains": len(set(chain_ids)),
        "use_covalent_edges": use_covalent_edges,
        "use_noncovalent_edges": use_noncovalent_edges,
        "allow_duplicate_edges": allow_duplicate_edges,
        "noncovalent_cutoff": noncovalent_cutoff
    }, "C")
    # #endregion agent log

    edge_index, edge_attr, edge_type, edge_dist = build_edges(
        distances,
        chain_ids,
        residue_info,
        use_covalent_edges=use_covalent_edges,
        use_noncovalent_edges=use_noncovalent_edges,
        noncovalent_cutoff=noncovalent_cutoff,
        allow_duplicate_edges=allow_duplicate_edges
    )

    # #region agent log
    log_debug("pdb_to_graph.py:pdb_to_graph", "Edge building completed", {
        "edge_index_shape": edge_index.shape,
        "edge_attr_shape": edge_attr.shape,
        "edge_type_shape": edge_type.shape,
        "edge_dist_shape": edge_dist.shape,
        "num_edges": edge_index.shape[1],
        "edge_type_counts": {int(et): int((edge_type == et).sum()) for et in edge_type.unique().int().tolist()}
    }, "C")
    # #endregion agent log
    
    # Create Data object with all visualization metadata
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([y], dtype=torch.float32) if y is not None else None,
        num_nodes=len(coords),
        # Visualization metadata
        pos=torch.tensor(coords, dtype=torch.float32),  # [num_nodes, 3] Cα coordinates
        chain_id=torch.tensor(chain_ids_int, dtype=torch.int64),  # [num_nodes] chain index
        chain_role=torch.tensor(chain_roles, dtype=torch.int64),  # [num_nodes] 0=antibody, 1=antigen
        res_id=torch.tensor(residue_numbers, dtype=torch.int64),  # [num_nodes] residue number
        aa=torch.tensor(residue_types, dtype=torch.int64),  # [num_nodes] amino acid index 0..19
        edge_type=edge_type,  # [num_edges] 0=covalent, 1=noncovalent
        edge_dist=edge_dist,  # [num_edges] distance in Angstroms
        # Interface markers
        min_inter_dist=min_inter_dist,  # [num_nodes] minimum distance to opposite molecule
        inter_contact_count=inter_contact_count,  # [num_nodes] count of opposite-molecule contacts
        is_interface=is_interface,  # [num_nodes] binary interface marker
    )
    
    # Store chain label mapping (for in-memory use; also stored in metadata JSON)
    data.chain_labels = unique_chains
    
    # Runtime assertions/logging
    num_covalent = int((edge_type == 0).sum())
    num_noncovalent = int((edge_type == 1).sum())
    num_interface = int(is_interface.sum())
    
    if data.num_nodes == 0:
        raise ValueError(f"Graph has 0 nodes: {pdb_path}")
    if edge_index.shape[1] == 0:
        warnings.warn(f"Graph has 0 edges: {pdb_path}")
    if not chain_role_array.any() or (chain_role_array == 0).sum() == 0 or (chain_role_array == 1).sum() == 0:
        raise ValueError(f"Missing antibody or antigen nodes: {pdb_path} (antibody: {(chain_role_array == 0).sum()}, antigen: {(chain_role_array == 1).sum()})")
    
    # Log statistics (warnings for suspicious cases)
    if num_interface == 0:
        warnings.warn(f"No interface nodes found (interface_cutoff={interface_cutoff} Å): {pdb_path}")
    
    return data
