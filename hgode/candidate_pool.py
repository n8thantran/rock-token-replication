"""
Candidate pool construction for HGODE.
Constructs E_cand from: observed edges, 2-hop neighbors, random pairs.
"""
import torch
import numpy as np
from torch_geometric.utils import to_scipy_sparse_matrix, add_self_loops
from scipy.sparse import csr_matrix


def build_candidate_pool(edge_index, num_nodes, num_hops=2, num_random=5, include_observed=True):
    """
    Build candidate edge pool from:
    1. Observed edges
    2. Multi-hop neighbors (2-hop by default)
    3. Random pairs
    
    Args:
        edge_index: [2, E] tensor of observed edges
        num_nodes: number of nodes
        num_hops: number of hops for neighborhood expansion
        num_random: number of random neighbors per node
        include_observed: whether to include observed edges
    
    Returns:
        cand_edge_index: [2, E_cand] tensor of candidate edges
    """
    device = edge_index.device
    edge_set = set()
    
    # 1. Observed edges
    if include_observed:
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            edge_set.add((src, dst))
    
    # 2. Multi-hop neighbors
    if num_hops >= 2:
        # Build adjacency matrix
        adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
        adj_power = adj.copy()
        for _ in range(num_hops - 1):
            adj_power = adj_power @ adj
        # Get non-zero entries from multi-hop adjacency
        rows, cols = adj_power.nonzero()
        for r, c in zip(rows, cols):
            if r != c:  # no self-loops
                edge_set.add((int(r), int(c)))
    
    # 3. Random pairs
    if num_random > 0:
        for i in range(num_nodes):
            random_targets = np.random.choice(num_nodes, size=min(num_random, num_nodes - 1), replace=False)
            for j in random_targets:
                if j != i:
                    edge_set.add((i, int(j)))
    
    # Convert to tensor
    if len(edge_set) == 0:
        return edge_index
    
    edges = list(edge_set)
    src = torch.tensor([e[0] for e in edges], dtype=torch.long, device=device)
    dst = torch.tensor([e[1] for e in edges], dtype=torch.long, device=device)
    cand_edge_index = torch.stack([src, dst], dim=0)
    
    return cand_edge_index


def build_candidate_pool_simple(edge_index, num_nodes, num_random=5):
    """
    Simplified candidate pool: observed edges + random pairs.
    More memory efficient for large graphs.
    """
    device = edge_index.device
    
    # Start with observed edges
    edges_src = [edge_index[0]]
    edges_dst = [edge_index[1]]
    
    # Add random pairs
    if num_random > 0:
        rand_src = torch.randint(0, num_nodes, (num_nodes * num_random,), device=device)
        rand_dst = torch.randint(0, num_nodes, (num_nodes * num_random,), device=device)
        # Remove self-loops
        mask = rand_src != rand_dst
        edges_src.append(rand_src[mask])
        edges_dst.append(rand_dst[mask])
    
    cand_src = torch.cat(edges_src)
    cand_dst = torch.cat(edges_dst)
    cand_edge_index = torch.stack([cand_src, cand_dst], dim=0)
    
    # Remove duplicates
    cand_edge_index = torch.unique(cand_edge_index, dim=1)
    
    return cand_edge_index
