"""
Adaptive Graph Augmentation (AGA) module for FALCON.
Implements Transitive Analysis-based Adaptive Graph Augmentation (Section III.B.2).
"""

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, degree
from typing import Set, Tuple
import networkx as nx


def identify_fault_related_subgraph(data: Data, faulty_node_indices: list) -> Set[int]:
    """
    Identify the fault-related subgraph using transitive closure.
    
    Finds all nodes that have a path to or from the faulty nodes.
    
    Args:
        data: PyTorch Geometric Data object
        faulty_node_indices: List of indices of faulty nodes (y=1)
    
    Returns:
        Set of node indices in the fault-related subgraph
    """
    if len(faulty_node_indices) == 0:
        return set()
    
    # Convert edge_index to NetworkX graph for easier path analysis
    edge_index = data.edge_index.cpu().numpy()
    num_nodes = data.num_nodes
    
    # Create directed graph
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    
    # Add edges
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        G.add_edge(int(src), int(dst))
    
    fault_related = set(faulty_node_indices)
    
    # For each faulty node, find all reachable nodes (forward)
    # and all nodes that can reach it (backward)
    for faulty_idx in faulty_node_indices:
        # Forward reachable (descendants)
        try:
            descendants = nx.descendants(G, faulty_idx)
            fault_related.update(descendants)
        except:
            pass
        
        # Backward reachable (ancestors)
        try:
            ancestors = nx.ancestors(G, faulty_idx)
            fault_related.update(ancestors)
        except:
            pass
        
        # Add the faulty node itself
        fault_related.add(faulty_idx)
    
    return fault_related


def calculate_edge_centrality(data: Data) -> torch.Tensor:
    """
    Calculate edge centrality based on node degrees.
    
    Edge centrality is computed as the sum of degrees of the two endpoint nodes.
    Higher centrality means the edge connects more important nodes.
    
    Args:
        data: PyTorch Geometric Data object
    
    Returns:
        Tensor of edge centrality scores (one per edge)
    """
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    
    # Calculate node degrees (both in and out)
    row, col = edge_index
    node_degree = degree(row, num_nodes) + degree(col, num_nodes)
    
    # Edge centrality = degree(src) + degree(dst)
    edge_centrality = node_degree[row] + node_degree[col]
    
    return edge_centrality


def augment_graph(data: Data, drop_prob: float = 0.2, use_transitive: bool = True) -> Data:
    """
    Apply Adaptive Graph Augmentation (AGA) to create G'_f.
    
    According to Equation 1 in the paper:
    p_drop(e) = drop_prob * (1 - centrality(e) / max_centrality) if e is fault-unrelated
    
    Args:
        data: PyTorch Geometric Data object
        drop_prob: Base dropping probability (alpha in the paper)
        use_transitive: Whether to use transitive analysis to identify fault-related edges
    
    Returns:
        Augmented Data object
    """
    # Clone the data to avoid modifying the original
    augmented_data = data.clone()
    
    edge_index = augmented_data.edge_index
    num_edges = edge_index.shape[1]
    
    if num_edges == 0:
        return augmented_data
    
    # Identify faulty nodes (y=1)
    if hasattr(augmented_data, 'y') and augmented_data.y is not None:
        faulty_nodes = torch.where(augmented_data.y == 1)[0].tolist()
    else:
        # If no labels, don't drop any edges
        return augmented_data
    
    # Identify fault-related subgraph
    if use_transitive and len(faulty_nodes) > 0:
        fault_related_nodes = identify_fault_related_subgraph(augmented_data, faulty_nodes)
    else:
        fault_related_nodes = set(faulty_nodes)
    
    # Calculate edge centrality
    edge_centrality = calculate_edge_centrality(augmented_data)
    
    # Normalize centrality to [0, 1]
    max_centrality = edge_centrality.max()
    if max_centrality > 0:
        normalized_centrality = edge_centrality / max_centrality
    else:
        normalized_centrality = torch.zeros_like(edge_centrality)
    
    # Determine which edges to keep
    edges_to_keep = []
    
    for i in range(num_edges):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        
        # Check if edge is fault-related
        is_fault_related = (src in fault_related_nodes) or (dst in fault_related_nodes)
        
        if is_fault_related:
            # Always keep fault-related edges
            edges_to_keep.append(i)
        else:
            # Apply adaptive dropping for fault-unrelated edges
            # p_drop = drop_prob * (1 - centrality)
            centrality = normalized_centrality[i].item()
            p_drop = drop_prob * (1 - centrality)
            
            # Keep edge with probability (1 - p_drop)
            if np.random.random() > p_drop:
                edges_to_keep.append(i)
    
    # Update edge_index
    if len(edges_to_keep) > 0:
        augmented_data.edge_index = edge_index[:, edges_to_keep]
    else:
        # Keep at least some edges to avoid empty graph
        augmented_data.edge_index = edge_index
    
    return augmented_data


def augment_graph_pair(data: Data, drop_prob: float = 0.2) -> Tuple[Data, Data]:
    """
    Create two augmented views of the same graph for contrastive learning.
    
    Args:
        data: PyTorch Geometric Data object
        drop_prob: Base dropping probability
    
    Returns:
        Tuple of two augmented Data objects (G'_f1, G'_f2)
    """
    view1 = augment_graph(data, drop_prob=drop_prob, use_transitive=True)
    view2 = augment_graph(data, drop_prob=drop_prob, use_transitive=True)
    
    return view1, view2


def batch_augment_graphs(data_list: list, drop_prob: float = 0.2) -> list:
    """
    Apply augmentation to a batch of graphs.
    
    Args:
        data_list: List of PyTorch Geometric Data objects
        drop_prob: Base dropping probability
    
    Returns:
        List of augmented Data objects
    """
    augmented_list = []
    
    for data in data_list:
        augmented = augment_graph(data, drop_prob=drop_prob)
        augmented_list.append(augmented)
    
    return augmented_list


if __name__ == "__main__":
    # Test the augmentation module
    print("Testing Adaptive Graph Augmentation...")
    
    # Create a simple test graph
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4],
        [1, 0, 2, 1, 3, 2, 4, 3]
    ], dtype=torch.long)
    
    x = torch.randn(5, 16)  # 5 nodes, 16 features
    y = torch.tensor([0, 0, 1, 0, 0], dtype=torch.long)  # Node 2 is faulty
    
    data = Data(x=x, edge_index=edge_index, y=y)
    
    print(f"Original graph: {data.num_nodes} nodes, {data.num_edges} edges")
    
    # Test fault-related subgraph identification
    faulty_nodes = [2]
    fault_related = identify_fault_related_subgraph(data, faulty_nodes)
    print(f"Fault-related nodes: {fault_related}")
    
    # Test edge centrality
    centrality = calculate_edge_centrality(data)
    print(f"Edge centrality: {centrality}")
    
    # Test augmentation
    augmented = augment_graph(data, drop_prob=0.3)
    print(f"Augmented graph: {augmented.num_nodes} nodes, {augmented.num_edges} edges")
    
    # Test pair augmentation
    view1, view2 = augment_graph_pair(data, drop_prob=0.3)
    print(f"View 1: {view1.num_edges} edges, View 2: {view2.num_edges} edges")

