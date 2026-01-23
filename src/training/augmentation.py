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
    
    # Convert edge_index to NetworkX graph tìm đường đi
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
    edge_centrality = node_degree[col]
    
    return edge_centrality


def augment_graph(data: Data, p_tau: float = 0.5, use_transitive: bool = True) -> Data:
    """
    Apply Adaptive Graph Augmentation (AGA) strictly following the paper.
    
    Equation 1: p_uv = min( (w_max - w_uv)/(w_max - w_avg), p_tau )
    """
    # Clone data
    augmented_data = data.clone()
    edge_index = augmented_data.edge_index
    num_edges = edge_index.shape[1]
    
    if num_edges == 0:
        return augmented_data
    
    # 1. Identify faulty nodes
    if hasattr(augmented_data, 'y') and augmented_data.y is not None:
        faulty_nodes = torch.where(augmented_data.y == 1)[0].tolist()
    else:
        return augmented_data # No labels, cannot augment
    
    # 2. Identify fault-related nodes (Vr)
    if use_transitive and len(faulty_nodes) > 0:
        fault_related_nodes = identify_fault_related_subgraph(augmented_data, faulty_nodes)
    else:
        fault_related_nodes = set(faulty_nodes)
    
    # 3. Calculate Edge Centrality (w_uv)
    edge_centrality = calculate_edge_centrality(augmented_data)
    
    # 4. Calculate Removal Probabilities (p_uv) according to Eq 1
    w_max = edge_centrality.max()
    w_avg = edge_centrality.mean()
    
    # Tránh chia cho 0 nếu w_max = w_avg
    if w_max - w_avg == 0:
        # Nếu tất cả cạnh có centrality bằng nhau, xác suất xóa bằng 0
        p_uv_all = torch.zeros_like(edge_centrality)
    else:
        # Phần trong ngoặc: (w_max - w_uv) / (w_max - w_avg)
        term = (w_max - edge_centrality) / (w_max - w_avg)
        
        # Áp dụng hàm min(..., p_tau)
        p_uv_all = torch.clamp(term, max=p_tau)
        
        # Cần đảm bảo p_uv không âm (xác suất >= 0)
        p_uv_all = torch.clamp(p_uv_all, min=0.0)

    # 5. Drop Edges
    edges_to_keep = []
    
    # Chuyển p_uv sang numpy hoặc list để truy xuất nhanh trong loop
    p_uv_list = p_uv_all.tolist()
    
    for i in range(num_edges):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        
        # Check Fault-related
        is_fault_related = (src in fault_related_nodes) or (dst in fault_related_nodes)
        
        if is_fault_related:
            # Luôn giữ lại cạnh liên quan lỗi
            edges_to_keep.append(i)
        else:
            # Fault-unrelated: Xóa dựa trên xác suất p_uv
            prob_drop = p_uv_list[i]
            
            # Sinh số ngẫu nhiên r [0, 1]
            # Nếu r < prob_drop thì XÓA. 
            # (Hoặc r >= prob_drop thì GIỮ)
            r = np.random.random()
            
            if r >= prob_drop: 
                edges_to_keep.append(i)
    
    # Update graph
    if len(edges_to_keep) > 0:
        augmented_data.edge_index = edge_index[:, edges_to_keep]
    else:
        augmented_data.edge_index = edge_index # Fallback
    
    return augmented_data


def augment_graph_pair(data: Data, p_tau: float = 0.2) -> Tuple[Data, Data]:
    """
    Create two augmented views of the same graph for contrastive learning.
    
    Args:
        data: PyTorch Geometric Data object
        p_tau: Base dropping probability

    Returns:
        Tuple of two augmented Data objects (G'_f1, G'_f2)
    """
    view1 = augment_graph(data, p_tau=p_tau, use_transitive=True)
    view2 = augment_graph(data, p_tau=p_tau, use_transitive=True)
    
    return view1, view2


def batch_augment_graphs(data_list: list, p_tau: float = 0.2) -> list:
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
        augmented = augment_graph(data, p_tau=p_tau)
        augmented_list.append(augmented)
    
    return augmented_list

