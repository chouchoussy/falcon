"""
Dataset module for FALCON.
Handles log parsing, graph construction, and adaptive augmentation.
"""

from .log_parser import (
    LogEvent,
    parse_log_file,
    parse_log_directory,
    get_unique_functions,
    get_call_sequence
)

from .graph_builder import (
    GraphBuilder,
    load_ground_truth
)

from .augmentation import (
    identify_fault_related_subgraph,
    calculate_edge_centrality,
    augment_graph,
    augment_graph_pair,
    batch_augment_graphs
)

__all__ = [
    # Log Parser
    'LogEvent',
    'parse_log_file',
    'parse_log_directory',
    'get_unique_functions',
    'get_call_sequence',
    
    # Graph Builder
    'GraphBuilder',
    'load_ground_truth',
    
    # Augmentation
    'identify_fault_related_subgraph',
    'calculate_edge_centrality',
    'augment_graph',
    'augment_graph_pair',
    'batch_augment_graphs',
]

