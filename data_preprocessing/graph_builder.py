"""
Graph Builder module for FALCON.
Constructs PyTorch Geometric Data objects from parsed log files.
"""

import torch
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer
import hashlib

from .log_parser import LogEvent, parse_log_file


class GraphBuilder:
    """
    Builds graph representations from execution traces.
    
    Graph structure:
    - Nodes: Log entries, Packages, Files, Methods
    - Edges: Hierarchical (Log->Package->File->Method) + Sequential (Method->Method)
    - Features: SentenceBERT embeddings of node names
    """
    
    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2', cache_dir: str = './processed_data'):
        """
        Initialize the GraphBuilder.
        
        Args:
            embedding_model_name: Name of SentenceTransformer model to use
            cache_dir: Directory to cache embeddings
        """
        self.encoder = SentenceTransformer(embedding_model_name)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for embeddings (in-memory)
        self.embedding_cache = {}
        
        # Load persistent cache if exists
        self.cache_file = self.cache_dir / 'embedding_cache.pkl'
        self._load_cache()
    
    def _load_cache(self):
        """Load embedding cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                print(f"Loaded {len(self.embedding_cache)} cached embeddings")
            except Exception as e:
                print(f"Warning: Could not load cache: {e}")
                self.embedding_cache = {}
    
    def _save_cache(self):
        """Save embedding cache to disk."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")
    
    def _get_embedding(self, text: str) -> torch.Tensor:
        """
        Get embedding for text, using cache if available.
        
        Args:
            text: Input text
        
        Returns:
            Embedding tensor
        """
        # Create hash for cache key
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash in self.embedding_cache:
            return torch.tensor(self.embedding_cache[text_hash])
        
        # Compute embedding
        embedding = self.encoder.encode(text, convert_to_tensor=False)
        
        # Cache it
        self.embedding_cache[text_hash] = embedding
        
        return torch.tensor(embedding)
    
    def _extract_package_name(self, filename: str) -> str:
        """
        Extract package name from filename.
        
        Args:
            filename: File name (e.g., 'tcpdump.c', 'print-ip.c')
        
        Returns:
            Package name (e.g., 'tcpdump', 'print-ip')
        """
        # Remove extension and use as package
        return Path(filename).stem
    
    def build_graph_from_events(
        self, 
        events: List[LogEvent],
        ground_truth: Optional[Dict] = None
    ) -> Data:
        """
        Build a PyTorch Geometric Data object from log events.
        
        Args:
            events: List of LogEvent objects
            ground_truth: Dictionary mapping function IDs to fault labels
        
        Returns:
            PyTorch Geometric Data object
        """
        if len(events) == 0:
            # Return empty graph
            return Data(x=torch.zeros(0, 384), edge_index=torch.zeros(2, 0, dtype=torch.long))
        
        # Node types: 0=Log, 1=Package, 2=File, 3=Method
        node_map = {}  # Maps node_id -> node_index
        node_features = []
        node_types = []
        node_names = []
        
        edges = []
        
        def get_or_create_node(node_id: str, node_type: int, display_name: str) -> int:
            """Helper to get or create a node."""
            if node_id not in node_map:
                idx = len(node_map)
                node_map[node_id] = idx
                
                # Get embedding
                embedding = self._get_embedding(display_name)
                node_features.append(embedding)
                node_types.append(node_type)
                node_names.append(display_name)
                
                return idx
            return node_map[node_id]
        
        # Build nodes and hierarchical edges
        previous_method_by_thread = {}  # Track last method per thread for sequential edges
        
        for log_idx, event in enumerate(events):
            # Create nodes for: Log -> Package -> File -> Method
            log_id = f"log_{log_idx}"
            package_id = f"pkg_{self._extract_package_name(event.filename)}"
            file_id = f"file_{event.filename}"
            method_id = f"method_{event.unique_id}"
            
            # Create nodes
            log_node = get_or_create_node(log_id, 0, f"Log {log_idx}")
            pkg_node = get_or_create_node(package_id, 1, self._extract_package_name(event.filename))
            file_node = get_or_create_node(file_id, 2, event.filename)
            method_node = get_or_create_node(method_id, 3, event.funcname)
            
            # Hierarchical edges: Log -> Package -> File -> Method
            edges.append([log_node, pkg_node])
            edges.append([pkg_node, file_node])
            edges.append([file_node, method_node])
            
            # Sequential edges: Method_i -> Method_{i+1} (ONLY within same thread)
            # CRITICAL: As per FALCON paper, we connect method nodes according to
            # "the order of their execution within threads" (not across threads)
            # This prevents creating fake edges between unrelated thread contexts
            thread_id = event.thread_id
            if thread_id in previous_method_by_thread:
                prev_method = previous_method_by_thread[thread_id]
                edges.append([prev_method, method_node])
            
            previous_method_by_thread[thread_id] = method_node
        
        # Convert to tensors
        x = torch.stack(node_features) if node_features else torch.zeros(0, 384)
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
        
        # Create labels from ground truth
        y = torch.zeros(len(node_map), dtype=torch.long)
        
        if ground_truth:
            for node_id, node_idx in node_map.items():
                # Check if this is a method node and if it's in ground truth
                if node_id.startswith("method_"):
                    func_id = node_id.replace("method_", "")
                    
                    # Check various possible formats in ground truth
                    if func_id in ground_truth:
                        if ground_truth[func_id]:
                            y[node_idx] = 1
                    
                    # Also check just the function name
                    func_name = func_id.split("::")[-1] if "::" in func_id else func_id
                    if func_name in ground_truth:
                        if ground_truth[func_name]:
                            y[node_idx] = 1
        
        # Create Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            y=y
        )
        
        # Store metadata
        data.node_types = torch.tensor(node_types, dtype=torch.long)
        data.node_names = node_names
        data.num_nodes = len(node_map)
        
        return data
    
    def build_graph_from_log_file(
        self,
        log_filepath: str,
        ground_truth: Optional[Dict] = None
    ) -> Data:
        """
        Build graph directly from a log file.
        
        Args:
            log_filepath: Path to log file
            ground_truth: Dictionary mapping function IDs to fault labels
        
        Returns:
            PyTorch Geometric Data object
        """
        events = parse_log_file(log_filepath)
        return self.build_graph_from_events(events, ground_truth)
    
    def build_graphs_from_directory(
        self,
        data_dir: str,
        ground_truth_file: Optional[str] = None,
        save_processed: bool = True
    ) -> Dict[str, Data]:
        """
        Build graphs for all log files in a directory.
        
        Args:
            data_dir: Directory containing log files
            ground_truth_file: Path to ground truth JSON file
            save_processed: Whether to save processed graphs to disk
        
        Returns:
            Dictionary mapping test_id -> Data object
        """
        data_dir = Path(data_dir)
        
        # Load ground truth if provided
        ground_truth = None
        if ground_truth_file and Path(ground_truth_file).exists():
            with open(ground_truth_file, 'r') as f:
                ground_truth = json.load(f)
            print(f"Loaded ground truth with {len(ground_truth)} entries")
        
        # Find all log files
        log_files = list(data_dir.glob("**/*.log"))
        print(f"Found {len(log_files)} log files")
        
        graphs = {}
        
        for log_file in log_files:
            test_id = log_file.stem
            
            # Check if already processed
            cache_path = self.cache_dir / f"{test_id}.pt"
            
            if cache_path.exists() and save_processed:
                # Load from cache
                try:
                    # PyTorch 2.6+ requires weights_only=False for custom objects
                    # Always use map_location='cpu' for preprocessing (portable files)
                    data = torch.load(cache_path, map_location='cpu', weights_only=False)
                    graphs[test_id] = data
                    continue
                except Exception as e:
                    print(f"Warning: Could not load cached graph for {test_id}: {e}")
            
            # Build graph
            try:
                data = self.build_graph_from_log_file(str(log_file), ground_truth)
                graphs[test_id] = data
                
                # Save to cache
                if save_processed:
                    torch.save(data, cache_path)
            
            except Exception as e:
                print(f"Error building graph for {test_id}: {e}")
        
        # Save embedding cache
        self._save_cache()
        
        print(f"Built {len(graphs)} graphs")
        return graphs


def load_ground_truth(ground_truth_path: str) -> Dict[str, bool]:
    """
    Load ground truth labels from JSON file.
    
    Args:
        ground_truth_path: Path to ground truth JSON file
    
    Returns:
        Dictionary mapping function IDs to fault labels (True/False)
    """
    with open(ground_truth_path, 'r') as f:
        data = json.load(f)
    
    # Convert to boolean if needed
    ground_truth = {}
    for key, value in data.items():
        if isinstance(value, bool):
            ground_truth[key] = value
        elif isinstance(value, int):
            ground_truth[key] = value == 1
        else:
            ground_truth[key] = bool(value)
    
    return ground_truth


if __name__ == "__main__":
    # Test the graph builder
    import sys
    from ..config import RAW_DATA_PATH, PROCESSED_PATH, GROUND_TRUTH_FILE
    
    print("Testing Graph Builder...")
    
    builder = GraphBuilder(cache_dir=PROCESSED_PATH)
    
    if len(sys.argv) > 1:
        # Test with specific log file
        log_file = sys.argv[1]
        print(f"\nBuilding graph from: {log_file}")
        
        data = builder.build_graph_from_log_file(log_file)
        print(f"Graph: {data.num_nodes} nodes, {data.num_edges} edges")
        print(f"Node features shape: {data.x.shape}")
        print(f"Labels: {data.y.sum().item()} faulty nodes")
    else:
        # Test with directory
        print(f"\nBuilding graphs from directory: {RAW_DATA_PATH}")
        
        gt_path = Path(RAW_DATA_PATH) / GROUND_TRUTH_FILE
        graphs = builder.build_graphs_from_directory(
            RAW_DATA_PATH,
            ground_truth_file=str(gt_path) if gt_path.exists() else None
        )
        
        if graphs:
            sample_id = list(graphs.keys())[0]
            sample_data = graphs[sample_id]
            print(f"\nSample graph ({sample_id}):")
            print(f"  Nodes: {sample_data.num_nodes}")
            print(f"  Edges: {sample_data.num_edges}")
            print(f"  Features: {sample_data.x.shape}")
            print(f"  Faulty nodes: {sample_data.y.sum().item()}")

