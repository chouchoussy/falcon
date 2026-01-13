"""
Graph Encoder module for FALCON.
Implements Gated Graph Neural Network (GGNN) encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch


class GraphEncoder(nn.Module):
    """
    Graph Encoder using Gated Graph Neural Network (GGNN).
    
    The encoder processes graph-structured execution traces and produces
    node-level embeddings using message passing.
    
    Architecture:
        - Input projection layer
        - Multiple GGNN layers (GatedGraphConv)
        - Layer normalization
        - Residual connections (optional)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = None,
        num_layers: int = 3,
        num_steps: int = 5,
        dropout: float = 0.1,
        use_residual: bool = True
    ):
        """
        Initialize the Graph Encoder.
        
        Args:
            input_dim: Dimension of input node features
            hidden_dim: Dimension of hidden representations
            output_dim: Dimension of output embeddings (default: same as hidden_dim)
            num_layers: Number of GGNN layers to stack
            num_steps: Number of recurrent steps in each GGNN layer
            dropout: Dropout rate
            use_residual: Whether to use residual connections
        """
        super(GraphEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.dropout = dropout
        self.use_residual = use_residual
        
        # Input projection layer
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GGNN layers
        self.ggnn_layers = nn.ModuleList()
        for i in range(num_layers):
            # GatedGraphConv: implements gated recurrent message passing
            self.ggnn_layers.append(
                GatedGraphConv(
                    out_channels=hidden_dim,
                    num_layers=num_steps
                )
            )
        
        # Layer normalization for each GGNN layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Output projection (if output_dim != hidden_dim)
        if self.output_dim != hidden_dim:
            self.output_proj = nn.Linear(hidden_dim, self.output_dim)
        else:
            self.output_proj = None
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass through the graph encoder.
        
        Args:
            x: Node feature matrix [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch vector [num_nodes] (for batched graphs)
        
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        # Input projection
        h = self.input_proj(x)
        h = F.relu(h)
        h = self.dropout_layer(h)
        
        # Apply GGNN layers with residual connections
        for i, (ggnn, norm) in enumerate(zip(self.ggnn_layers, self.layer_norms)):
            h_in = h
            
            # GGNN message passing
            h = ggnn(h, edge_index)
            
            # Layer normalization
            h = norm(h)
            
            # Residual connection
            if self.use_residual and i > 0:
                h = h + h_in
            
            # Activation and dropout
            h = F.relu(h)
            h = self.dropout_layer(h)
        
        # Output projection
        if self.output_proj is not None:
            h = self.output_proj(h)
        
        return h
    
    def encode_graph(self, data):
        """
        Convenience method to encode a single Data object.
        
        Args:
            data: PyTorch Geometric Data object
        
        Returns:
            Node embeddings
        """
        return self.forward(data.x, data.edge_index, data.batch if hasattr(data, 'batch') else None)
    
    def get_graph_embedding(self, x, edge_index, batch=None, pooling='mean'):
        """
        Get graph-level embedding by pooling node embeddings.
        
        Args:
            x: Node feature matrix
            edge_index: Graph connectivity
            batch: Batch vector
            pooling: Pooling method ('mean', 'max', or 'both')
        
        Returns:
            Graph-level embedding
        """
        node_emb = self.forward(x, edge_index, batch)
        
        if batch is None:
            # Single graph: pool all nodes
            if pooling == 'mean':
                return node_emb.mean(dim=0)
            elif pooling == 'max':
                return node_emb.max(dim=0)[0]
            elif pooling == 'both':
                return torch.cat([node_emb.mean(dim=0), node_emb.max(dim=0)[0]], dim=0)
        else:
            # Batched graphs: pool by batch
            if pooling == 'mean':
                return global_mean_pool(node_emb, batch)
            elif pooling == 'max':
                return global_max_pool(node_emb, batch)
            elif pooling == 'both':
                return torch.cat([
                    global_mean_pool(node_emb, batch),
                    global_max_pool(node_emb, batch)
                ], dim=1)


class MultiScaleGraphEncoder(GraphEncoder):
    """
    Enhanced Graph Encoder with multi-scale feature extraction.
    
    Captures information at different scales by combining embeddings
    from multiple GGNN layers.
    """
    
    def __init__(self, *args, **kwargs):
        super(MultiScaleGraphEncoder, self).__init__(*args, **kwargs)
        
        # Attention weights for multi-scale fusion
        self.scale_attention = nn.Parameter(torch.ones(self.num_layers))
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass with multi-scale feature extraction.
        
        Returns:
            Node embeddings combining all scales
        """
        # Input projection
        h = self.input_proj(x)
        h = F.relu(h)
        h = self.dropout_layer(h)
        
        # Collect embeddings from all layers
        layer_outputs = []
        
        for i, (ggnn, norm) in enumerate(zip(self.ggnn_layers, self.layer_norms)):
            h_in = h
            
            # GGNN message passing
            h = ggnn(h, edge_index)
            h = norm(h)
            
            if self.use_residual and i > 0:
                h = h + h_in
            
            h = F.relu(h)
            h = self.dropout_layer(h)
            
            # Store layer output
            layer_outputs.append(h)
        
        # Weighted combination of multi-scale features
        attention_weights = F.softmax(self.scale_attention, dim=0)
        h_combined = sum(w * h_layer for w, h_layer in zip(attention_weights, layer_outputs))
        
        # Output projection
        if self.output_proj is not None:
            h_combined = self.output_proj(h_combined)
        
        return h_combined


if __name__ == "__main__":
    # Test the encoder
    print("Testing Graph Encoder...")
    
    # Create a simple test graph
    num_nodes = 10
    input_dim = 384  # SentenceBERT embedding dim
    hidden_dim = 128
    
    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]
    ], dtype=torch.long)
    
    # Test standard encoder
    encoder = GraphEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=64,
        num_layers=3,
        num_steps=5
    )
    
    print(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    
    # Forward pass
    node_emb = encoder(x, edge_index)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {node_emb.shape}")
    
    # Test graph-level embedding
    graph_emb = encoder.get_graph_embedding(x, edge_index, pooling='mean')
    print(f"Graph embedding shape: {graph_emb.shape}")
    
    # Test multi-scale encoder
    print("\nTesting Multi-Scale Encoder...")
    multi_encoder = MultiScaleGraphEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=64,
        num_layers=3
    )
    
    multi_emb = multi_encoder(x, edge_index)
    print(f"Multi-scale output shape: {multi_emb.shape}")
    
    print("\n✓ All tests passed!")

