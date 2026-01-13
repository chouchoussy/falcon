"""
Models module for FALCON.
Contains the complete FALCON model architecture for fault localization.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .encoder import GraphEncoder, MultiScaleGraphEncoder
from .heads import ProjectionHead, RankHead, DualHead, AttentionRankHead


class FALCONModel(nn.Module):
    """
    Complete FALCON model for fault localization.
    
    The model consists of three main components:
    1. GraphEncoder: Encodes graph structure using GGNN
    2. ProjectionHead: Projects embeddings for contrastive learning (Phase 1)
    3. RankHead: Predicts suspiciousness scores (Phase 2)
    
    Training happens in two phases:
    - Phase 1: Contrastive learning with projection head
    - Phase 2: Supervised fault localization with rank head
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int = 64,
        proj_hidden_dim: int = 256,
        proj_output_dim: int = 64,
        rank_hidden_dim: int = None,
        num_gnn_layers: int = 3,
        num_gnn_steps: int = 5,
        dropout: float = 0.1,
        use_multi_scale: bool = False
    ):
        """
        Initialize FALCON model.
        
        Args:
            input_dim: Dimension of input node features (e.g., 384 for SentenceBERT)
            hidden_dim: Dimension of GNN hidden layers
            embedding_dim: Dimension of node embeddings from encoder
            proj_hidden_dim: Hidden dimension for projection head
            proj_output_dim: Output dimension for projection head
            rank_hidden_dim: Hidden dimension for rank head (None = single layer)
            num_gnn_layers: Number of GGNN layers
            num_gnn_steps: Number of recurrent steps per GGNN layer
            dropout: Dropout rate
            use_multi_scale: Whether to use multi-scale encoder
        """
        super(FALCONModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        # Graph Encoder (GGNN)
        encoder_class = MultiScaleGraphEncoder if use_multi_scale else GraphEncoder
        self.encoder = encoder_class(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_layers=num_gnn_layers,
            num_steps=num_gnn_steps,
            dropout=dropout,
            use_residual=True
        )
        
        # Projection Head (for Phase 1: Contrastive Learning)
        self.projection_head = ProjectionHead(
            input_dim=embedding_dim,
            hidden_dim=proj_hidden_dim,
            output_dim=proj_output_dim,
            dropout=dropout
        )
        
        # Rank Head (for Phase 2: Fault Localization)
        self.rank_head = RankHead(
            input_dim=embedding_dim,
            hidden_dim=rank_hidden_dim,
            dropout=dropout,
            use_sigmoid=False  # Use raw scores for ranking
        )
    
    def forward(
        self,
        x,
        edge_index,
        batch=None,
        return_projection: bool = True,
        return_rank: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through FALCON model.
        
        Args:
            x: Node feature matrix [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch vector [num_nodes] for batched graphs
            return_projection: Whether to compute projection (Phase 1)
            return_rank: Whether to compute rank scores (Phase 2)
        
        Returns:
            Dictionary containing:
                - 'node_emb': Node embeddings [num_nodes, embedding_dim]
                - 'proj_emb': Projected embeddings [num_nodes, proj_output_dim] (if return_projection)
                - 'rank_score': Suspiciousness scores [num_nodes] (if return_rank)
        """
        # Encode graph structure
        node_emb = self.encoder(x, edge_index, batch)
        
        outputs = {'node_emb': node_emb}
        
        # Projection for contrastive learning (Phase 1)
        if return_projection:
            proj_emb = self.projection_head(node_emb)
            outputs['proj_emb'] = proj_emb
        
        # Ranking for fault localization (Phase 2)
        if return_rank:
            rank_score = self.rank_head(node_emb)
            outputs['rank_score'] = rank_score
        
        return outputs
    
    def encode(self, x, edge_index, batch=None):
        """
        Encode graph to get node embeddings only.
        
        Args:
            x: Node features
            edge_index: Graph connectivity
            batch: Batch vector
        
        Returns:
            Node embeddings
        """
        return self.encoder(x, edge_index, batch)
    
    def predict_rank(self, x, edge_index, batch=None):
        """
        Predict suspiciousness scores for fault localization.
        
        Args:
            x: Node features
            edge_index: Graph connectivity
            batch: Batch vector
        
        Returns:
            Suspiciousness scores [num_nodes]
        """
        node_emb = self.encoder(x, edge_index, batch)
        rank_score = self.rank_head(node_emb)
        return rank_score
    
    def get_projection(self, x, edge_index, batch=None):
        """
        Get projected embeddings for contrastive learning.
        
        Args:
            x: Node features
            edge_index: Graph connectivity
            batch: Batch vector
        
        Returns:
            Projected embeddings [num_nodes, proj_output_dim]
        """
        node_emb = self.encoder(x, edge_index, batch)
        proj_emb = self.projection_head(node_emb)
        return proj_emb
    
    def freeze_encoder(self):
        """Freeze encoder parameters (useful for Phase 2)."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def freeze_projection_head(self):
        """Freeze projection head parameters."""
        for param in self.projection_head.parameters():
            param.requires_grad = False
    
    def get_num_params(self):
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FALCONModelV2(FALCONModel):
    """
    Enhanced version of FALCON with attention-based ranking.
    """
    
    def __init__(self, *args, use_attention_rank: bool = True, **kwargs):
        super(FALCONModelV2, self).__init__(*args, **kwargs)
        
        # Replace rank head with attention-based version
        if use_attention_rank:
            rank_hidden_dim = kwargs.get('rank_hidden_dim', 64)
            self.rank_head = AttentionRankHead(
                input_dim=self.embedding_dim,
                hidden_dim=rank_hidden_dim if rank_hidden_dim else 64,
                num_heads=4,
                dropout=kwargs.get('dropout', 0.1)
            )


def build_falcon_model(
    input_dim: int = 384,
    hidden_dim: int = 128,
    embedding_dim: int = 64,
    model_version: str = 'v1',
    **kwargs
) -> FALCONModel:
    """
    Factory function to build FALCON model with default configurations.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension
        embedding_dim: Embedding dimension
        model_version: 'v1' or 'v2'
        **kwargs: Additional model parameters
    
    Returns:
        FALCON model instance
    """
    if model_version == 'v2':
        return FALCONModelV2(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            **kwargs
        )
    else:
        return FALCONModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            **kwargs
        )


# Export all components
__all__ = [
    # Encoder
    'GraphEncoder',
    'MultiScaleGraphEncoder',
    
    # Heads
    'ProjectionHead',
    'RankHead',
    'DualHead',
    'AttentionRankHead',
    
    # Complete Model
    'FALCONModel',
    'FALCONModelV2',
    'build_falcon_model',
]


if __name__ == "__main__":
    # Test the complete model
    print("Testing FALCON Model...")
    
    # Create model
    model = FALCONModel(
        input_dim=384,
        hidden_dim=128,
        embedding_dim=64,
        proj_hidden_dim=256,
        proj_output_dim=64,
        rank_hidden_dim=64,
        num_gnn_layers=3,
        num_gnn_steps=5
    )
    
    print(f"Total parameters: {model.get_num_params():,}")
    
    # Create test data
    num_nodes = 50
    num_edges = 100
    
    x = torch.randn(num_nodes, 384)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    print(f"\nTest input:")
    print(f"  Nodes: {num_nodes}")
    print(f"  Edges: {num_edges}")
    print(f"  Features: {x.shape}")
    
    # Test Phase 1 (Contrastive Learning)
    print("\n--- Phase 1: Contrastive Learning ---")
    outputs_phase1 = model(x, edge_index, return_projection=True, return_rank=False)
    print(f"  Node embeddings: {outputs_phase1['node_emb'].shape}")
    print(f"  Projected embeddings: {outputs_phase1['proj_emb'].shape}")
    
    # Test Phase 2 (Fault Localization)
    print("\n--- Phase 2: Fault Localization ---")
    outputs_phase2 = model(x, edge_index, return_projection=False, return_rank=True)
    print(f"  Node embeddings: {outputs_phase2['node_emb'].shape}")
    print(f"  Rank scores: {outputs_phase2['rank_score'].shape}")
    print(f"  Score range: [{outputs_phase2['rank_score'].min().item():.4f}, {outputs_phase2['rank_score'].max().item():.4f}]")
    
    # Test convenience methods
    print("\n--- Testing Convenience Methods ---")
    rank_scores = model.predict_rank(x, edge_index)
    print(f"  predict_rank output: {rank_scores.shape}")
    
    proj_emb = model.get_projection(x, edge_index)
    print(f"  get_projection output: {proj_emb.shape}")
    
    # Test FALCONModelV2
    print("\n--- Testing FALCON V2 (with Attention) ---")
    model_v2 = FALCONModelV2(
        input_dim=384,
        hidden_dim=128,
        embedding_dim=64,
        use_attention_rank=True
    )
    
    outputs_v2 = model_v2(x, edge_index, return_rank=True)
    print(f"  Rank scores: {outputs_v2['rank_score'].shape}")
    
    # Test factory function
    print("\n--- Testing Factory Function ---")
    model_factory = build_falcon_model(model_version='v1')
    print(f"  Built model with {model_factory.get_num_params():,} parameters")
    
    print("\n✓ All tests passed!")

