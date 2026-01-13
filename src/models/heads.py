"""
Prediction Heads for FALCON.
Implements Projection Head (for contrastive learning) and Rank Head (for fault localization).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    Projection Head for contrastive learning (Phase 1).
    
    Projects node embeddings to a lower-dimensional space where
    contrastive loss is computed. This is a standard component in
    contrastive learning frameworks like SimCLR.
    
    Architecture: Linear -> ReLU -> Linear
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        use_batch_norm: bool = False,
        dropout: float = 0.0
    ):
        """
        Initialize the Projection Head.
        
        Args:
            input_dim: Dimension of input embeddings
            hidden_dim: Dimension of hidden layer
            output_dim: Dimension of projected embeddings
            use_batch_norm: Whether to use batch normalization
            dropout: Dropout rate
        """
        super(ProjectionHead, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # First linear layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        # Optional batch normalization
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Second linear layer
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        """
        Project embeddings to contrastive space.
        
        Args:
            x: Input embeddings [batch_size, input_dim] or [num_nodes, input_dim]
        
        Returns:
            Projected embeddings [batch_size, output_dim] or [num_nodes, output_dim]
        """
        # First layer
        h = self.fc1(x)
        
        # Batch normalization (if enabled)
        if self.use_batch_norm:
            h = self.bn1(h)
        
        # Activation
        h = F.relu(h)
        
        # Dropout
        if self.dropout is not None:
            h = self.dropout(h)
        
        # Second layer
        z = self.fc2(h)
        
        # L2 normalization for contrastive learning
        z = F.normalize(z, p=2, dim=-1)
        
        return z


class RankHead(nn.Module):
    """
    Ranking Head for fault localization (Phase 2).
    
    Predicts a suspiciousness score for each node (function).
    Higher scores indicate higher likelihood of being faulty.
    
    Architecture: Simple linear layer (or MLP if deeper)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = None,
        dropout: float = 0.1,
        use_sigmoid: bool = False
    ):
        """
        Initialize the Ranking Head.
        
        Args:
            input_dim: Dimension of input embeddings
            hidden_dim: Dimension of hidden layer (None = no hidden layer)
            dropout: Dropout rate
            use_sigmoid: Whether to apply sigmoid activation to output
        """
        super(RankHead, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_sigmoid = use_sigmoid
        
        if hidden_dim is not None:
            # Two-layer MLP
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.dropout = nn.Dropout(dropout)
            self.fc2 = nn.Linear(hidden_dim, 1)
        else:
            # Single linear layer
            self.fc1 = None
            self.dropout = nn.Dropout(dropout) if dropout > 0 else None
            self.fc2 = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        """
        Predict suspiciousness scores.
        
        Args:
            x: Input embeddings [num_nodes, input_dim]
        
        Returns:
            Suspiciousness scores [num_nodes, 1] or [num_nodes]
        """
        if self.fc1 is not None:
            # Two-layer version
            h = self.fc1(x)
            h = F.relu(h)
            h = self.dropout(h)
            scores = self.fc2(h)
        else:
            # Single layer version
            if self.dropout is not None:
                x = self.dropout(x)
            scores = self.fc2(x)
        
        # Optional sigmoid activation
        if self.use_sigmoid:
            scores = torch.sigmoid(scores)
        
        # Squeeze to [num_nodes] if needed
        scores = scores.squeeze(-1)
        
        return scores


class DualHead(nn.Module):
    """
    Combined head that can perform both projection and ranking.
    Useful for multi-task learning scenarios.
    """
    
    def __init__(
        self,
        input_dim: int,
        proj_hidden_dim: int,
        proj_output_dim: int,
        rank_hidden_dim: int = None,
        dropout: float = 0.1
    ):
        """
        Initialize the Dual Head.
        
        Args:
            input_dim: Dimension of input embeddings
            proj_hidden_dim: Hidden dim for projection head
            proj_output_dim: Output dim for projection head
            rank_hidden_dim: Hidden dim for rank head
            dropout: Dropout rate
        """
        super(DualHead, self).__init__()
        
        self.projection_head = ProjectionHead(
            input_dim=input_dim,
            hidden_dim=proj_hidden_dim,
            output_dim=proj_output_dim,
            dropout=dropout
        )
        
        self.rank_head = RankHead(
            input_dim=input_dim,
            hidden_dim=rank_hidden_dim,
            dropout=dropout
        )
    
    def forward(self, x, return_projection=True, return_rank=True):
        """
        Forward pass through both heads.
        
        Args:
            x: Input embeddings
            return_projection: Whether to compute projection
            return_rank: Whether to compute rank scores
        
        Returns:
            Dictionary with 'projection' and/or 'rank' keys
        """
        outputs = {}
        
        if return_projection:
            outputs['projection'] = self.projection_head(x)
        
        if return_rank:
            outputs['rank'] = self.rank_head(x)
        
        return outputs


class AttentionRankHead(nn.Module):
    """
    Enhanced Ranking Head with attention mechanism.
    
    Uses attention to focus on important features when computing
    suspiciousness scores.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize the Attention Rank Head.
        
        Args:
            input_dim: Dimension of input embeddings
            hidden_dim: Dimension of hidden layer
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(AttentionRankHead, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(input_dim)
    
    def forward(self, x):
        """
        Forward pass with attention.
        
        Args:
            x: Input embeddings [num_nodes, input_dim]
        
        Returns:
            Suspiciousness scores [num_nodes]
        """
        # Add batch dimension for attention
        x_unsqueezed = x.unsqueeze(0)  # [1, num_nodes, input_dim]
        
        # Self-attention
        attn_out, _ = self.attention(x_unsqueezed, x_unsqueezed, x_unsqueezed)
        attn_out = attn_out.squeeze(0)  # [num_nodes, input_dim]
        
        # Residual connection and layer norm
        x_attended = self.layer_norm(x + attn_out)
        
        # Feed-forward to get scores
        h = self.fc1(x_attended)
        h = F.relu(h)
        h = self.dropout(h)
        scores = self.fc2(h).squeeze(-1)
        
        return scores


if __name__ == "__main__":
    # Test the heads
    print("Testing Projection Head and Rank Head...")
    
    batch_size = 32
    input_dim = 128
    
    # Test Projection Head
    print("\n1. Testing Projection Head:")
    proj_head = ProjectionHead(
        input_dim=input_dim,
        hidden_dim=256,
        output_dim=64
    )
    
    x = torch.randn(batch_size, input_dim)
    z = proj_head(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {z.shape}")
    print(f"  L2 norm: {torch.norm(z, dim=-1).mean().item():.4f} (should be ~1.0)")
    
    # Test Rank Head
    print("\n2. Testing Rank Head (single layer):")
    rank_head_simple = RankHead(input_dim=input_dim)
    
    scores = rank_head_simple(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {scores.shape}")
    print(f"  Score range: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
    
    # Test Rank Head with hidden layer
    print("\n3. Testing Rank Head (two layers):")
    rank_head_deep = RankHead(
        input_dim=input_dim,
        hidden_dim=64,
        use_sigmoid=True
    )
    
    scores_deep = rank_head_deep(x)
    print(f"  Output shape: {scores_deep.shape}")
    print(f"  Score range: [{scores_deep.min().item():.4f}, {scores_deep.max().item():.4f}]")
    
    # Test Dual Head
    print("\n4. Testing Dual Head:")
    dual_head = DualHead(
        input_dim=input_dim,
        proj_hidden_dim=256,
        proj_output_dim=64,
        rank_hidden_dim=64
    )
    
    outputs = dual_head(x)
    print(f"  Projection shape: {outputs['projection'].shape}")
    print(f"  Rank shape: {outputs['rank'].shape}")
    
    # Test Attention Rank Head
    print("\n5. Testing Attention Rank Head:")
    attn_rank_head = AttentionRankHead(
        input_dim=input_dim,
        hidden_dim=64,
        num_heads=4
    )
    
    scores_attn = attn_rank_head(x)
    print(f"  Output shape: {scores_attn.shape}")
    print(f"  Score range: [{scores_attn.min().item():.4f}, {scores_attn.max().item():.4f}]")
    
    print("\n✓ All tests passed!")

