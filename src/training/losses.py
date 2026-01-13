"""
Loss functions for FALCON training.
Implements the three losses from the paper (Equations 3, 4, 5).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
from typing import Optional


class NodeContrastiveLoss(nn.Module):
    """
    Node-level Contrastive Loss (Equation 3 in paper).
    
    InfoNCE loss between nodes of original graph (G_f) and augmented graph (G'_f).
    Maximizes agreement between corresponding nodes in different views.
    
    Formula:
        L_node = -log( exp(sim(z_i, z'_i) / τ) / Σ_k exp(sim(z_i, z'_k) / τ) )
    
    where z_i and z'_i are projected embeddings of node i in two views.
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Initialize Node Contrastive Loss.
        
        Args:
            temperature: Temperature parameter (τ) for softmax scaling
        """
        super(NodeContrastiveLoss, self).__init__()
        self.temperature = temperature
    
    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute node-level contrastive loss.
        
        Args:
            z1: Projected embeddings from view 1 [num_nodes, embed_dim]
            z2: Projected embeddings from view 2 [num_nodes, embed_dim]
            mask: Optional mask for valid nodes [num_nodes] (True = valid)
        
        Returns:
            Scalar loss value
        """
        # Ensure embeddings are L2-normalized
        z1 = F.normalize(z1, p=2, dim=-1)
        z2 = F.normalize(z2, p=2, dim=-1)
        
        num_nodes = z1.shape[0]
        
        # Apply mask if provided
        if mask is not None:
            z1 = z1[mask]
            z2 = z2[mask]
            num_nodes = z1.shape[0]
        
        if num_nodes == 0:
            return torch.tensor(0.0, device=z1.device)
        
        # Compute similarity matrix: [num_nodes, num_nodes]
        # sim[i,j] = cosine_similarity(z1[i], z2[j])
        similarity_matrix = torch.matmul(z1, z2.t()) / self.temperature
        
        # For each node i in z1, the positive is z2[i]
        # Diagonal elements are positives
        positives = torch.diag(similarity_matrix)
        
        # For each row, we have one positive (diagonal) and (num_nodes-1) negatives
        # InfoNCE: -log(exp(pos) / sum(exp(all)))
        
        # Log-sum-exp trick for numerical stability
        loss = -positives + torch.logsumexp(similarity_matrix, dim=1)
        
        return loss.mean()


class GraphContrastiveLoss(nn.Module):
    """
    Graph-level Contrastive Loss (Equation 4 in paper).
    
    Triplet Margin Loss for graph-level embeddings.
    - Anchor: Augmented fail graph (G'_f)
    - Positive: Original fail graph (G_f)
    - Negative: Pass graph (G_p)
    
    Ensures fail graphs are closer to each other than to pass graphs.
    """
    
    def __init__(self, margin: float = 1.0, distance: str = 'cosine'):
        """
        Initialize Graph Contrastive Loss.
        
        Args:
            margin: Margin for triplet loss
            distance: Distance metric ('cosine' or 'euclidean')
        """
        super(GraphContrastiveLoss, self).__init__()
        self.margin = margin
        self.distance = distance
    
    def _compute_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distance."""
        if self.distance == 'cosine':
            # Cosine distance: 1 - cosine_similarity
            x1_norm = F.normalize(x1, p=2, dim=-1)
            x2_norm = F.normalize(x2, p=2, dim=-1)
            return 1 - (x1_norm * x2_norm).sum(dim=-1)
        elif self.distance == 'euclidean':
            return F.pairwise_distance(x1, x2, p=2)
        else:
            raise ValueError(f"Unknown distance: {self.distance}")
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute graph-level triplet loss.
        
        Args:
            anchor: Embeddings of augmented fail graphs [batch_size, embed_dim]
            positive: Embeddings of original fail graphs [batch_size, embed_dim]
            negative: Embeddings of pass graphs [batch_size, embed_dim]
        
        Returns:
            Scalar loss value
        """
        # Distance between anchor and positive (should be small)
        d_ap = self._compute_distance(anchor, positive)
        
        # Distance between anchor and negative (should be large)
        d_an = self._compute_distance(anchor, negative)
        
        # Triplet loss: max(0, d_ap - d_an + margin)
        loss = F.relu(d_ap - d_an + self.margin)
        
        return loss.mean()


class ListwiseLoss(nn.Module):
    """
    Listwise Ranking Loss for fault localization (Equation 5 in paper).
    
    Uses Softmax + CrossEntropy to rank faulty nodes higher.
    For a fail graph, faulty nodes (y=1) should have higher scores.
    
    This is equivalent to a multi-label ranking loss that emphasizes
    putting faulty functions at the top of the ranked list.
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Initialize Listwise Loss.
        
        Args:
            reduction: How to reduce loss ('mean', 'sum', or 'none')
        """
        super(ListwiseLoss, self).__init__()
        self.reduction = reduction
    
    def forward(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute listwise ranking loss.
        
        Args:
            scores: Predicted suspiciousness scores [num_nodes]
            labels: Ground truth labels [num_nodes], 1=faulty, 0=normal
            mask: Optional mask for valid nodes [num_nodes]
        
        Returns:
            Scalar loss value
        """
        # Apply mask if provided
        if mask is not None:
            scores = scores[mask]
            labels = labels[mask]
        
        # Filter only nodes that belong to functions (not Log/Package/File nodes)
        # Assume labels are -1 for non-function nodes, 0 for normal, 1 for faulty
        valid_mask = labels >= 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=scores.device)
        
        scores = scores[valid_mask]
        labels = labels[valid_mask].float()
        
        # Count faulty and normal nodes
        num_faulty = labels.sum()
        num_total = labels.numel()
        
        if num_faulty == 0 or num_faulty == num_total:
            # No contrast to learn from
            return torch.tensor(0.0, device=scores.device)
        
        # Compute softmax probabilities
        probs = F.softmax(scores, dim=0)
        
        # Cross-entropy loss: -Σ(y_i * log(p_i))
        # Where y_i is binary label (1 for faulty, 0 for normal)
        # This encourages faulty nodes to have higher probabilities
        loss = -torch.sum(labels * torch.log(probs + 1e-8))
        
        # Normalize by number of faulty nodes
        loss = loss / num_faulty
        
        return loss


class CombinedPhaseLoss(nn.Module):
    """
    Combined loss for Phase 1 (Representation Learning).
    
    Combines NodeContrastiveLoss and GraphContrastiveLoss with weights.
    """
    
    def __init__(
        self,
        node_weight: float = 1.0,
        graph_weight: float = 0.5,
        temperature: float = 0.07,
        margin: float = 1.0
    ):
        """
        Initialize Combined Phase 1 Loss.
        
        Args:
            node_weight: Weight for node-level contrastive loss
            graph_weight: Weight for graph-level contrastive loss
            temperature: Temperature for node contrastive loss
            margin: Margin for graph triplet loss
        """
        super(CombinedPhaseLoss, self).__init__()
        
        self.node_weight = node_weight
        self.graph_weight = graph_weight
        
        self.node_contrastive = NodeContrastiveLoss(temperature=temperature)
        self.graph_contrastive = GraphContrastiveLoss(margin=margin)
    
    def forward(
        self,
        z_f: torch.Tensor,
        z_f_aug: torch.Tensor,
        g_f: torch.Tensor,
        g_f_aug: torch.Tensor,
        g_p: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Compute combined Phase 1 loss.
        
        Args:
            z_f: Node embeddings from fail graph [num_nodes, dim]
            z_f_aug: Node embeddings from augmented fail graph [num_nodes, dim]
            g_f: Graph embedding from fail graph [batch_size, dim]
            g_f_aug: Graph embedding from augmented fail graph [batch_size, dim]
            g_p: Graph embedding from pass graph [batch_size, dim]
            node_mask: Optional mask for valid nodes
        
        Returns:
            Tuple of (total_loss, node_loss, graph_loss)
        """
        # Node-level contrastive loss
        node_loss = self.node_contrastive(z_f, z_f_aug, mask=node_mask)
        
        # Graph-level contrastive loss
        graph_loss = self.graph_contrastive(g_f_aug, g_f, g_p)
        
        # Combined loss
        total_loss = self.node_weight * node_loss + self.graph_weight * graph_loss
        
        return total_loss, node_loss, graph_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Useful for Phase 2 when there are many more normal nodes than faulty nodes.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter (higher = more focus on hard examples)
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            scores: Predicted scores [num_nodes]
            labels: Ground truth labels [num_nodes]
        
        Returns:
            Scalar loss value
        """
        # Convert to binary classification
        probs = torch.sigmoid(scores)
        
        # Focal loss formula
        ce_loss = F.binary_cross_entropy(probs, labels.float(), reduction='none')
        p_t = probs * labels + (1 - probs) * (1 - labels)
        focal_weight = (1 - p_t) ** self.gamma
        
        if self.alpha >= 0:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            focal_weight = alpha_t * focal_weight
        
        loss = focal_weight * ce_loss
        
        return loss.mean()


if __name__ == "__main__":
    # Test the loss functions
    print("Testing FALCON Loss Functions...")
    
    # Test Node Contrastive Loss
    print("\n1. Testing NodeContrastiveLoss:")
    node_loss = NodeContrastiveLoss(temperature=0.07)
    
    num_nodes = 50
    embed_dim = 64
    z1 = F.normalize(torch.randn(num_nodes, embed_dim), p=2, dim=-1)
    z2 = F.normalize(torch.randn(num_nodes, embed_dim), p=2, dim=-1)
    
    loss = node_loss(z1, z2)
    print(f"  Input shapes: z1={z1.shape}, z2={z2.shape}")
    print(f"  Loss: {loss.item():.4f}")
    
    # Test Graph Contrastive Loss
    print("\n2. Testing GraphContrastiveLoss:")
    graph_loss = GraphContrastiveLoss(margin=1.0)
    
    batch_size = 8
    anchor = torch.randn(batch_size, embed_dim)
    positive = torch.randn(batch_size, embed_dim)
    negative = torch.randn(batch_size, embed_dim)
    
    loss = graph_loss(anchor, positive, negative)
    print(f"  Input shapes: anchor={anchor.shape}")
    print(f"  Loss: {loss.item():.4f}")
    
    # Test Listwise Loss
    print("\n3. Testing ListwiseLoss:")
    listwise_loss = ListwiseLoss()
    
    scores = torch.randn(num_nodes)
    labels = torch.zeros(num_nodes)
    labels[[5, 10, 15]] = 1  # 3 faulty nodes
    
    loss = listwise_loss(scores, labels)
    print(f"  Input shapes: scores={scores.shape}, labels={labels.shape}")
    print(f"  Num faulty: {labels.sum().item()}")
    print(f"  Loss: {loss.item():.4f}")
    
    # Test Combined Loss
    print("\n4. Testing CombinedPhaseLoss:")
    combined_loss = CombinedPhaseLoss(node_weight=1.0, graph_weight=0.5)
    
    z_f = F.normalize(torch.randn(num_nodes, embed_dim), p=2, dim=-1)
    z_f_aug = F.normalize(torch.randn(num_nodes, embed_dim), p=2, dim=-1)
    g_f = torch.randn(batch_size, embed_dim)
    g_f_aug = torch.randn(batch_size, embed_dim)
    g_p = torch.randn(batch_size, embed_dim)
    
    total_loss, node_l, graph_l = combined_loss(z_f, z_f_aug, g_f, g_f_aug, g_p)
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Node loss: {node_l.item():.4f}")
    print(f"  Graph loss: {graph_l.item():.4f}")
    
    # Test Focal Loss
    print("\n5. Testing FocalLoss:")
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    
    scores = torch.randn(num_nodes)
    labels = torch.zeros(num_nodes)
    labels[[5, 10, 15]] = 1
    
    loss = focal_loss(scores, labels)
    print(f"  Loss: {loss.item():.4f}")
    
    print("\n✓ All loss tests passed!")

