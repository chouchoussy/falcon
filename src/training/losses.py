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
    """
    
    def __init__(self, temperature: float = 0.07, chunk_size: int = 5000, max_nodes: int = 50000):
        """
        Initialize Node Contrastive Loss.
        
        Args:
            temperature: Temperature parameter (τ) for softmax scaling
            chunk_size: Chunk size for memory-efficient computation (default: 5000)
            max_nodes: Maximum nodes to process (if exceeded, will sample)
        """
        super(NodeContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.max_nodes = max_nodes
    
    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
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
        
        # If graph is too large, sample nodes to avoid OOM
        if num_nodes > self.max_nodes:
            indices = torch.randperm(num_nodes, device=z1.device)[:self.max_nodes]
            z1 = z1[indices]
            z2 = z2[indices]
            num_nodes = self.max_nodes
        
        losses = []
        
        for i in range(0, num_nodes, self.chunk_size):
            end_idx = min(i + self.chunk_size, num_nodes)
            z1_chunk = z1[i:end_idx]  # [chunk_size, embed_dim]
            z2_chunk = z2[i:end_idx]  # [chunk_size, embed_dim] 

            # 1. Tử số (Positive pair): z1_i vs z2_i
            pos_sim = (z1_chunk * z2_chunk).sum(dim=1) / self.temperature

            # 2. Mẫu số phần 1 (Inter-view negatives): z1_i vs tất cả z2_k
            sim_z1_z2 = torch.matmul(z1_chunk, z2.t()) / self.temperature
            
            # 3. Mẫu số phần 2 (Intra-view negatives): z1_i vs tất cả z1_k (với k != i)
            sim_z1_z1 = torch.matmul(z1_chunk, z1.t()) / self.temperature
            
            # Cần loại bỏ z1_i vs z1_i (k=i) ra khỏi mẫu số phần 2
            # Tạo mask cho chunk hiện tại
            # Mask có kích thước [chunk_size, num_nodes]
            # Giá trị 1 ở vị trí diagonal tương ứng
            diag_mask = torch.zeros_like(sim_z1_z1, dtype=torch.bool)
            
            # Điền True vào đường chéo tương ứng với chunk này
            # Các node từ i đến end_idx trong batch lớn tương ứng với cột i đến end_idx
            chunk_indices = torch.arange(end_idx - i, device=z1.device)
            global_indices = torch.arange(i, end_idx, device=z1.device)
            diag_mask[chunk_indices, global_indices] = True
            
            # Gán giá trị cực nhỏ (-inf) cho vị trí chính nó để exp(-inf) = 0
            sim_z1_z1.masked_fill_(diag_mask, -float('inf'))
            
            # 4. Gom tất cả lại vào mẫu số (LogSumExp)
            # Mẫu số = exp(sim_z1_z2) + exp(sim_z1_z1_no_diag)
            # Để dùng logsumexp ổn định số học, ta nối 2 ma trận lại
            # combined_sim: [chunk_size, 2 * num_nodes]
            combined_sim = torch.cat([sim_z1_z2, sim_z1_z1], dim=1)
            
            # L_i = - log( exp(pos) / sum(exp(all)) )
            #     = - pos + log(sum(exp(all)))
            chunk_losses = -pos_sim + torch.logsumexp(combined_sim, dim=1)

            losses.append(chunk_losses)
        
        # Concatenate and average
        loss = torch.cat(losses).mean()
        
        return loss


class GraphContrastiveLoss(nn.Module):
    """
    Graph-level Contrastive Loss (Equation 4 in paper).
    
    Triplet Margin Loss for graph-level embeddings.
    - Anchor: Augmented fail graph (G'_f)
    - Positive: Original fail graph (G_f)
    - Negative: Pass graph (G_p)
    
    """
    
    def __init__(self, margin: float = 1.0, distance: str = 'cosine'):
        """
        Initialize Graph Contrastive Loss.
        
        Args:
            margin: Margin for triplet loss
            distance: Euclidean distance (L2 norm squared).
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
            # Paper uses Squared L2 Norm: ||x1 - x2||_2^2
            # F.pairwise_distance returns L2 norm (without square)
            dist = F.pairwise_distance(x1, x2, p=2)
            return dist.pow(2)  # SQUARED L2
        else:
            raise ValueError(f"Unknown distance: {self.distance}")
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:

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
        margin: float = 1.0,
        node_loss_chunk_size: int = 5000,
        node_loss_max_nodes: int = 50000
    ):
        """
        Initialize Combined Phase 1 Loss.
        
        Args:
            node_weight: Weight for node-level contrastive loss
            graph_weight: Weight for graph-level contrastive loss
            temperature: Temperature for node contrastive loss
            margin: Margin for graph triplet loss
            node_loss_chunk_size: Chunk size for node contrastive loss (memory optimization)
            node_loss_max_nodes: Max nodes for node contrastive loss (will sample if exceeded)
        """
        super(CombinedPhaseLoss, self).__init__()
        
        self.node_weight = node_weight
        self.graph_weight = graph_weight
        
        self.node_contrastive = NodeContrastiveLoss(
            temperature=temperature,
            chunk_size=node_loss_chunk_size,
            max_nodes=node_loss_max_nodes
        )
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

