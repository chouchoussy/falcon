"""
Trainer module for FALCON.
Implements two-phase training: representation learning + fault localization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
from tqdm import tqdm

from .losses import NodeContrastiveLoss, GraphContrastiveLoss, ListwiseLoss, CombinedPhaseLoss


class FALCONTrainer:
    """
    Trainer for FALCON model with two-phase training.
    
    Phase 1: Representation Learning (Contrastive Learning)
        - Uses both pass and fail graphs
        - Learns semantic graph representations
        - Trains: Encoder + ProjectionHead
        - Loss: NodeContrastiveLoss + GraphContrastiveLoss
    
    Phase 2: Fault Localization (Ranking)
        - Uses only fail graphs
        - Learns to rank faulty functions
        - Trains: RankHead (+ fine-tune Encoder)
        - Loss: ListwiseLoss
    """
    
    def __init__(
        self,
        model,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate_phase1: float = 1e-3,
        learning_rate_phase2: float = 1e-4,
        weight_decay: float = 1e-5,
        node_loss_weight: float = 1.0,
        graph_loss_weight: float = 0.5,
        temperature: float = 0.07,
        margin: float = 1.0,
        augment_fn=None,
        freeze_encoder_phase2: bool = False
    ):
        """
        Initialize FALCON Trainer.
        
        Args:
            model: FALCONModel instance
            device: Device to train on
            learning_rate_phase1: Learning rate for Phase 1
            learning_rate_phase2: Learning rate for Phase 2
            weight_decay: Weight decay for optimizer
            node_loss_weight: Weight for node contrastive loss
            graph_loss_weight: Weight for graph contrastive loss
            temperature: Temperature for contrastive loss
            margin: Margin for triplet loss
            augment_fn: Function to augment graphs (from dataset module)
            freeze_encoder_phase2: Whether to freeze encoder in Phase 2
        """
        self.model = model.to(device)
        self.device = device
        
        self.learning_rate_phase1 = learning_rate_phase1
        self.learning_rate_phase2 = learning_rate_phase2
        self.weight_decay = weight_decay
        self.freeze_encoder_phase2 = freeze_encoder_phase2
        
        # Loss functions
        self.combined_loss = CombinedPhaseLoss(
            node_weight=node_loss_weight,
            graph_weight=graph_loss_weight,
            temperature=temperature,
            margin=margin
        )
        
        self.listwise_loss = ListwiseLoss()
        
        # Augmentation function
        self.augment_fn = augment_fn
        
        # Training history
        self.history = {
            'phase1': {'losses': [], 'node_losses': [], 'graph_losses': []},
            'phase2': {'losses': []}
        }
    
    def _get_graph_embedding(self, node_emb: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get graph-level embedding by pooling node embeddings.
        
        Args:
            node_emb: Node embeddings [num_nodes, dim]
            batch: Batch vector [num_nodes] (None for single graph)
        
        Returns:
            Graph embedding [1, dim] or [batch_size, dim]
        """
        if batch is None:
            # Single graph: mean pooling over all nodes
            return node_emb.mean(dim=0, keepdim=True)
        else:
            # Batched graphs: pool by batch
            return global_mean_pool(node_emb, batch)
    
    def train_phase1_representation(
        self,
        fail_graphs: List[Data],
        pass_graphs: List[Data],
        epochs: int = 10,
        batch_size: int = 32,
        drop_prob: float = 0.2,
        verbose: bool = True
    ) -> Dict:
        """
        Phase 1: Representation Learning with Contrastive Learning.
        
        Args:
            fail_graphs: List of failed test graphs (G_f)
            pass_graphs: List of passed test graphs (G_p)
            epochs: Number of training epochs
            batch_size: Batch size (not implemented for simplicity, processes one by one)
            drop_prob: Dropout probability for augmentation
            verbose: Whether to print progress
        
        Returns:
            Dictionary with training metrics
        """
        if verbose:
            print("=" * 70)
            print("PHASE 1: REPRESENTATION LEARNING (Contrastive)")
            print("=" * 70)
            print(f"Fail graphs: {len(fail_graphs)}")
            print(f"Pass graphs: {len(pass_graphs)}")
            print(f"Epochs: {epochs}")
            print(f"Learning rate: {self.learning_rate_phase1}")
        
        # Setup optimizer for Phase 1 (Encoder + ProjectionHead)
        optimizer = optim.Adam(
            list(self.model.encoder.parameters()) + 
            list(self.model.projection_head.parameters()),
            lr=self.learning_rate_phase1,
            weight_decay=self.weight_decay
        )
        
        self.model.train()
        
        # Check if augmentation function is available
        if self.augment_fn is None:
            print("Warning: No augmentation function provided. Using identity.")
            from ..dataset import augment_graph
            self.augment_fn = augment_graph
        
        for epoch in range(epochs):
            epoch_losses = []
            epoch_node_losses = []
            epoch_graph_losses = []
            
            # Create progress bar
            iterator = tqdm(range(len(fail_graphs)), desc=f"Epoch {epoch+1}/{epochs}") if verbose else range(len(fail_graphs))
            
            for i in iterator:
                # Get a fail graph and a pass graph
                g_f = fail_graphs[i].to(self.device)
                g_p = pass_graphs[i % len(pass_graphs)].to(self.device)  # Cycle through pass graphs
                
                # Create augmented fail graph
                g_f_aug = self.augment_fn(g_f, drop_prob=drop_prob)
                g_f_aug = g_f_aug.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass for fail graph
                outputs_f = self.model(
                    g_f.x, g_f.edge_index,
                    return_projection=True,
                    return_rank=False
                )
                node_emb_f = outputs_f['node_emb']
                proj_emb_f = outputs_f['proj_emb']
                
                # Forward pass for augmented fail graph
                outputs_f_aug = self.model(
                    g_f_aug.x, g_f_aug.edge_index,
                    return_projection=True,
                    return_rank=False
                )
                node_emb_f_aug = outputs_f_aug['node_emb']
                proj_emb_f_aug = outputs_f_aug['proj_emb']
                
                # Forward pass for pass graph
                outputs_p = self.model(
                    g_p.x, g_p.edge_index,
                    return_projection=True,
                    return_rank=False
                )
                node_emb_p = outputs_p['node_emb']
                proj_emb_p = outputs_p['proj_emb']
                
                # Get graph-level embeddings (pooling)
                graph_emb_f = self._get_graph_embedding(proj_emb_f)
                graph_emb_f_aug = self._get_graph_embedding(proj_emb_f_aug)
                graph_emb_p = self._get_graph_embedding(proj_emb_p)
                
                # Compute combined loss
                # Node embeddings should have same size (use min size if different)
                min_nodes = min(proj_emb_f.shape[0], proj_emb_f_aug.shape[0])
                
                total_loss, node_loss, graph_loss = self.combined_loss(
                    proj_emb_f[:min_nodes],
                    proj_emb_f_aug[:min_nodes],
                    graph_emb_f,
                    graph_emb_f_aug,
                    graph_emb_p
                )
                
                # Backward and optimize
                total_loss.backward()
                optimizer.step()
                
                # Record losses
                epoch_losses.append(total_loss.item())
                epoch_node_losses.append(node_loss.item())
                epoch_graph_losses.append(graph_loss.item())
                
                if verbose and isinstance(iterator, tqdm):
                    iterator.set_postfix({
                        'loss': f"{total_loss.item():.4f}",
                        'node': f"{node_loss.item():.4f}",
                        'graph': f"{graph_loss.item():.4f}"
                    })
            
            # Epoch summary
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            avg_node_loss = sum(epoch_node_losses) / len(epoch_node_losses)
            avg_graph_loss = sum(epoch_graph_losses) / len(epoch_graph_losses)
            
            self.history['phase1']['losses'].append(avg_loss)
            self.history['phase1']['node_losses'].append(avg_node_loss)
            self.history['phase1']['graph_losses'].append(avg_graph_loss)
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Loss: {avg_loss:.4f}, "
                      f"Node: {avg_node_loss:.4f}, "
                      f"Graph: {avg_graph_loss:.4f}")
        
        if verbose:
            print("✓ Phase 1 training completed!")
        
        return self.history['phase1']
    
    def train_phase2_ranking(
        self,
        fail_graphs: List[Data],
        epochs: int = 10,
        batch_size: int = 32,
        verbose: bool = True
    ) -> Dict:
        """
        Phase 2: Fault Localization with Ranking Loss.
        
        Args:
            fail_graphs: List of failed test graphs with labels (y)
            epochs: Number of training epochs
            batch_size: Batch size (not implemented for simplicity)
            verbose: Whether to print progress
        
        Returns:
            Dictionary with training metrics
        """
        if verbose:
            print("\n" + "=" * 70)
            print("PHASE 2: FAULT LOCALIZATION (Ranking)")
            print("=" * 70)
            print(f"Fail graphs: {len(fail_graphs)}")
            print(f"Epochs: {epochs}")
            print(f"Learning rate: {self.learning_rate_phase2}")
            print(f"Freeze encoder: {self.freeze_encoder_phase2}")
        
        # Optionally freeze encoder
        if self.freeze_encoder_phase2:
            self.model.freeze_encoder()
            if verbose:
                print("  Encoder frozen - only training RankHead")
            
            # Optimizer only for RankHead
            optimizer = optim.Adam(
                self.model.rank_head.parameters(),
                lr=self.learning_rate_phase2,
                weight_decay=self.weight_decay
            )
        else:
            # Fine-tune encoder + train RankHead
            optimizer = optim.Adam(
                list(self.model.encoder.parameters()) + 
                list(self.model.rank_head.parameters()),
                lr=self.learning_rate_phase2,
                weight_decay=self.weight_decay
            )
        
        self.model.train()
        
        for epoch in range(epochs):
            epoch_losses = []
            
            iterator = tqdm(fail_graphs, desc=f"Epoch {epoch+1}/{epochs}") if verbose else fail_graphs
            
            for g_f in iterator:
                g_f = g_f.to(self.device)
                
                # Check if graph has labels
                if not hasattr(g_f, 'y') or g_f.y is None:
                    continue
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(
                    g_f.x, g_f.edge_index,
                    return_projection=False,
                    return_rank=True
                )
                
                rank_scores = outputs['rank_score']
                
                # Compute listwise ranking loss
                loss = self.listwise_loss(rank_scores, g_f.y)
                
                if loss.item() > 0:  # Only backprop if loss is valid
                    loss.backward()
                    optimizer.step()
                    
                    epoch_losses.append(loss.item())
                
                if verbose and isinstance(iterator, tqdm):
                    if epoch_losses:
                        iterator.set_postfix({'loss': f"{epoch_losses[-1]:.4f}"})
            
            # Epoch summary
            if epoch_losses:
                avg_loss = sum(epoch_losses) / len(epoch_losses)
                self.history['phase2']['losses'].append(avg_loss)
                
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - No valid loss computed")
        
        # Unfreeze encoder if it was frozen
        if self.freeze_encoder_phase2:
            self.model.unfreeze_encoder()
        
        if verbose:
            print("✓ Phase 2 training completed!")
        
        return self.history['phase2']
    
    def predict(
        self,
        graph: Data,
        return_top_k: int = None
    ) -> torch.Tensor:
        """
        Predict suspiciousness scores for a graph.
        
        Args:
            graph: Test graph (Data object)
            return_top_k: If specified, return only top-k node indices and scores
        
        Returns:
            Suspiciousness scores [num_nodes] or top-k (indices, scores) tuple
        """
        self.model.eval()
        
        graph = graph.to(self.device)
        
        with torch.no_grad():
            scores = self.model.predict_rank(graph.x, graph.edge_index)
        
        if return_top_k is not None:
            top_k = min(return_top_k, len(scores))
            top_scores, top_indices = torch.topk(scores, k=top_k)
            return top_indices.cpu(), top_scores.cpu()
        
        return scores.cpu()
    
    def evaluate(
        self,
        test_graphs: List[Data],
        top_k_values: List[int] = [1, 3, 5, 10],
        verbose: bool = True
    ) -> Dict:
        """
        Evaluate model on test graphs.
        
        Args:
            test_graphs: List of test graphs with labels
            top_k_values: List of k values for Top-K accuracy
            verbose: Whether to print results
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        metrics = {
            'top_k_acc': {k: [] for k in top_k_values},
            'mrr': [],  # Mean Reciprocal Rank
            'mfr': []   # Mean First Rank
        }
        
        for graph in test_graphs:
            graph = graph.to(self.device)
            
            if not hasattr(graph, 'y') or graph.y is None:
                continue
            
            with torch.no_grad():
                scores = self.model.predict_rank(graph.x, graph.edge_index)
            
            # Get ground truth faulty nodes
            faulty_indices = (graph.y == 1).nonzero(as_tuple=True)[0]
            
            if len(faulty_indices) == 0:
                continue
            
            # Rank nodes by scores (descending)
            ranked_indices = torch.argsort(scores, descending=True)
            
            # Find positions of faulty nodes in ranked list
            faulty_positions = []
            for faulty_idx in faulty_indices:
                pos = (ranked_indices == faulty_idx).nonzero(as_tuple=True)[0].item()
                faulty_positions.append(pos + 1)  # 1-indexed
            
            # Top-K accuracy
            for k in top_k_values:
                top_k_set = set(ranked_indices[:k].cpu().tolist())
                hit = any(idx.item() in top_k_set for idx in faulty_indices)
                metrics['top_k_acc'][k].append(1 if hit else 0)
            
            # MRR: Mean Reciprocal Rank (1 / rank of first faulty)
            first_rank = min(faulty_positions)
            metrics['mrr'].append(1.0 / first_rank)
            
            # MFR: Mean First Rank
            metrics['mfr'].append(first_rank)
        
        # Compute averages
        results = {}
        for k in top_k_values:
            if metrics['top_k_acc'][k]:
                results[f'Top-{k}'] = sum(metrics['top_k_acc'][k]) / len(metrics['top_k_acc'][k])
        
        if metrics['mrr']:
            results['MRR'] = sum(metrics['mrr']) / len(metrics['mrr'])
        
        if metrics['mfr']:
            results['MFR'] = sum(metrics['mfr']) / len(metrics['mfr'])
        
        if verbose:
            print("\n" + "=" * 70)
            print("EVALUATION RESULTS")
            print("=" * 70)
            for metric, value in results.items():
                print(f"{metric:10} {value:.4f}")
        
        return results
    
    def save_checkpoint(self, path: str, epoch: int = None, phase: str = None):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
            'epoch': epoch,
            'phase': phase
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        return checkpoint


if __name__ == "__main__":
    # Test the trainer
    print("Testing FALCON Trainer...")
    
    from ..models import FALCONModel
    from ..dataset import augment_graph
    
    # Create dummy model
    model = FALCONModel(
        input_dim=384,
        hidden_dim=128,
        embedding_dim=64
    )
    
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Create dummy data
    num_graphs = 5
    fail_graphs = []
    pass_graphs = []
    
    for i in range(num_graphs):
        num_nodes = 30
        x = torch.randn(num_nodes, 384)
        edge_index = torch.randint(0, num_nodes, (2, 50))
        y = torch.zeros(num_nodes, dtype=torch.long)
        y[[5, 10]] = 1  # 2 faulty nodes
        
        fail_graphs.append(Data(x=x, edge_index=edge_index, y=y))
        pass_graphs.append(Data(x=x, edge_index=edge_index))
    
    print(f"\nCreated {len(fail_graphs)} fail graphs and {len(pass_graphs)} pass graphs")
    
    # Create trainer
    trainer = FALCONTrainer(
        model=model,
        device='cpu',
        augment_fn=augment_graph
    )
    
    print("\n--- Testing Phase 1 ---")
    history_p1 = trainer.train_phase1_representation(
        fail_graphs=fail_graphs,
        pass_graphs=pass_graphs,
        epochs=2,
        verbose=True
    )
    
    print("\n--- Testing Phase 2 ---")
    history_p2 = trainer.train_phase2_ranking(
        fail_graphs=fail_graphs,
        epochs=2,
        verbose=True
    )
    
    print("\n--- Testing Prediction ---")
    test_graph = fail_graphs[0]
    scores = trainer.predict(test_graph)
    print(f"Predicted scores shape: {scores.shape}")
    
    top_indices, top_scores = trainer.predict(test_graph, return_top_k=5)
    print(f"Top-5 nodes: {top_indices.tolist()}")
    print(f"Top-5 scores: {[f'{s:.4f}' for s in top_scores.tolist()]}")
    
    print("\n--- Testing Evaluation ---")
    results = trainer.evaluate(fail_graphs, verbose=True)
    
    print("\n✓ All trainer tests passed!")

