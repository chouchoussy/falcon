"""
Training module for FALCON.
Implements two-phase training and loss functions.
"""

from .losses import (
    NodeContrastiveLoss,
    GraphContrastiveLoss,
    ListwiseLoss,
    CombinedPhaseLoss,
    FocalLoss
)

from .trainer import FALCONTrainer

from .augmentation import (
    augment_graph,
    augment_graph_pair
)

__all__ = [
    # Loss Functions
    'NodeContrastiveLoss',
    'GraphContrastiveLoss',
    'ListwiseLoss',
    'CombinedPhaseLoss',
    'FocalLoss',
    
    # Trainer
    'FALCONTrainer',
    
    # Augmentation
    'augment_graph',
    'augment_graph_pair',
]

