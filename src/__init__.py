"""
FALCON: Fault Localization with Contrastive Learning.
A deep learning approach for software fault localization using graph neural networks.
"""

__version__ = "1.0.0"
__author__ = "FALCON Team"

# Make src a proper package
from . import config
from . import dataset
from . import models
from . import training
from . import utils

__all__ = [
    'config',
    'dataset',
    'models',
    'training',
    'utils',
]

