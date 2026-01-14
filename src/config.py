"""
Configuration file for FALCON pipeline.
Contains constants, paths, and hyperparameters.
"""

import torch
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

# Data paths
RAW_DATA_PATH = "../data_tcpdump"
PROCESSED_PATH = "./processed_data"
GROUND_TRUTH_FILE = "ground_truth.json"

# Output paths
RESULT_CSV_PATH = "./results/falcon_results.csv"
CHECKPOINT_DIR = "./checkpoints"
LOG_DIR = "./logs"

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================

# Node feature dimension (SentenceBERT output)
INPUT_DIM = 384

# Model architecture
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
PROJECTION_HIDDEN_DIM = 256
PROJECTION_OUTPUT_DIM = 64
RANK_HIDDEN_DIM = 64

# GGNN parameters
NUM_GNN_LAYERS = 3
NUM_GNN_STEPS = 5
DROPOUT = 0.1

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================

# Phase 1: Representation Learning
PHASE1_EPOCHS = 10
LEARNING_RATE_PHASE1 = 1e-3
NODE_LOSS_WEIGHT = 1.0
GRAPH_LOSS_WEIGHT = 0.5
TEMPERATURE = 0.07
MARGIN = 1.0
AUGMENTATION_DROP_PROB = 0.2

# Phase 2: Fault Localization
PHASE2_EPOCHS = 10
LEARNING_RATE_PHASE2 = 1e-4
FREEZE_ENCODER_PHASE2 = False

# Optimization
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 1  # Process one graph at a time

# Memory Optimization (for large graphs)
NODE_LOSS_CHUNK_SIZE = 5000  # Chunk size for node contrastive loss computation
NODE_LOSS_MAX_NODES = 50000  # Max nodes to process (will sample if exceeded)

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_MIXED_PRECISION = False

# ============================================================================
# EVALUATION METRICS
# ============================================================================

TOP_K_VALUES = [1, 3, 5, 10]

# ============================================================================
# LOOCV CONFIGURATION
# ============================================================================

# Whether to use cached processed graphs
USE_CACHE = True

# Whether to save checkpoints during LOOCV
SAVE_CHECKPOINTS = False

# Verbose logging
VERBOSE = True

# ============================================================================
# SENTENCEBERT CONFIGURATION
# ============================================================================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def ensure_directories():
    """Create necessary directories if they don't exist."""
    Path(PROCESSED_PATH).mkdir(parents=True, exist_ok=True)
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    Path(RESULT_CSV_PATH).parent.mkdir(parents=True, exist_ok=True)


def print_config():
    """Print current configuration."""
    print("=" * 70)
    print("FALCON CONFIGURATION")
    print("=" * 70)
    print(f"Data Path:        {RAW_DATA_PATH}")
    print(f"Processed Path:   {PROCESSED_PATH}")
    print(f"Device:           {DEVICE}")
    print(f"\nModel Architecture:")
    print(f"  Input Dim:      {INPUT_DIM}")
    print(f"  Hidden Dim:     {HIDDEN_DIM}")
    print(f"  Embedding Dim:  {EMBEDDING_DIM}")
    print(f"  GNN Layers:     {NUM_GNN_LAYERS}")
    print(f"\nTraining:")
    print(f"  Phase 1 Epochs: {PHASE1_EPOCHS}")
    print(f"  Phase 2 Epochs: {PHASE2_EPOCHS}")
    print(f"  LR Phase 1:     {LEARNING_RATE_PHASE1}")
    print(f"  LR Phase 2:     {LEARNING_RATE_PHASE2}")
    print(f"  Aug Drop Prob:  {AUGMENTATION_DROP_PROB}")
    print("=" * 70)

