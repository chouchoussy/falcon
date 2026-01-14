"""
Configuration for Data Preprocessing Module.
Independent from the training pipeline.
"""

from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

# Raw data directory (relative to this module)
RAW_DATA_PATH = "../../data_tcpdump"

# Output directory for processed graphs
OUTPUT_PATH = "../processed_data"

# Ground truth file name
GROUND_TRUTH_FILE = "ground_truth.json"

# ============================================================================
# SENTENCEBERT CONFIGURATION
# ============================================================================

# SentenceBERT model for text embeddings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Embedding dimension (output from SentenceBERT)
EMBEDDING_DIM = 384

# ============================================================================
# LOG PARSING
# ============================================================================

# Regex pattern for extracting function calls from logs
LOG_PATTERN = r'Call\s+([^\s]+)\s+.*?\(([\w\-\.]+):\d+\)'

# ============================================================================
# GRAPH STRUCTURE
# ============================================================================

# Node type IDs
NODE_TYPE_LOG = 0
NODE_TYPE_PACKAGE = 1
NODE_TYPE_FILE = 2
NODE_TYPE_METHOD = 3

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_raw_data_path() -> Path:
    """Get absolute path to raw data directory."""
    return (Path(__file__).parent / RAW_DATA_PATH).resolve()

def get_output_path() -> Path:
    """Get absolute path to output directory."""
    return (Path(__file__).parent / OUTPUT_PATH).resolve()

def ensure_output_dir() -> Path:
    """Create output directory if it doesn't exist."""
    output_path = get_output_path()
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path

