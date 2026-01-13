# FALCON: Fault Localization with Contrastive Learning

Implementation of FALCON (ICSE'25) - A deep learning approach for software fault localization using Graph Neural Networks and Contrastive Learning.

## 📋 Overview

FALCON uses a **two-phase training strategy** to localize faults in software:

1. **Phase 1: Representation Learning** - Learn semantic graph representations using contrastive learning
2. **Phase 2: Fault Localization** - Fine-tune the model to rank faulty functions

### Key Features

- **Graph-based Representation**: Converts execution traces to heterogeneous graphs
- **GGNN Encoder**: Gated Graph Neural Network for graph encoding
- **Adaptive Graph Augmentation (AGA)**: Transitive closure-based augmentation
- **Contrastive Learning**: Node-level and graph-level contrastive objectives
- **Listwise Ranking**: Learns to rank faulty functions higher
- **LOOCV Evaluation**: Leave-One-Out Cross Validation for robust evaluation

## 🏗️ Project Structure

```
Falcon/
├── src/                          # Source code
│   ├── config.py                 # Configuration & hyperparameters
│   ├── dataset/                  # Data processing
│   │   ├── log_parser.py         # Parse execution logs
│   │   ├── graph_builder.py      # Build PyG graphs
│   │   └── augmentation.py       # Adaptive graph augmentation
│   ├── models/                   # Neural network models
│   │   ├── encoder.py            # GGNN encoder
│   │   ├── heads.py              # Projection & Rank heads
│   │   └── __init__.py           # FALCONModel
│   ├── training/                 # Training logic
│   │   ├── losses.py             # Loss functions (Eqs 3, 4, 5)
│   │   └── trainer.py            # Two-phase trainer
│   └── utils/                    # Utilities
│       └── metrics.py            # Evaluation metrics
├── processed_data/               # Cached graph data (.pt files)
├── results/                      # Output results (CSV)
├── checkpoints/                  # Model checkpoints (optional)
├── main.py                       # Main pipeline script
└── README.md                     # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric
- CUDA (optional, for GPU acceleration)

### Installation

```bash
# Clone or navigate to project directory
cd Falcon

# Create virtual environment (recommended)
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Data Preparation

Ensure your data is organized as:

```
../data_tcpdump/
├── v1-12896/
│   └── fail.log           # Execution trace
├── v2-13141/
│   └── fail.log
├── ...
└── ground_truth.json      # Faulty function labels
```

**Ground Truth Format:**
```json
{
  "filename.c::function_name": true,
  "another.c::normal_func": false,
  ...
}
```

## 📊 Usage

### Quick Start

```bash
# Run complete FALCON pipeline with LOOCV
python main.py
```

This will:
1. Load and preprocess all execution traces
2. Run Leave-One-Out Cross Validation (LOOCV)
3. Train FALCON for each fold (Phase 1 + Phase 2)
4. Evaluate and save results to `results/falcon_results.csv`

**⏱️ Estimated Runtime:**
- **First run (CPU)**: ~5-9 hours (includes graph building)
- **First run (GPU)**: ~1.5-3 hours
- **Subsequent runs (with cache)**: ~4.5-9 hours (CPU) or ~1.5-2.7 hours (GPU)
- **Quick test (5 epochs, 3 folds)**: ~10-30 minutes

See [ESTIMATE_TIME.md](ESTIMATE_TIME.md) for detailed breakdown.

### Configuration

Edit `src/config.py` to customize:

```python
# Model Architecture
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
NUM_GNN_LAYERS = 3

# Training
PHASE1_EPOCHS = 10          # Contrastive learning epochs
PHASE2_EPOCHS = 10          # Ranking epochs
LEARNING_RATE_PHASE1 = 1e-3
LEARNING_RATE_PHASE2 = 1e-4

# Augmentation
AUGMENTATION_DROP_PROB = 0.2

# Device
DEVICE = "cuda"  # or "cpu"
```

## 📈 Evaluation Metrics

FALCON reports the following metrics:

- **Top-K Accuracy**: % of test cases where faulty function is in top-K predictions
  - Top-1, Top-3, Top-5, Top-10
- **MFR (Mean First Rank)**: Average rank of first faulty function (lower is better)
- **MRR (Mean Reciprocal Rank)**: Average of 1/rank (higher is better)

### Example Results

```
=================================================================
                    FALCON Final Results
=================================================================

Top-K Accuracy (%):
  Top-1       65.00%
  Top-3       82.50%
  Top-5       91.25%
  Top-10      97.50%

Ranking Metrics:
  MFR          2.30  (Mean First Rank)
  MRR        0.7300  (Mean Reciprocal Rank)

Statistics:
  Count          40  (Number of test cases)
  Median        1.0
  Min Rank        1
  Max Rank       15
=================================================================
```

## 🔬 Architecture Details

### Graph Structure

FALCON constructs heterogeneous graphs from execution traces:

- **Nodes**: Log entries, Packages, Files, Methods
- **Edges**:
  - Hierarchical: Log → Package → File → Method
  - Sequential: Method_i → Method_{i+1} (control flow)
- **Features**: SentenceBERT embeddings (384-dim)
- **Labels**: Binary (1=faulty, 0=normal)

### Model Architecture

```
Input Graph (x, edge_index)
         ↓
  [GraphEncoder (GGNN)]
    - Input Projection
    - 3× GGNN Layers
    - Layer Normalization
    - Residual Connections
         ↓
    Node Embeddings [num_nodes, 64]
         ↓
    ┌─────────────┴─────────────┐
    ↓                           ↓
[ProjectionHead]          [RankHead]
(Phase 1)                 (Phase 2)
Linear→ReLU→Linear        Linear→Linear
    ↓                           ↓
Projected Embeddings      Suspiciousness Scores
[num_nodes, 64]           [num_nodes]
```

### Loss Functions

**Phase 1: Contrastive Learning**
- **Node Contrastive Loss** (Eq. 3): InfoNCE loss
  ```
  L_node = -log(exp(sim(z_i, z'_i)/τ) / Σ_k exp(sim(z_i, z'_k)/τ))
  ```
- **Graph Contrastive Loss** (Eq. 4): Triplet margin loss
  ```
  L_graph = max(0, d(anchor, positive) - d(anchor, negative) + margin)
  ```

**Phase 2: Fault Localization**
- **Listwise Loss** (Eq. 5): Softmax + CrossEntropy
  ```
  L_rank = -Σ(y_i * log(softmax(scores)_i))
  ```

## 🎯 Performance Tips

### Speed Optimization

1. **Enable Caching**: Set `USE_CACHE = True` in config (default)
   - Processed graphs are cached as `.pt` files
   - Subsequent runs load from cache (~100x faster)

2. **Use GPU**: Set `DEVICE = "cuda"` in config
   - Requires NVIDIA GPU with CUDA support

3. **Reduce Epochs**: For quick testing
   ```python
   PHASE1_EPOCHS = 5
   PHASE2_EPOCHS = 5
   ```

### Memory Management

For limited GPU memory:
- Reduce `HIDDEN_DIM` (e.g., 64 instead of 128)
- Reduce `NUM_GNN_LAYERS` (e.g., 2 instead of 3)
- Enable gradient checkpointing (requires code modification)

## 📝 File Formats

### Input: Execution Log

```
Call main from source (tcpdump.c:123)
Call process_packet from source (print-ip.c:45)
Call decode_header from source (print-ip.c:67)
...
```

Pattern: `Call <function_name> ... (<filename>:<line>)`

### Output: Results CSV

```csv
Version,Rank,Reciprocal_Rank,Top1,Top3,Top5,Top10
v1-12896,1,1.0000,1,1,1,1
v2-13141,3,0.3333,0,1,1,1
v3-14552,2,0.5000,0,1,1,1
...

Summary Metrics
Metric,Value
Top-1,65.0000
Top-5,91.2500
MFR,2.3000
MRR,0.7300
```

