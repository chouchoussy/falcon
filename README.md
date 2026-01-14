# FALCON: Fault Localization with Contrastive Learning

Implementation of FALCON (ICSE'25) - A deep learning approach for software fault localization using Graph Neural Networks and Contrastive Learning.

## 📋 Overview

FALCON uses a **two-phase training strategy** to localize faults in software:

1. **Phase 1: Representation Learning** - Learn semantic graph representations using contrastive learning
2. **Phase 2: Fault Localization** - Fine-tune the model to rank faulty functions

## 🏗️ Project Structure

```
Falcon/
├── data_preprocessing/           # 📦 DATA PREPROCESSING MODULE (Independent)
│   ├── config.py                 # Config for preprocessing
│   ├── log_parser.py             # Parse execution logs
│   ├── graph_builder.py          # Build PyG graphs
│   ├── preprocess.py             # Main preprocessing script
│   └── __init__.py
│
├── processed_data/               # 💾 PREPROCESSED GRAPHS (.pt files)
│
├── src/                          # 🧠 TRAINING MODULE
│   ├── config.py                 # Config for training
│   ├── models/                   # Neural network models
│   │   ├── encoder.py            # GGNN encoder
│   │   ├── heads.py              # Projection & Rank heads
│   │   └── __init__.py
│   ├── training/                 # Training logic
│   │   ├── losses.py             # Loss functions
│   │   ├── trainer.py            # Two-phase trainer
│   │   ├── augmentation.py       # Graph augmentation
│   │   └── __init__.py
│   └── utils/                    # Utilities
│       └── metrics.py            # Evaluation metrics
│
├── training.py                   # 🚀 Training script
├── results/                      # 📊 Training results (CSV, JSON)
└── README.md
```

## 🔄 Workflow

FALCON có **2 modules độc lập**:

### 1️⃣ Data Preprocessing Module (`data_preprocessing/`)
- **Mục đích**: Parse logs và build graphs từ raw data
- **Input**: `../data_tcpdump/`
- **Output**: `processed_data/*.pt`
- **Độc lập**: Có config và dependencies riêng

### 2️⃣ Training Module (`src/` + `training.py`)
- **Mục đích**: Train model và evaluate
- **Input**: `processed_data/*.pt` 
- **Output**: `results/*.csv`, `results/*.json`
- **Độc lập**: Chỉ đọc từ processed_data, không cần raw data

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric
- CUDA (optional)

### Installation

```bash
cd Falcon

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## 📊 Usage

### Step 1: Data Preprocessing

**Option A: Python Script (Local)**

```bash
cd data_preprocessing

# Run preprocessing
python preprocess.py

# Options:
python preprocess.py --force                        # Force rebuild
python preprocess.py --versions v1-12896 v2-12893  # Specific versions
```

**Option B: Jupyter Notebook (Colab/Kaggle/Local)**

```bash
cd data_preprocessing
jupyter notebook preprocess.ipynb

# Or upload preprocess.ipynb to Google Colab / Kaggle
# See data_preprocessing/NOTEBOOK_GUIDE.md for details
```

**Output**: `../processed_data/*.pt` files

---

### Step 2: Training

```bash
cd ..  # Back to Falcon/

# Run training (80/20 split)
python training.py

# Options:
python training.py --train_ratio 0.7      # 70% train, 30% test
python training.py --epochs1 5 --epochs2 5  # Custom epochs
python training.py --seed 123             # Different seed
```

**Output**: `results/falcon_results_*.csv` and `.json`

---

### Quick Start (All Steps)

```bash
cd Falcon
source venv/bin/activate

# Step 1: Preprocess
cd data_preprocessing
python preprocess.py
cd ..

# Step 2: Train
python training.py
```

---

## ☁️ Kaggle Training (Recommended for GPU)

**Chạy training trên Kaggle với GPU T4 (free)** - Nhanh hơn và không tốn tài nguyên local!

### Quick Setup

1. **Upload code lên GitHub** (xem `KAGGLE_SETUP.md`)
2. **Upload dataset lên Kaggle Datasets**
3. **Tạo Kaggle Notebook** với GPU enabled
4. **Run `kaggle_training.ipynb`**

### Detailed Guide

📖 **Xem hướng dẫn chi tiết**: [`KAGGLE_SETUP.md`](KAGGLE_SETUP.md)

**Lợi ích**:
- ✅ GPU T4 free (nhanh hơn CPU 10-20x)
- ✅ Không tốn RAM local
- ✅ Dễ share và reproduce
- ✅ Auto-save results

**Expected Runtime**: 20-45 phút (vs 2-4 giờ trên CPU)

---

## 💻 CPU/GPU Support

FALCON hỗ trợ **cả CPU và GPU** với khả năng **cross-platform**:

### Device Detection

Training tự động detect device:

```bash
# Auto-detect (recommended)
python training.py

# Force CPU
python training.py --device cpu

# Force GPU (if available)
python training.py --device cuda
```

### Cross-Platform Workflow

✅ **Preprocessing trên GPU (Kaggle) → Training trên CPU (Local)**
```bash
# Kaggle: Run preprocess.ipynb với T4 GPU
# Download processed_data/ về local
# MacBook: python training.py  # Auto-load với CPU
```

✅ **Preprocessing trên CPU → Training trên GPU**
```bash
# Local: cd data_preprocessing && python preprocess.py
# Upload processed_data/ lên Colab/Kaggle
# GPU: python training.py --device cuda
```

**Lý do hoạt động**: 
- Preprocessing tạo files `.pt` portable (device-independent)
- Training tự động chuyển đổi tensors với `map_location`

📖 Xem thêm: `CPU_GPU_SUPPORT.md`

---

## ⚙️ Configuration

### Preprocessing Config

Edit `data_preprocessing/config.py`:

```python
RAW_DATA_PATH = "../../data_tcpdump"
OUTPUT_PATH = "../processed_data"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
```

### Training Config

Edit `src/config.py`:

```python
# Model
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
NUM_GNN_LAYERS = 3

# Training
PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 10
LEARNING_RATE_PHASE1 = 1e-3
LEARNING_RATE_PHASE2 = 1e-4

# Device
DEVICE = "cuda"  # or "cpu"
```

## 📈 Evaluation Metrics

- **Top-K Accuracy**: % of test cases where faulty function is in top-K
- **MFR (Mean First Rank)**: Average rank (lower is better)
- **MRR (Mean Reciprocal Rank)**: Average of 1/rank (higher is better)

### Example Results

```
======================================================================
                         FALCON Results
======================================================================

Top-K Accuracy (%):
  Top-1       65.00%
  Top-3       82.50%
  Top-5       91.25%
  Top-10      97.50%

Ranking Metrics:
  MFR          2.30
  MRR        0.7300
======================================================================
```

## 🔬 Architecture

### Graph Structure
- **Nodes**: Log, Package, File, Method
- **Edges**: Hierarchical + Sequential
- **Features**: SentenceBERT (384-dim)

### Model
- **Encoder**: GGNN (Gated Graph Neural Network)
- **Phase 1**: Contrastive Learning (Node + Graph)
- **Phase 2**: Listwise Ranking

## 📝 Command Options

### data_preprocessing/preprocess.py

| Option | Description |
|--------|-------------|
| `--data_path` | Path to raw data |
| `--output_path` | Path to save .pt files |
| `--force` | Force rebuild (ignore cache) |
| `--versions` | Specific versions to process |

### training.py

| Option | Default | Description |
|--------|---------|-------------|
| `--data_path` | `./processed_data` | Path to .pt files |
| `--train_ratio` | 0.8 | Train/test split ratio |
| `--seed` | 42 | Random seed |
| `--epochs1` | 10 | Phase 1 epochs |
| `--epochs2` | 10 | Phase 2 epochs |
| `--device` | auto | cuda or cpu |

## 📁 Output Files

### Preprocessing
```
processed_data/
├── v1-12896.pt
├── v2-12893.pt
├── ...
├── embedding_cache.pkl
└── preprocessing_summary.json
```

### Training
```
results/
├── falcon_results_20260113_120000.csv
└── falcon_results_20260113_120000.json
```

## 🎯 Key Features

✅ **Modular Design**: Preprocessing và Training hoàn toàn độc lập

✅ **Caching**: Preprocessed graphs được cache để tăng tốc

✅ **Flexible**: Dễ dàng thay đổi config và parameters

✅ **Reproducible**: Random seed cho kết quả nhất quán

## 📚 Data Format

### Input: Raw Logs
```
../data_tcpdump/
├── v1-12896/fail/*.log
├── v2-12893/fail/*.log
└── ground_truth.json
```

### Intermediate: Processed Graphs
```
processed_data/
└── v*.pt  # PyTorch Geometric Data objects
```

### Output: Results
```
results/
├── *.csv  # Detailed rankings
└── *.json # Full results with metadata
```
