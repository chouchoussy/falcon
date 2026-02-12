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

```bash
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
python training.py --train_path ./train --test_path ./test --epochs1 100 --epochs2 50
```

**Output**: `results/falcon_results_*.csv` and `.json`

---

## 📈 Evaluation Metrics

- **Top-K Accuracy**: % of test cases where faulty function is in top-K
- **MFR (Mean First Rank)**: Average rank (lower is better)
- **MRR (Mean Reciprocal Rank)**: Average of 1/rank (higher is better)


## 🔬 Architecture

### Graph Structure
- **Nodes**: Log, Package, File, Method
- **Edges**: Hierarchical + Sequential
- **Features**: SentenceBERT (384-dim)

### Model
- **Encoder**: GGNN (Gated Graph Neural Network)
- **Phase 1**: Contrastive Learning (Node + Graph)
- **Phase 2**: Listwise Ranking