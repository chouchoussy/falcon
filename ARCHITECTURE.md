# FALCON Architecture Overview

## 📂 Project Structure

```
Falcon/
│
├── data_preprocessing/          # 📦 PREPROCESSING MODULE (Independent)
│   ├── config.py               # Preprocessing configuration
│   ├── log_parser.py           # Parse execution logs
│   ├── graph_builder.py        # Build PyG Data objects
│   ├── preprocess.py           # Main preprocessing script
│   └── README.md               # Module documentation
│
├── processed_data/             # 💾 PROCESSED GRAPHS
│   ├── *.pt                    # PyTorch Geometric Data objects
│   ├── embedding_cache.pkl     # Cached SentenceBERT embeddings
│   └── preprocessing_summary.json
│
├── src/                        # 🧠 TRAINING MODULE
│   ├── config.py               # Training configuration
│   │
│   ├── models/                 # Neural network architectures
│   │   ├── encoder.py          # GGNN (Gated Graph Neural Network)
│   │   ├── heads.py            # ProjectionHead & RankHead
│   │   └── __init__.py
│   │
│   ├── training/               # Training logic
│   │   ├── losses.py           # NodeContrastive, GraphContrastive, Listwise
│   │   ├── trainer.py          # Two-phase trainer
│   │   ├── augmentation.py     # Adaptive Graph Augmentation (AGA)
│   │   └── __init__.py
│   │
│   └── utils/                  # Utilities
│       ├── metrics.py          # Top-K, MFR, MRR
│       └── __init__.py
│
├── training.py                 # 🚀 Main training script
├── results/                    # 📊 Training results
│   ├── *.csv                   # Detailed rankings
│   └── *.json                  # Full results + metadata
│
├── requirements.txt            # Dependencies
├── README.md                   # Main documentation
└── ARCHITECTURE.md             # This file
```

## 🔄 Data Flow

```
Raw Logs                Graph Objects           Trained Model
   ↓                         ↓                       ↓
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ data_tcpdump │  →   │ processed_   │  →   │   results/   │
│   *.log      │      │    data/     │      │    *.csv     │
│ ground_truth │      │    *.pt      │      │    *.json    │
└──────────────┘      └──────────────┘      └──────────────┘
       ↓                     ↓                      ↓
  preprocess.py          training.py           evaluate
```

## 🎯 Module Independence

### 1. Data Preprocessing (`data_preprocessing/`)

**Purpose**: Convert raw logs → PyG graphs

**Input**: 
- `../../data_tcpdump/v*-*/fail/*.log`
- `../../data_tcpdump/ground_truth.json`

**Output**:
- `../processed_data/*.pt`

**Run**:
```bash
cd data_preprocessing
python preprocess.py
```

**Dependencies**:
- torch
- torch-geometric
- sentence-transformers

---

### 2. Training (`src/` + `training.py`)

**Purpose**: Train FALCON model and evaluate

**Input**:
- `./processed_data/*.pt`

**Output**:
- `./results/*.csv`
- `./results/*.json`

**Run**:
```bash
python training.py
```

**Dependencies**:
- torch
- torch-geometric
- scikit-learn

---

## 🧩 Component Details

### Preprocessing Components

| File | Purpose |
|------|---------|
| `log_parser.py` | Parse logs using regex → extract function calls |
| `graph_builder.py` | Build graph: nodes (Log/Package/File/Method), edges (hierarchical + sequential) |
| `config.py` | Paths, SentenceBERT model, node types |
| `preprocess.py` | Main script: load → parse → build → save |

### Training Components

| File | Purpose |
|------|---------|
| `models/encoder.py` | GGNN encoder for graph representation |
| `models/heads.py` | ProjectionHead (contrastive) + RankHead (ranking) |
| `training/losses.py` | NodeContrastive, GraphContrastive, Listwise losses |
| `training/trainer.py` | Two-phase training logic |
| `training/augmentation.py` | Adaptive Graph Augmentation (AGA) |
| `utils/metrics.py` | Top-K, MFR, MRR evaluation |

---

## 🔬 Two-Phase Training

### Phase 1: Representation Learning
- **Goal**: Learn semantic graph representations
- **Data**: Fail graphs + Augmented graphs
- **Loss**: NodeContrastive + GraphContrastive
- **Update**: Encoder + ProjectionHead

### Phase 2: Fault Localization
- **Goal**: Rank faulty functions
- **Data**: Fail graphs only
- **Loss**: Listwise ranking loss
- **Update**: RankHead + fine-tune Encoder

---

## 📊 Evaluation Pipeline

```
Test Graph → Encoder → RankHead → Scores
                                      ↓
                              Sort descending
                                      ↓
                          Find rank of faulty node
                                      ↓
                        Calculate Top-K, MFR, MRR
```

---

## 🚀 Quick Commands

```bash
# Full pipeline
cd Falcon
source venv/bin/activate

# Step 1: Preprocess
cd data_preprocessing
python preprocess.py
cd ..

# Step 2: Train
python training.py

# Results
cat results/falcon_results_*.csv
```

---

## ✅ Key Features

- ✅ **Modular**: Preprocessing and training are independent
- ✅ **Cached**: Graphs and embeddings are cached
- ✅ **Flexible**: Easy to modify configs
- ✅ **Scalable**: Can handle large datasets
- ✅ **Reproducible**: Fixed random seeds

