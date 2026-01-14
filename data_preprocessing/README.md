# FALCON Data Preprocessing Module

Independent module for preprocessing raw execution logs into graph representations.

## 📦 Overview

This module converts raw execution logs into PyTorch Geometric `Data` objects suitable for training FALCON models. It is **completely independent** from the training pipeline.

## 🔧 Components

### 1. `config.py`
Configuration for preprocessing:
- Paths to raw data and output directory
- SentenceBERT model configuration
- Graph structure parameters

### 2. `log_parser.py`
Parse execution logs:
- Extract function calls using regex
- Create unique identifiers for functions
- Build call sequences

### 3. `graph_builder.py`
Build PyTorch Geometric graphs:
- Create hierarchical structure (Log → Package → File → Method)
- Generate sequential edges (call order)
- Embed node features using SentenceBERT
- Label faulty nodes from ground truth

### 4. `preprocess.py`
Main preprocessing script:
- Load raw logs from `data_tcpdump`
- Build graphs for all versions
- Cache results as `.pt` files
- Generate preprocessing summary

## 🚀 Usage

### Option 1: Python Script (Recommended for Local)

```bash
cd data_preprocessing

# Process all versions
python preprocess.py

# Force rebuild (ignore cache)
python preprocess.py --force

# Process specific versions
python preprocess.py --versions v1-12896 v2-12893

# Custom paths
python preprocess.py --data_path ../../custom_data --output_path ../custom_output
```

### Option 2: Jupyter Notebook (For Colab/Kaggle)

Use `preprocess.ipynb` for interactive preprocessing:

```bash
# Local
jupyter notebook preprocess.ipynb

# Or upload to Google Colab / Kaggle
```

**✅ Note**: Notebook implementation đã được đồng bộ 100% với Python scripts (updated 2026-01-14).  
Output từ notebook **GIỐNG HỆT** output từ `preprocess.py`. Xem `NOTEBOOK_FIXES.md` để biết chi tiết.

## 📊 Input/Output

### Input
```
../../data_tcpdump/
├── v1-12896/
│   └── fail/*.log
├── v2-12893/
│   └── fail/*.log
└── ground_truth.json
```

### Output
```
../processed_data/
├── v1-12896.pt
├── v2-12893.pt
├── ...
├── embedding_cache.pkl
└── preprocessing_summary.json
```

## 🎯 Graph Structure

### Nodes
- **Log**: Root node for each execution
- **Package**: Groups of related files
- **File**: Source code files
- **Method**: Individual functions

### Edges
- **Hierarchical**: Log → Package → File → Method
- **Sequential**: Method A → Method B (call order)

### Features
- Node embeddings: SentenceBERT (384-dim)
- Node types: One-hot encoded
- Labels: Binary (1 = faulty, 0 = normal)

## ⏱️ Performance

| Dataset Size | First Run | With Cache |
|--------------|-----------|------------|
| 48 versions  | ~15-35 min | ~1 min |

## 🔍 Example Output

```
======================================================================
FALCON DATA PREPROCESSING
======================================================================
Start Time: 2026-01-13 12:00:00
Data Path: ../../data_tcpdump
Output Path: ../processed_data
Force Rebuild: False
======================================================================

Processing versions: 100%|██████████| 48/48 [15:30<00:00, 19.4s/it]

Successfully processed: 48/48 versions
Failed: 0
Total time: 15:30

Results saved to: ../processed_data/
```

## 🛠️ Configuration

Edit `config.py` to customize:

```python
# Paths
RAW_DATA_PATH = "../../data_tcpdump"
OUTPUT_PATH = "../processed_data"

# SentenceBERT
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Node types
NODE_TYPE_LOG = 0
NODE_TYPE_PACKAGE = 1
NODE_TYPE_FILE = 2
NODE_TYPE_METHOD = 3
```

## 📝 Dependencies

```
torch
torch-geometric
sentence-transformers
networkx
```

## ✅ Features

- ✅ **Caching**: Skip already processed versions
- ✅ **Progress tracking**: Real-time progress bar
- ✅ **Error handling**: Continue on failures
- ✅ **Logging**: Detailed summary and statistics
- ✅ **Flexible**: Easy to customize paths and parameters

