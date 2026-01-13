# FALCON Models Module

Module chứa kiến trúc mạng Neural Network cho FALCON pipeline.

## Cấu trúc

```
models/
├── __init__.py          # FALCONModel & exports
├── encoder.py           # GGNN Graph Encoder
├── heads.py             # Projection & Rank Heads
└── README.md           # File này
```

## Kiến trúc tổng quan

FALCON model bao gồm 3 components chính:

```
Input Graph (x, edge_index)
         ↓
    [GraphEncoder (GGNN)]
         ↓
    Node Embeddings
         ↓
    ┌─────────────┴─────────────┐
    ↓                           ↓
[ProjectionHead]          [RankHead]
(Phase 1: Contrastive)    (Phase 2: Ranking)
    ↓                           ↓
Projected Embeddings      Suspiciousness Scores
```

## 1. Graph Encoder (`encoder.py`)

### `GraphEncoder`

Sử dụng **Gated Graph Neural Network (GGNN)** để encode graph structure.

**Kiến trúc:**
- Input projection layer
- Multiple GatedGraphConv layers (message passing)
- Layer normalization
- Residual connections
- Output projection

**Sử dụng:**

```python
from src.models import GraphEncoder

encoder = GraphEncoder(
    input_dim=384,        # SentenceBERT embedding dim
    hidden_dim=128,       # Hidden layer dim
    output_dim=64,        # Output embedding dim
    num_layers=3,         # Number of GGNN layers
    num_steps=5,          # Recurrent steps per layer
    dropout=0.1
)

# Forward pass
node_embeddings = encoder(x, edge_index)
# node_embeddings.shape = [num_nodes, output_dim]

# Get graph-level embedding
graph_emb = encoder.get_graph_embedding(x, edge_index, pooling='mean')
```

**Parameters:**
- `input_dim`: Dimension của input features (384 cho SentenceBERT)
- `hidden_dim`: Dimension của hidden layers (128)
- `output_dim`: Dimension của output embeddings (64)
- `num_layers`: Số lượng GGNN layers (3)
- `num_steps`: Số bước recurrent trong mỗi layer (5)
- `dropout`: Dropout rate (0.1)

### `MultiScaleGraphEncoder`

Enhanced version với multi-scale feature extraction.

**Đặc điểm:**
- Kết hợp embeddings từ tất cả các layers
- Sử dụng attention weights để fusion
- Capture thông tin ở nhiều scales khác nhau

```python
from src.models import MultiScaleGraphEncoder

encoder = MultiScaleGraphEncoder(
    input_dim=384,
    hidden_dim=128,
    output_dim=64,
    num_layers=3
)

node_emb = encoder(x, edge_index)
```

## 2. Prediction Heads (`heads.py`)

### `ProjectionHead` (Phase 1)

MLP head cho **contrastive learning**.

**Kiến trúc:** Linear → ReLU → Linear → L2 Normalization

**Sử dụng:**

```python
from src.models import ProjectionHead

proj_head = ProjectionHead(
    input_dim=64,         # Node embedding dim
    hidden_dim=256,       # Hidden layer dim
    output_dim=64,        # Projection dim
    dropout=0.1
)

# Project embeddings
proj_emb = proj_head(node_embeddings)
# proj_emb.shape = [num_nodes, output_dim]
# proj_emb is L2-normalized for contrastive loss
```

**Đặc điểm:**
- Output được L2-normalized (norm = 1)
- Phục vụ cho contrastive learning trong Phase 1
- Tương tự projection head trong SimCLR

### `RankHead` (Phase 2)

Linear head cho **fault localization**.

**Kiến trúc:** 
- Simple: Linear layer
- Deep: Linear → ReLU → Dropout → Linear

**Sử dụng:**

```python
from src.models import RankHead

# Simple version (single layer)
rank_head = RankHead(input_dim=64)

# Deep version (two layers)
rank_head = RankHead(
    input_dim=64,
    hidden_dim=64,
    dropout=0.1,
    use_sigmoid=False  # Use raw scores for ranking
)

# Predict suspiciousness scores
scores = rank_head(node_embeddings)
# scores.shape = [num_nodes]
# Higher score = more suspicious
```

**Output:**
- Scalar score cho mỗi node
- Score cao → Khả năng lỗi cao
- Sử dụng để ranking functions theo mức độ nghi ngờ

### `DualHead`

Combined head cho multi-task learning.

```python
from src.models import DualHead

dual_head = DualHead(
    input_dim=64,
    proj_hidden_dim=256,
    proj_output_dim=64,
    rank_hidden_dim=64
)

outputs = dual_head(node_embeddings, return_projection=True, return_rank=True)
# outputs['projection'].shape = [num_nodes, 64]
# outputs['rank'].shape = [num_nodes]
```

### `AttentionRankHead`

Enhanced ranking head với attention mechanism.

```python
from src.models import AttentionRankHead

attn_rank_head = AttentionRankHead(
    input_dim=64,
    hidden_dim=64,
    num_heads=4
)

scores = attn_rank_head(node_embeddings)
```

## 3. Complete FALCON Model (`__init__.py`)

### `FALCONModel`

Complete model tích hợp tất cả components.

**Sử dụng:**

```python
from src.models import FALCONModel

model = FALCONModel(
    input_dim=384,           # SentenceBERT features
    hidden_dim=128,          # GNN hidden dim
    embedding_dim=64,        # Node embedding dim
    proj_hidden_dim=256,     # Projection head hidden
    proj_output_dim=64,      # Projection output
    rank_hidden_dim=64,      # Rank head hidden (None = single layer)
    num_gnn_layers=3,        # Number of GGNN layers
    num_gnn_steps=5,         # Steps per GGNN layer
    dropout=0.1
)

print(f"Total parameters: {model.get_num_params():,}")
```

**Forward Pass:**

```python
# Full forward (all components)
outputs = model(x, edge_index, return_projection=True, return_rank=True)
# outputs['node_emb'].shape = [num_nodes, 64]
# outputs['proj_emb'].shape = [num_nodes, 64]
# outputs['rank_score'].shape = [num_nodes]

# Phase 1 only (Contrastive Learning)
outputs = model(x, edge_index, return_projection=True, return_rank=False)

# Phase 2 only (Fault Localization)
outputs = model(x, edge_index, return_projection=False, return_rank=True)
```

**Convenience Methods:**

```python
# Get node embeddings only
node_emb = model.encode(x, edge_index)

# Get projected embeddings (Phase 1)
proj_emb = model.get_projection(x, edge_index)

# Predict suspiciousness scores (Phase 2)
scores = model.predict_rank(x, edge_index)

# Freeze/unfreeze components
model.freeze_encoder()        # Freeze encoder for Phase 2
model.unfreeze_encoder()      # Unfreeze for fine-tuning
model.freeze_projection_head()
```

### `FALCONModelV2`

Enhanced version với attention-based ranking.

```python
from src.models import FALCONModelV2

model_v2 = FALCONModelV2(
    input_dim=384,
    hidden_dim=128,
    embedding_dim=64,
    use_attention_rank=True  # Use AttentionRankHead
)
```

### Factory Function

```python
from src.models import build_falcon_model

# Build default V1
model = build_falcon_model(
    input_dim=384,
    hidden_dim=128,
    embedding_dim=64,
    model_version='v1'
)

# Build V2 with attention
model_v2 = build_falcon_model(
    input_dim=384,
    hidden_dim=128,
    embedding_dim=64,
    model_version='v2',
    use_attention_rank=True
)
```

## Two-Phase Training

FALCON được train theo 2 phases:

### Phase 1: Contrastive Learning

```python
# Use projection head
model.train()
outputs = model(x, edge_index, return_projection=True, return_rank=False)
proj_emb = outputs['proj_emb']

# Compute contrastive loss (InfoNCE)
# ... (implemented in training module)
```

**Mục đích:** Học representations tốt từ augmented graphs

### Phase 2: Fault Localization

```python
# Optionally freeze encoder
# model.freeze_encoder()

model.train()
outputs = model(x, edge_index, return_projection=False, return_rank=True)
rank_scores = outputs['rank_score']

# Compute ranking loss
# ... (implemented in training module)
```

**Mục đích:** Fine-tune để predict suspiciousness scores

## Integration với Dataset Module

```python
from src.dataset import GraphBuilder
from src.models import FALCONModel
from src.config import RAW_DATA_PATH, EMBEDDING_DIM, HIDDEN_DIM

# Build graph
builder = GraphBuilder()
data = builder.build_graph_from_log_file("test.log")

# Create model
model = FALCONModel(
    input_dim=data.x.shape[1],  # 384
    hidden_dim=HIDDEN_DIM,       # 128
    embedding_dim=EMBEDDING_DIM  # 64
)

# Forward pass
outputs = model(data.x, data.edge_index)

# Get top-k suspicious functions
scores = outputs['rank_score']
top_k_indices = torch.topk(scores, k=10).indices
```

## Model Architecture Summary

```
FALCONModel(
  (encoder): GraphEncoder(
    (input_proj): Linear(384 → 128)
    (ggnn_layers): ModuleList(
      3 x GatedGraphConv(128, num_layers=5)
    )
    (layer_norms): ModuleList(3 x LayerNorm)
    (output_proj): Linear(128 → 64)
  )
  (projection_head): ProjectionHead(
    (fc1): Linear(64 → 256)
    (fc2): Linear(256 → 64)
  )
  (rank_head): RankHead(
    (fc1): Linear(64 → 64)
    (fc2): Linear(64 → 1)
  )
)

Total parameters: ~150K (depends on config)
```

## Dependencies

- `torch`: PyTorch
- `torch_geometric`: PyTorch Geometric (GatedGraphConv)

## Notes

- GGNN (Gated Graph Neural Network) được chọn theo paper FALCON
- Projection head L2-normalize output cho contrastive learning
- Rank head output raw scores (không sigmoid) để preserve ranking order
- Model có thể freeze/unfreeze từng component cho training linh hoạt
- Multi-scale encoder capture thông tin ở nhiều levels

