# FALCON Training Module

Module chứa logic training 2-phase cho FALCON pipeline.

## Cấu trúc

```
training/
├── __init__.py          # Exports
├── losses.py            # Loss functions (Equations 3, 4, 5)
├── trainer.py           # FALCONTrainer (2-phase training)
└── README.md           # File này
```

## Two-Phase Training Strategy

FALCON sử dụng **2-stage training**:

```
Phase 1: Representation Learning (Contrastive)
  Input: Pass graphs (G_p) + Fail graphs (G_f)
  Goal: Learn semantic graph representations
  Loss: NodeContrastiveLoss + GraphContrastiveLoss
  Trains: Encoder + ProjectionHead
  
Phase 2: Fault Localization (Ranking)
  Input: Fail graphs (G_f) only
  Goal: Learn to rank faulty functions
  Loss: ListwiseLoss
  Trains: RankHead (+ optional fine-tune Encoder)
```

## 1. Loss Functions (`losses.py`)

### `NodeContrastiveLoss` (Equation 3)

**InfoNCE loss** cho node-level contrastive learning.

**Công thức:**
```
L_node = -log( exp(sim(z_i, z'_i) / τ) / Σ_k exp(sim(z_i, z'_k) / τ) )
```

- `z_i`: Projected embedding của node i từ graph gốc (G_f)
- `z'_i`: Projected embedding của node i từ augmented graph (G'_f)
- `τ`: Temperature parameter (default: 0.07)
- `sim`: Cosine similarity

**Mục đích:** Maximize agreement giữa corresponding nodes trong 2 views khác nhau.

**Sử dụng:**

```python
from src.training import NodeContrastiveLoss

loss_fn = NodeContrastiveLoss(temperature=0.07)

# z1, z2: projected embeddings from two views [num_nodes, embed_dim]
loss = loss_fn(z1, z2)
```

### `GraphContrastiveLoss` (Equation 4)

**Triplet Margin Loss** cho graph-level embeddings.

**Triplet structure:**
- **Anchor:** Augmented fail graph (G'_f)
- **Positive:** Original fail graph (G_f)  
- **Negative:** Pass graph (G_p)

**Công thức:**
```
L_graph = max(0, d(anchor, positive) - d(anchor, negative) + margin)
```

**Mục đích:** Fail graphs should be closer to each other than to pass graphs.

**Sử dụng:**

```python
from src.training import GraphContrastiveLoss

loss_fn = GraphContrastiveLoss(margin=1.0, distance='cosine')

# Graph embeddings: [batch_size, embed_dim]
loss = loss_fn(anchor, positive, negative)
```

### `ListwiseLoss` (Equation 5)

**Listwise ranking loss** cho fault localization.

**Công thức:**
```
L_rank = -Σ(y_i * log(softmax(scores)_i))
```

- `y_i`: Binary label (1 = faulty, 0 = normal)
- `scores`: Predicted suspiciousness scores

**Mục đích:** Rank faulty nodes higher in the list.

**Sử dụng:**

```python
from src.training import ListwiseLoss

loss_fn = ListwiseLoss()

# scores: [num_nodes], labels: [num_nodes] (1=faulty, 0=normal)
loss = loss_fn(scores, labels)
```

### `CombinedPhaseLoss`

Combines NodeContrastiveLoss + GraphContrastiveLoss cho Phase 1.

```python
from src.training import CombinedPhaseLoss

loss_fn = CombinedPhaseLoss(
    node_weight=1.0,
    graph_weight=0.5,
    temperature=0.07,
    margin=1.0
)

# Forward returns tuple: (total_loss, node_loss, graph_loss)
total_loss, node_loss, graph_loss = loss_fn(
    z_f,        # Node embeddings from G_f
    z_f_aug,    # Node embeddings from G'_f
    g_f,        # Graph embedding from G_f
    g_f_aug,    # Graph embedding from G'_f
    g_p         # Graph embedding from G_p
)
```

### `FocalLoss` (Bonus)

Focal Loss để handle class imbalance trong Phase 2.

```python
from src.training import FocalLoss

loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
loss = loss_fn(scores, labels)
```

## 2. Trainer (`trainer.py`)

### `FALCONTrainer` Class

Complete trainer quản lý toàn bộ training pipeline.

**Khởi tạo:**

```python
from src.models import FALCONModel
from src.training import FALCONTrainer
from src.dataset import augment_graph

model = FALCONModel(input_dim=384, hidden_dim=128, embedding_dim=64)

trainer = FALCONTrainer(
    model=model,
    device='cuda',
    learning_rate_phase1=1e-3,      # LR cho Phase 1
    learning_rate_phase2=1e-4,      # LR cho Phase 2
    weight_decay=1e-5,
    node_loss_weight=1.0,           # Weight cho node contrastive
    graph_loss_weight=0.5,          # Weight cho graph contrastive
    temperature=0.07,               # Temperature cho InfoNCE
    margin=1.0,                     # Margin cho triplet
    augment_fn=augment_graph,       # Augmentation function
    freeze_encoder_phase2=False     # Có freeze encoder ở Phase 2 không
)
```

### Phase 1: Representation Learning

**train_phase1_representation()**

```python
# Chuẩn bị data
fail_graphs = [...]  # List of Data objects (failed tests)
pass_graphs = [...]  # List of Data objects (passed tests)

# Train Phase 1
history = trainer.train_phase1_representation(
    fail_graphs=fail_graphs,
    pass_graphs=pass_graphs,
    epochs=10,
    drop_prob=0.2,      # Augmentation dropout
    verbose=True
)

# history['losses']: list of epoch losses
# history['node_losses']: node contrastive losses
# history['graph_losses']: graph contrastive losses
```

**Quá trình:**
1. Loop qua từng fail graph
2. Lấy 1 pass graph tương ứng (cycle nếu cần)
3. Tạo augmented fail graph: `G'_f = augment(G_f)`
4. Forward qua model:
   - `outputs_f = model(G_f)` → get `proj_emb_f`
   - `outputs_f_aug = model(G'_f)` → get `proj_emb_f_aug`
   - `outputs_p = model(G_p)` → get `proj_emb_p`
5. Pool node embeddings → graph embeddings
6. Compute loss:
   - `L_node = NodeContrastiveLoss(proj_emb_f, proj_emb_f_aug)`
   - `L_graph = GraphContrastiveLoss(graph_emb_f_aug, graph_emb_f, graph_emb_p)`
   - `L_total = node_weight * L_node + graph_weight * L_graph`
7. Backprop và update Encoder + ProjectionHead

### Phase 2: Fault Localization

**train_phase2_ranking()**

```python
# Train Phase 2
history = trainer.train_phase2_ranking(
    fail_graphs=fail_graphs,  # Only fail graphs (with labels)
    epochs=10,
    verbose=True
)

# history['losses']: list of epoch losses
```

**Quá trình:**
1. Optional: Freeze encoder (`freeze_encoder_phase2=True`)
2. Loop qua từng fail graph
3. Forward qua model:
   - `outputs = model(G_f)` → get `rank_score`
4. Compute loss:
   - `L_rank = ListwiseLoss(rank_score, G_f.y)`
5. Backprop và update RankHead (và Encoder nếu không frozen)

**Options:**
- `freeze_encoder_phase2=False`: Fine-tune encoder (better accuracy)
- `freeze_encoder_phase2=True`: Freeze encoder (faster, prevent overfitting)

### Prediction

**predict()**

```python
# Dự đoán suspiciousness scores
test_graph = Data(...)
scores = trainer.predict(test_graph)
# scores.shape = [num_nodes]

# Get top-k suspicious nodes
top_indices, top_scores = trainer.predict(test_graph, return_top_k=10)
print(f"Top-10 suspicious nodes: {top_indices}")
print(f"Scores: {top_scores}")
```

### Evaluation

**evaluate()**

```python
# Đánh giá trên test set
test_graphs = [...]  # List of Data objects with labels

results = trainer.evaluate(
    test_graphs=test_graphs,
    top_k_values=[1, 3, 5, 10],
    verbose=True
)

# results = {
#     'Top-1': 0.65,   # Top-1 accuracy
#     'Top-3': 0.82,
#     'Top-5': 0.91,
#     'Top-10': 0.97,
#     'MRR': 0.73,     # Mean Reciprocal Rank
#     'MFR': 2.3       # Mean First Rank
# }
```

**Metrics:**
- **Top-K Accuracy:** % test cases có ít nhất 1 faulty node trong top-K
- **MRR (Mean Reciprocal Rank):** Trung bình 1/rank của faulty node đầu tiên
- **MFR (Mean First Rank):** Trung bình rank của faulty node đầu tiên

### Save/Load Checkpoints

```python
# Save checkpoint
trainer.save_checkpoint(
    path='checkpoints/falcon_epoch10.pt',
    epoch=10,
    phase='phase1'
)

# Load checkpoint
trainer.load_checkpoint('checkpoints/falcon_epoch10.pt')
```

## Complete Training Pipeline

```python
from src.dataset import GraphBuilder
from src.models import FALCONModel
from src.training import FALCONTrainer
from src.dataset import augment_graph
from src.config import RAW_DATA_PATH, EMBEDDING_DIM, HIDDEN_DIM

# 1. Build graphs from logs
builder = GraphBuilder()
graphs = builder.build_graphs_from_directory(
    RAW_DATA_PATH,
    ground_truth_file=f"{RAW_DATA_PATH}/ground_truth.json"
)

# 2. Split into fail and pass graphs
fail_graphs = [g for g in graphs.values() if g.y.sum() > 0]
pass_graphs = [g for g in graphs.values() if g.y.sum() == 0]

print(f"Fail: {len(fail_graphs)}, Pass: {len(pass_graphs)}")

# 3. Create model
model = FALCONModel(
    input_dim=384,
    hidden_dim=HIDDEN_DIM,
    embedding_dim=EMBEDDING_DIM
)

# 4. Create trainer
trainer = FALCONTrainer(
    model=model,
    augment_fn=augment_graph,
    learning_rate_phase1=1e-3,
    learning_rate_phase2=1e-4
)

# 5. Phase 1: Representation Learning
print("=== PHASE 1 ===")
trainer.train_phase1_representation(
    fail_graphs=fail_graphs,
    pass_graphs=pass_graphs,
    epochs=10
)

trainer.save_checkpoint('checkpoints/phase1.pt', phase='phase1')

# 6. Phase 2: Fault Localization
print("\n=== PHASE 2 ===")
trainer.train_phase2_ranking(
    fail_graphs=fail_graphs,
    epochs=10
)

trainer.save_checkpoint('checkpoints/phase2.pt', phase='phase2')

# 7. Evaluate
print("\n=== EVALUATION ===")
results = trainer.evaluate(fail_graphs[:20])  # Test on subset

# 8. Predict on new test
test_graph = fail_graphs[0]
top_indices, top_scores = trainer.predict(test_graph, return_top_k=10)

print("\nTop-10 suspicious functions:")
for i, (idx, score) in enumerate(zip(top_indices, top_scores), 1):
    func_name = test_graph.node_names[idx] if hasattr(test_graph, 'node_names') else f"Node {idx}"
    print(f"  {i}. {func_name}: {score:.4f}")
```

## Hyperparameter Tuning

### Phase 1 Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate_phase1` | 1e-3 | Learning rate |
| `node_loss_weight` | 1.0 | Weight for node contrastive |
| `graph_loss_weight` | 0.5 | Weight for graph contrastive |
| `temperature` | 0.07 | Temperature for InfoNCE |
| `margin` | 1.0 | Margin for triplet loss |
| `drop_prob` | 0.2 | Augmentation dropout |
| `epochs` | 10 | Training epochs |

**Tips:**
- Higher `temperature` → softer distribution, easier optimization
- Higher `margin` → enforce stronger separation giữa fail và pass
- Higher `drop_prob` → more aggressive augmentation

### Phase 2 Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate_phase2` | 1e-4 | Learning rate (lower than Phase 1) |
| `freeze_encoder_phase2` | False | Freeze encoder or not |
| `epochs` | 10 | Training epochs |

**Tips:**
- Use lower LR trong Phase 2 để fine-tune carefully
- `freeze_encoder=True` nếu overfitting
- `freeze_encoder=False` cho better accuracy (nếu đủ data)

## Training History

Access training history:

```python
# Phase 1 history
print(trainer.history['phase1']['losses'])
print(trainer.history['phase1']['node_losses'])
print(trainer.history['phase1']['graph_losses'])

# Phase 2 history
print(trainer.history['phase2']['losses'])

# Plot training curves
import matplotlib.pyplot as plt

plt.plot(trainer.history['phase1']['losses'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Phase 1 Training Loss')
plt.show()
```

## Advanced Features

### Custom Augmentation Function

```python
def custom_augment(data, drop_prob):
    # Your custom augmentation logic
    return augmented_data

trainer = FALCONTrainer(
    model=model,
    augment_fn=custom_augment
)
```

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

# Enable mixed precision in trainer
# (Need to modify trainer.py to support this)
```

### Learning Rate Scheduling

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

# After creating optimizer in trainer, add scheduler
# (Need to modify trainer.py to support this)
```

## Dependencies

- `torch`: PyTorch
- `torch_geometric`: Graph operations
- `tqdm`: Progress bars

## Notes

- Phase 1 learns general representations → Phase 2 fine-tunes for ranking
- Graph augmentation chỉ áp dụng cho fail graphs trong Phase 1
- ListwiseLoss focuses on relative ordering, không cần absolute scores
- Model có thể transfer tốt sang unseen test cases sau Phase 1

