# FALCON Runtime Estimation

Ước tính thời gian chạy cho pipeline FALCON với LOOCV.

## 📊 Thông Số Cơ Bản

- **Số versions**: 48 (theo ground_truth.json)
- **LOOCV folds**: 48
- **Phase 1 epochs**: 10 (default)
- **Phase 2 epochs**: 10 (default)
- **Device**: CPU hoặc GPU

## ⏱️ Ước Tính Thời Gian

### Scenario 1: Lần Chạy Đầu (Chưa có Cache)

#### **CPU Mode**
```
Bước 1: Load & Preprocess Data
  - Parse logs: ~2-5 giây/version × 48 = 2-4 phút
  - Build graphs: ~10-30 giây/version × 48 = 8-24 phút
  - SentenceBERT embeddings: ~5-10 giây/version × 48 = 4-8 phút
  ──────────────────────────────────────────────────────
  Tổng Bước 1: ~14-36 phút

Bước 2: LOOCV Training (48 folds)
  Mỗi fold:
    - Phase 1: ~30-60 giây/epoch × 10 epochs × 47 train graphs = 4-8 phút
    - Phase 2: ~10-20 giây/epoch × 10 epochs × 47 train graphs = 1.5-3 phút
    - Inference: ~1-2 giây
    ──────────────────────────────────────────────────────
    Tổng mỗi fold: ~5.5-11 phút
  
  Tổng 48 folds: 5.5-11 phút × 48 = 4.4-8.8 giờ

Bước 3: Evaluation & Reporting
  - Calculate metrics: ~1 giây
  - Save CSV: ~1 giây
  ──────────────────────────────────────────────────────
  Tổng Bước 3: ~2 giây

──────────────────────────────────────────────────────
TỔNG THỜI GIAN (CPU, lần đầu): ~5-9 giờ
```

#### **GPU Mode (CUDA)**
```
Bước 1: Load & Preprocess Data
  - Parse logs: ~2-5 giây/version × 48 = 2-4 phút
  - Build graphs: ~10-30 giây/version × 48 = 8-24 phút
  - SentenceBERT embeddings: ~5-10 giây/version × 48 = 4-8 phút
  ──────────────────────────────────────────────────────
  Tổng Bước 1: ~14-36 phút

Bước 2: LOOCV Training (48 folds)
  Mỗi fold:
    - Phase 1: ~5-15 giây/epoch × 10 epochs × 47 train graphs = 0.8-2.5 phút
    - Phase 2: ~2-5 giây/epoch × 10 epochs × 47 train graphs = 0.3-0.8 phút
    - Inference: ~0.1-0.5 giây
    ──────────────────────────────────────────────────────
    Tổng mỗi fold: ~1.2-3.3 phút
  
  Tổng 48 folds: 1.2-3.3 phút × 48 = 1-2.6 giờ

Bước 3: Evaluation & Reporting
  - Calculate metrics: ~1 giây
  - Save CSV: ~1 giây
  ──────────────────────────────────────────────────────
  Tổng Bước 3: ~2 giây

──────────────────────────────────────────────────────
TỔNG THỜI GIAN (GPU, lần đầu): ~1.5-3 giờ
```

### Scenario 2: Lần Chạy Sau (Đã có Cache)

#### **CPU Mode**
```
Bước 1: Load Cached Graphs
  - Load từ .pt files: ~0.5-1 giây/version × 48 = 24-48 giây
  ──────────────────────────────────────────────────────
  Tổng Bước 1: ~0.5-1 phút

Bước 2: LOOCV Training (48 folds)
  - Giống như Scenario 1: 4.4-8.8 giờ

Bước 3: Evaluation & Reporting
  - ~2 giây

──────────────────────────────────────────────────────
TỔNG THỜI GIAN (CPU, có cache): ~4.5-9 giờ
```

#### **GPU Mode (CUDA)**
```
Bước 1: Load Cached Graphs
  - Load từ .pt files: ~0.5-1 giây/version × 48 = 24-48 giây
  ──────────────────────────────────────────────────────
  Tổng Bước 1: ~0.5-1 phút

Bước 2: LOOCV Training (48 folds)
  - Giống như Scenario 1: 1-2.6 giờ

Bước 3: Evaluation & Reporting
  - ~2 giây

──────────────────────────────────────────────────────
TỔNG THỜI GIAN (GPU, có cache): ~1.5-2.7 giờ
```

## 🚀 Tối Ưu Hóa Thời Gian

### 1. Giảm Số Epochs (Cho Testing)
```python
# Trong src/config.py
PHASE1_EPOCHS = 5   # Thay vì 10
PHASE2_EPOCHS = 5   # Thay vì 10
```
**Tiết kiệm**: ~50% thời gian training

### 2. Giảm Số Folds (Cho Quick Test)
```python
# Trong main.py, chỉ test một vài folds
all_graphs = all_graphs[:5]  # Chỉ test 5 folds
```
**Tiết kiệm**: ~90% thời gian (nếu test 5/48 folds)

### 3. Sử Dụng GPU
**Tăng tốc**: ~3-5x so với CPU

### 4. Enable Caching
**Tiết kiệm**: ~14-35 phút ở lần chạy đầu

### 5. Giảm Model Size
```python
HIDDEN_DIM = 64      # Thay vì 128
NUM_GNN_LAYERS = 2   # Thay vì 3
```
**Tiết kiệm**: ~20-30% thời gian training

## 📈 Bảng Tóm Tắt

| Scenario | Device | Cache | Thời Gian Ước Tính |
|----------|--------|-------|-------------------|
| Lần đầu | CPU | ❌ | **5-9 giờ** |
| Lần đầu | GPU | ❌ | **1.5-3 giờ** |
| Lần sau | CPU | ✅ | **4.5-9 giờ** |
| Lần sau | GPU | ✅ | **1.5-2.7 giờ** |
| Quick test (5 epochs) | GPU | ✅ | **0.5-1 giờ** |
| Quick test (5 folds) | GPU | ✅ | **~10 phút** |

## ⚡ Quick Test Mode

Để test nhanh pipeline:

```python
# src/config.py
PHASE1_EPOCHS = 2
PHASE2_EPOCHS = 2

# main.py (sau khi load all_graphs)
all_graphs = all_graphs[:3]  # Chỉ test 3 folds
```

**Thời gian**: ~5-10 phút (GPU) hoặc ~20-30 phút (CPU)

## 🔍 Monitoring Progress

Pipeline sẽ in progress cho mỗi fold:
```
Fold 1/48: Testing on v1-12896
  [Phase 1: Representation Learning]
  Epoch 1/10: Loss: 0.5234, Node: 0.4123, Graph: 0.1111
  ...
  [Phase 2: Fault Localization]
  Epoch 1/10: Loss: 0.2341
  ...
  ✓ Bug found at Rank: 3
```

## 💡 Tips

1. **Chạy qua đêm**: Với 48 folds, nên chạy qua đêm hoặc khi không dùng máy
2. **Checkpoint**: Có thể thêm checkpoint để resume nếu bị gián đoạn
3. **Parallel**: Có thể chạy song song nhiều folds (cần modify code)
4. **Cloud GPU**: Sử dụng Google Colab, AWS, etc. để tăng tốc

## 📊 Breakdown Chi Tiết

### Phase 1 (Contrastive Learning)
- **Per epoch**: ~30-60s (CPU) hoặc ~5-15s (GPU)
- **Per fold**: ~5-10 phút (CPU) hoặc ~1-2.5 phút (GPU)
- **Total 48 folds**: ~4-8 giờ (CPU) hoặc ~1-2 giờ (GPU)

### Phase 2 (Ranking)
- **Per epoch**: ~10-20s (CPU) hoặc ~2-5s (GPU)
- **Per fold**: ~1.5-3 phút (CPU) hoặc ~0.3-0.8 phút (GPU)
- **Total 48 folds**: ~1.2-2.4 giờ (CPU) hoặc ~0.25-0.65 giờ (GPU)

### Data Loading
- **First run**: ~14-36 phút (build graphs + embeddings)
- **Cached**: ~0.5-1 phút (load from .pt files)

---

**Lưu ý**: Thời gian thực tế phụ thuộc vào:
- Kích thước graphs (số nodes, edges)
- Hardware (CPU/GPU model, RAM)
- Số lượng train graphs trong mỗi fold
- Augmentation complexity

