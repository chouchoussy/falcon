# FALCON Dataset Module

Module xử lý dữ liệu cho FALCON pipeline, bao gồm parsing log files, xây dựng graph, và augmentation.

## Cấu trúc

```
dataset/
├── __init__.py          # Exports chính
├── log_parser.py        # Parse log files để trích xuất function calls
├── graph_builder.py     # Xây dựng PyG Data objects từ logs
├── augmentation.py      # Adaptive Graph Augmentation (AGA)
└── README.md           # File này
```

## 1. Log Parser (`log_parser.py`)

### Chức năng
- Parse execution trace logs để trích xuất function call events
- Sử dụng regex pattern: `r'Call\s+([^\s]+)\s+.*?\(([\w\-\.]+):\d+\)'`
- Tạo unique ID cho mỗi function: `filename::funcname`

### Sử dụng

```python
from src.dataset import parse_log_file, LogEvent

# Parse một log file
events = parse_log_file("path/to/logfile.log")

# Mỗi event chứa:
for event in events:
    print(event.filename)    # Tên file (e.g., 'tcpdump.c')
    print(event.funcname)    # Tên hàm (e.g., 'main')
    print(event.unique_id)   # ID duy nhất (e.g., 'tcpdump.c::main')
    print(event.thread_id)   # Thread ID
```

### API chính

- `parse_log_file(filepath)`: Parse một file log
- `parse_log_directory(directory)`: Parse tất cả .log files trong thư mục
- `get_unique_functions(events)`: Lấy danh sách unique functions
- `get_call_sequence(events)`: Lấy chuỗi thực thi

## 2. Graph Builder (`graph_builder.py`)

### Chức năng
- Chuyển đổi log events thành PyTorch Geometric Data objects
- Tạo 4 loại nodes: Log, Package, File, Method
- Tạo 2 loại edges:
  - **Hierarchical**: Log → Package → File → Method
  - **Sequential**: Method_i → Method_{i+1} (trong cùng thread)
- Sử dụng SentenceBERT để encode node features
- Cache embeddings để tăng tốc

### Sử dụng

```python
from src.dataset import GraphBuilder, load_ground_truth

# Khởi tạo builder
builder = GraphBuilder(
    embedding_model_name='all-MiniLM-L6-v2',
    cache_dir='./processed_data'
)

# Build graph từ một log file
data = builder.build_graph_from_log_file("test.log")

# Build tất cả graphs từ thư mục
ground_truth = load_ground_truth("../data_tcpdump/ground_truth.json")
graphs = builder.build_graphs_from_directory(
    data_dir="../data_tcpdump",
    ground_truth_file="../data_tcpdump/ground_truth.json",
    save_processed=True  # Lưu cache .pt files
)
```

### Graph Structure

- `data.x`: Node features (SentenceBERT embeddings, shape: [num_nodes, 384])
- `data.edge_index`: Edge connections (shape: [2, num_edges])
- `data.y`: Labels (0=normal, 1=faulty)
- `data.node_types`: Node type IDs (0=Log, 1=Package, 2=File, 3=Method)
- `data.node_names`: List of node names

### Cache System

- Embeddings được cache trong `processed_data/embedding_cache.pkl`
- Processed graphs được lưu trong `processed_data/{test_id}.pt`
- Tự động load từ cache nếu có sẵn

## 3. Augmentation (`augmentation.py`)

### Chức năng
Implement **Adaptive Graph Augmentation (AGA)** theo Section III.B.2 của bài báo ICSE'25:

- **Transitive Analysis**: Tìm fault-related subgraph bằng transitive closure
- **Edge Centrality**: Tính importance của edges dựa trên node degrees
- **Adaptive Dropping**: Drop edges với xác suất adaptive:
  ```
  p_drop(e) = α × (1 - centrality(e) / max_centrality)
  ```
  chỉ áp dụng cho fault-unrelated edges

### Sử dụng

```python
from src.dataset import augment_graph, augment_graph_pair

# Augment một graph
augmented_data = augment_graph(
    data,
    drop_prob=0.2,  # α parameter
    use_transitive=True
)

# Tạo 2 augmented views cho contrastive learning
view1, view2 = augment_graph_pair(data, drop_prob=0.2)

# Augment một batch
from src.dataset import batch_augment_graphs
augmented_batch = batch_augment_graphs(data_list, drop_prob=0.2)
```

### API chính

- `identify_fault_related_subgraph(data, faulty_nodes)`: Tìm nodes liên quan đến lỗi
- `calculate_edge_centrality(data)`: Tính edge centrality scores
- `augment_graph(data, drop_prob)`: Augment một graph
- `augment_graph_pair(data)`: Tạo 2 augmented views

## Workflow tổng quát

```python
from src.dataset import GraphBuilder, augment_graph_pair
from src.config import RAW_DATA_PATH, PROCESSED_PATH, GROUND_TRUTH_FILE

# 1. Build graphs
builder = GraphBuilder(cache_dir=PROCESSED_PATH)
graphs = builder.build_graphs_from_directory(
    data_dir=RAW_DATA_PATH,
    ground_truth_file=f"{RAW_DATA_PATH}/{GROUND_TRUTH_FILE}"
)

# 2. Lấy một graph
test_id = list(graphs.keys())[0]
data = graphs[test_id]

print(f"Graph: {data.num_nodes} nodes, {data.num_edges} edges")
print(f"Faulty nodes: {data.y.sum().item()}")

# 3. Augment cho contrastive learning
view1, view2 = augment_graph_pair(data, drop_prob=0.2)

print(f"View 1: {view1.num_edges} edges")
print(f"View 2: {view2.num_edges} edges")
```

## Dependencies

- `torch`: PyTorch
- `torch_geometric`: PyTorch Geometric
- `sentence-transformers`: SentenceBERT embeddings
- `networkx`: Graph analysis (cho transitive closure)
- `numpy`: Numerical operations

## Notes

- SentenceBERT model `all-MiniLM-L6-v2` tạo embeddings 384 chiều
- Cache được lưu tự động để tăng tốc độ xử lý
- Transitive closure được tính bằng NetworkX cho hiệu suất tốt
- Edge dropping sử dụng random seed, có thể set `np.random.seed()` để reproducibility

