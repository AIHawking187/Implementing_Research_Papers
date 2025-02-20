# Graph Neural Networks: GCN and GAT Implementations

## 1. Graph Classification with GCN

### Overview
- **Task**: Classify entire graphs into different categories
- **Dataset**: Synthetic dataset with two graph types:
  - Cycle-like graphs (Class 0)
  - Star-like graphs (Class 1)

### GCN Architecture
```
Input Graph → GCN Layer 1 → ReLU → Dropout → 
GCN Layer 2 → ReLU → Dropout → 
GCN Layer 3 → ReLU → 
Global Mean Pooling → Linear → Output
```

### Key Components
- **Graph Convolution**: Aggregates node features based on neighborhood structure
- **Global Pooling**: Converts node-level features to graph-level representation
- **Multiple GCN Layers**: Allow for capturing increasingly complex graph patterns

### Performance Metrics
- Training/validation/test accuracy curves
- Confusion matrix for classification results
- Detailed classification report

## 2. Node Classification with GAT

### Overview
- **Task**: Classify individual nodes in a graph
- **Dataset**: Synthetic community graph with:
  - Nodes belonging to different communities
  - Feature vectors with class-specific patterns
  - Higher intra-community connectivity

### GAT Architecture
```
Input Nodes → GAT Layer 1 (8 heads) → ELU → Dropout → 
GAT Layer 2 (1 head) → LogSoftmax → Output
```

### Key Features
- **Attention Mechanism**: Learns to weight neighbor importance
- **Multi-head Attention**: Multiple independent attention computations
- **Node-level Classification**: Direct prediction for each node

### Advantages of Each Architecture

**GCN (Graph Classification)**
- Simple and effective for graph-level tasks
- Efficient neighborhood aggregation
- Good at capturing global structure

**GAT (Node Classification)**
- Dynamic edge weights through attention
- Can focus on most relevant neighbors
- Better handling of heterogeneous node neighborhoods

## Implementation Highlights

### Data Generation
- Custom synthetic datasets for clear pattern recognition
- Controlled difficulty through parameter tuning
- Built-in train/val/test splits

### Visualization
- Network structure visualization
- Training progress monitoring
- Prediction analysis
- Confusion matrices

### Training Features
- Early stopping
- Learning rate optimization
- Dropout regularization
- Performance metrics tracking

## Conclusion

The implementations demonstrate two fundamental tasks in graph learning:
1. **Graph Classification**: Using GCN to learn patterns in entire graph structures
2. **Node Classification**: Using GAT to learn node-level patterns through attention mechanisms

Both implementations achieve good performance on their respective synthetic datasets, showing the effectiveness of graph neural networks for different types of graph learning tasks.