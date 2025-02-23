# Comprehensive Guide to Transformer Architectures

## Table of Contents
1. [The Transformer Architecture](#the-transformer-architecture)
2. [BERT](#bert)
3. [Machine Translation with Transformers](#machine-translation-with-transformers)
4. [Vision Transformers](#vision-transformers)

## The Transformer Architecture

### Core Concepts

The Transformer architecture, introduced in the "Attention Is All You Need" paper (Vaswanyi et al., 2017), revolutionized natural language processing and later computer vision. Its key innovation was replacing recurrent and convolutional neural networks with self-attention mechanisms.

### Key Components

#### 1. Self-Attention Mechanism
- **Query, Key, Value (QKV) Matrices**: Each input token is transformed into three vectors:
  - Query (Q): What the current token is looking for
  - Key (K): What the token offers to others
  - Value (V): The actual information carried by the token
- **Attention Formula**: Attention(Q,K,V) = softmax(QK^T/√d_k)V
  - √d_k scaling prevents extremely small gradients

#### 2. Multi-Head Attention
- Runs multiple attention operations in parallel
- Each head can focus on different aspects of the input
- Outputs are concatenated and linearly transformed
- Formula: MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O

#### 3. Position Encodings
- Adds position information to token embeddings
- Can be learned or fixed sinusoidal encodings
- Essential because self-attention has no inherent order sensitivity

#### 4. Feed-Forward Networks
- Two linear transformations with ReLU activation
- Applied to each position separately
- Formula: FFN(x) = max(0, xW_1 + b_1)W_2 + b_2

### Architecture Structure

```
Input
  ↓
Embedding + Positional Encoding
  ↓
Encoder Stack (x N)
  - Multi-Head Self-Attention
  - Add & Norm
  - Feed Forward
  - Add & Norm
  ↓
Decoder Stack (x N)
  - Masked Multi-Head Self-Attention
  - Multi-Head Cross-Attention
  - Feed Forward
  - Add & Norm
  ↓
Linear + Softmax
  ↓
Output
```

## BERT

### Overview
BERT (Bidirectional Encoder Representations from Transformers) revolutionized NLP by introducing bidirectional pre-training and achieving state-of-the-art results across multiple tasks.

### Key Innovations

#### 1. Masked Language Modeling (MLM)
- Randomly masks 15% of input tokens
- Model must predict masked tokens
- Forces bidirectional understanding
- Example:
  ```
  Input: "The [MASK] jumped over the [MASK]."
  Task: Predict "cat" and "fence"
  ```

#### 2. Next Sentence Prediction (NSP)
- Model predicts if two sentences are consecutive
- Helps understand sentence relationships
- Input format:
  ```
  [CLS] Sentence A [SEP] Sentence B [SEP]
  ```

### Architecture Details
- BERT-base: 12 layers, 768 hidden size, 12 attention heads
- BERT-large: 24 layers, 1024 hidden size, 16 attention heads
- Uses learned positional embeddings
- Special tokens: [CLS], [SEP], [MASK]

### Fine-tuning Approaches
1. **Sequence Classification**
   - Use [CLS] token representation
   - Add classification head
   
2. **Token Classification**
   - Use token-level representations
   - Common for NER, POS tagging

3. **Question Answering**
   - Predict start and end positions
   - Uses cross-attention between question and context

## Machine Translation with Transformers

### Implementation Details

#### 1. Data Processing
```python
class TranslationDataset:
    def __init__(self):
        # Tokenization
        # Vocabulary creation
        # Add special tokens ([PAD], [START], [END])
```

#### 2. Model Architecture
- Encoder-decoder structure
- Source language → Encoder → Decoder → Target language
- Cross-attention between encoder and decoder

#### 3. Training Process
```python
# Training loop
for epoch in range(epochs):
    for src, tgt in dataloader:
        # Teacher forcing
        # Calculate loss
        # Update parameters
```

### Key Considerations
1. **Vocabulary Management**
   - Subword tokenization (BPE, WordPiece)
   - Handling unknown tokens
   
2. **Beam Search**
   - Maintains top-k hypotheses
   - Improves translation quality
   
3. **Evaluation Metrics**
   - BLEU score
   - ROUGE score
   - Human evaluation

## Vision Transformers

### Core Concepts

#### 1. Image Patching
- Split image into fixed-size patches
- Linear projection of flattened patches
- Similar to word embeddings in NLP

#### 2. Architecture Modifications
```python
class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim):
        # Convert image patches to embeddings
        # Add position embeddings
        # Add [CLS] token
```

### Implementation Details

#### 1. Patch Processing
- Image size: H × W
- Patch size: P × P
- Number of patches: (H×W)/(P×P)
- Embedding dimension: D

#### 2. Position Embeddings
- Learned positional embeddings
- Added to patch embeddings
- Include [CLS] token position

#### 3. Transformer Blocks
```python
class TransformerBlock(nn.Module):
    def __init__(self):
        # Multi-head self-attention
        # MLP block
        # Layer normalization
```

### Training Considerations

1. **Data Augmentation**
   - Random cropping
   - Random horizontal flipping
   - Color jittering
   
2. **Regularization**
   - Dropout
   - Stochastic depth
   - Weight decay

3. **Optimization**
   - AdamW optimizer
   - Cosine learning rate schedule
   - Gradient clipping

### Applications

1. **Image Classification**
   - Use [CLS] token
   - Standard cross-entropy loss
   
2. **Object Detection**
   - Add detection head
   - Predict bounding boxes
   
3. **Semantic Segmentation**
   - Keep spatial information
   - Patch-level predictions

## Conclusion

Transformers have become the foundation of modern deep learning, showing remarkable performance across:
- Natural Language Processing
- Computer Vision
- Multi-modal Tasks
- Speech Recognition

Their success lies in:
- Parallel processing capability
- Ability to capture long-range dependencies
- Scalability to large datasets
- Adaptability to various domains

The field continues to evolve with innovations like:
- Efficient attention mechanisms
- Sparse transformers
- Domain-specific architectures
- Hybrid approaches

Understanding these architectures and their implementations is crucial for modern deep learning practitioners.