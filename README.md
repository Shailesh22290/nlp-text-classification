# NLP Text Classification with NumPy

This project implements **RNN, LSTM, and Transformer** architectures from scratch using only **NumPy** for sentiment analysis on COVID-19 tweets.

## Project Overview

A complete text classification pipeline built for Advanced NLP coursework, featuring custom neural network implementations without any deep learning frameworks.

### Dataset
- **Source**: Corona NLP Dataset
- **Task**: 5-class sentiment classification
- **Classes**: Extremely Negative, Negative, Neutral, Positive, Extremely Positive
- **Domain**: COVID-19 related tweets

## Architecture Implementations

All models are implemented from scratch using only NumPy:

### 1. Simple RNN
- Vanilla recurrent neural network
- Custom forward and backward propagation
- **Performance**: Macro F1-Score: 0.3818

### 2. LSTM (Long Short-Term Memory)
- Complete LSTM cell implementation with forget, input, and output gates
- Handles vanishing gradient problem effectively
- **Performance**: Macro F1-Score: 0.6923 ⭐ (Best performing)

### 3. Transformer Encoder
- Multi-head self-attention mechanism
- Position encoding and layer normalization
- Feed-forward networks with residual connections
- **Performance**: Macro F1-Score: 0.5896

## 📊 Performance Comparison

| Model | Macro F1 | Inference Time | Best Class Performance |
|-------|----------|----------------|------------------------|
| **LSTM** | **0.6923** | 2.79s | Neutral (0.774 F1) |
| **Transformer** | 0.5896 | 20.01s | Neutral (0.680 F1) |
| **RNN** | 0.3818 | 1.78s | Neutral (0.542 F1) |



### Run  Inference in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GpdR0MiHlYRSqE5DMtbcvlZuu8xfprWf?usp=sharing)



##  Results

### LSTM (Best Performance)
```
Macro F1: 0.6923
Class Performance:
- Extremely Negative: P=0.728, R=0.650, F1=0.687
- Extremely Positive: P=0.751, R=0.725, F1=0.737
- Negative: P=0.607, R=0.678, F1=0.640
- Neutral: P=0.792, R=0.756, F1=0.774
- Positive: P=0.627, R=0.620, F1=0.623
```

### Transformer
```
Macro F1: 0.5896
Class Performance:
- Extremely Negative: P=0.652, R=0.551, F1=0.597
- Extremely Positive: P=0.732, R=0.447, F1=0.555
- Negative: P=0.537, R=0.583, F1=0.559
- Neutral: P=0.645, R=0.721, F1=0.680
- Positive: P=0.515, R=0.604, F1=0.556
```

### Simple RNN
```
Macro F1: 0.3818
Class Performance:
- Extremely Negative: P=0.244, R=0.177, F1=0.205
- Extremely Positive: P=0.451, R=0.477, F1=0.464
- Negative: P=0.350, R=0.317, F1=0.332
- Neutral: P=0.515, R=0.572, F1=0.542
- Positive: P=0.339, R=0.395, F1=0.365
```
