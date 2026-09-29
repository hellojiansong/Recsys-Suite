# RecSys Benchmark Suite: From Classical Collaborative Filtering to Generative Retrieval

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)

A unified, reproducible benchmark exploring the evolution of modern recommendation paradigms:

- **Neural Collaborative Filtering (NCF)**
- **Self-Attentive Sequential Modeling (SASRec)**
- **Generative Retrieval (TIGER via RQ-VAE)**
- **Hybrid Score Fusion (SASRec + EASE)**

The benchmark evaluates these approaches under different data regimes, including **extreme sparsity**, short interaction sequences, and full-catalog ranking.

---

## 1. Project Overview & Architectural Paradigms

This repository provides end-to-end **PyTorch implementations** and empirical ablation studies across four distinct recommendation paradigms.

### Neural Collaborative Filtering (NCF)

Evaluates dual-branch representation learning combining:

- Linear **Generalized Matrix Factorization (GMF)**
- Non-linear **Multi-Layer Perceptrons (MLP)**
- Implicit-feedback learning with dynamic negative sampling
- Capacity ablation across different MLP architectures

The experiments investigate how model capacity affects generalization under sparse user-item interactions.

### Self-Attentive Sequential Recommendation (SASRec)

Implements causal-masked multi-head self-attention to model both short- and long-term sequential dynamics.

The experiments explore:

- Hidden dimensionality ($d$)
- Number of attention heads ($h$)
- Maximum sequence length ($L$)
- Negative sample ratio ($n$)
- Transformer block depth ($b$)

### Generative Retrieval (TIGER)

Reformulates item retrieval as autoregressive generation over discrete, hierarchical **Semantic IDs**.

The pipeline combines:

1. Sentence-T5 text embeddings
2. Residual-Quantized VAE (RQ-VAE)
3. Hierarchical discrete Semantic IDs
4. Seq2Seq Transformers
5. DFA-constrained beam search

The experiments further investigate the effect of user-ID feature hashing and quantization resolution.

### Hybrid Industrial Fusion (SASRec + EASE)

Combines sequential modeling with a closed-form linear autoencoder to address extremely sparse interaction data.

The hybrid system combines:

- **SASRec** for sequential dynamics
- **EASE** for item-item co-occurrence structure
- User-wise min-max score normalization
- Linear score-level fusion

The approach is evaluated on an e-commerce dataset with approximately **99.95% sparsity** and an average sequence length of only **6.32 interactions**.

---

## 2. Experimental Benchmark Matrix

All models are evaluated under a strict **full-catalog ranking protocol**, with historical interactions masked during evaluation.

| Paradigm | Architecture | Dataset | Sparsity / Profile | Recall@10 | NDCG@10 | Key Empirical Finding |
|---|---|---|---|---:|---:|---|
| **Neural CF** | NCF (`[32, 16, 8]` MLP + GMF) | MovieLens 1M (`r ≥ 4`) | ~95.5% (binarized) | **0.1508** | **0.2195** | Compact architectures act as implicit regularizers; larger MLPs overfit rapidly by epoch 4. |
| **Sequential** | SASRec (`d=128, h=2, b=2, L=100`) | MovieLens 1M | Leave-last-two-out | **0.0419**<br>*(Peak: 0.0456 w/ n=3)* | **0.0198**<br>*(Peak: 0.0223 w/ h=1)* | Short sequences (`L=50`) capture higher SNR; contrastive negative sampling (`n=3`) outperforms deeper configurations. |
| **Generative** | TIGER (Sentence-T5 + RQ-VAE + T5) | Amazon Toys & Games (2014) | 5-core, 11.9K items | **0.0275** | **0.0151** | Quantization resolution dominates ranking quality; hashing user IDs introduces representation aliasing. |
| **Hybrid** | SASRec + EASE Fusion (`w=0.90`) | E-Commerce Challenge | 99.9529% (`L_avg=6.32`) | **0.01801** (Kaggle) | — | Linear EASE co-occurrence stabilizes SASRec on short sessions, mitigating tail noise. |

---

## 3. Repository Structure

```text
recsys-benchmark-suite/
├── README.md
├── requirements.txt
│
├── notebooks/
│   ├── 01_tiger_item_only.ipynb
│   └── 02_tiger_user_conditioned.ipynb
│
└── src/
    ├── ncf.py
    ├── sasrec.py
    ├── hybrid_fusion.py
    └── plot_utils.py
```

---

## 4. Environment Setup

### Requirements

- Python 3.10+
- PyTorch 2.2+
- CUDA-compatible GPU recommended / required for the larger experiments

The code has been tested with Python versions up to 3.13.

### Installation

```bash
git clone [https://github.com/hellojiansong/recsys-benchmark-suite.git](https://github.com/hellojiansong/recsys-benchmark-suite.git)
cd recsys-benchmark-suite
pip install -r requirements.txt
```

---

## 5. Dataset Preparation

To keep the repository lightweight and clean, raw dataset files are not tracked in version control. Before running experiments, create a local `data/` directory and place the corresponding datasets inside:

```text
recsys-benchmark-suite/
└── data/
    ├── ratings.dat                   # MovieLens 1M (for NCF & SASRec)
    ├── train.csv                     # E-commerce competition train set (for Hybrid Fusion)
    ├── test.csv                      # E-commerce competition test set (for Hybrid Fusion)
    └── item_meta.csv                 # E-commerce competition metadata (for Hybrid Fusion)
```

- **MovieLens 1M**: Download from the official GroupLens repository and place `ratings.dat` in `data/`.
  - For **NCF**: Explicit ratings $\ge 4$ are converted to positive implicit feedback, and ratings $< 4$ are discarded.
  - For **SASRec**: Explicit ratings $\ge 4$ are converted to implicit feedback, and users with fewer than 5 interactions are filtered out.
- **Amazon Toys & Games (2014)**: Download the 2014 version of Amazon Toys and Games:
  - `Toys_and_Games_5.json.gz` (or `Toys_and_Games.csv`)
  - `meta_Toys_and_Games.json.gz`
  - **Source**: [https://jmcauley.ucsd.edu/data/amazon/](https://jmcauley.ucsd.edu/data/amazon/)
- **E-Commerce Challenge**: Place the competition files (`train.csv`, `test.csv`, and `item_meta.csv`) containing 23,284 users, 13,441 items, and 99.9529% interaction sparsity directly into `data/`.

---

## 6. Reproduction Guide

### Module 1: Neural Collaborative Filtering (NCF)

Runs per-epoch dynamic 1:4 negative sampling, train/val/test evaluation, and capacity ablation over `[128, 64, 32]`, `[64, 32, 16]`, and `[32, 16, 8]` hidden layers:

```bash
python src/ncf.py
```

- **Optimization**: Binary Cross-Entropy (BCE) loss with validation early stopping (patience = 3).
- **Evaluation**: Full-catalog ranking measuring Recall@10 and NDCG@10.

### Module 2: Self-Attentive Sequential Recommendation (SASRec)

Executes next-item prediction over right-aligned sequences (maxlen = 100) and runs automated ablation studies over hidden dimensions ($d \in \{64, 128, 256\}$), heads ($h \in \{1, 2, 4\}$), sequence lengths ($L \in \{50, 100, 200\}$), negative sample ratios ($n \in \{1, 3, 5\}$), and Transformer blocks ($b \in \{1, 2, 3\}$):

```bash
python src/sasrec.py
```

- **Evaluation Protocol**: Chronological leave-last-two-out split with full-catalog candidate scoring.

### Module 3: Generative Retrieval via Semantic IDs (TIGER)

Execute sequentially inside Google Colab (GPU instance recommended) or locally:

- **`notebooks/01_tiger_item_only.ipynb`**: Standard Item-Only baseline. Extracts 768-d text representations from `Title + Brand + Categories + Description` using Sentence-T5, maps continuous embeddings to discrete Semantic IDs via a 3-level RQ-VAE ($K = 256$), and autoregressively generates next-item tokens using an encoder-decoder Transformer (4 layers, $d_{model}=384$, 6 heads, FFN=1024) with DFA-constrained beam search (Beam Size = 20, length penalty $\alpha=0.6$).
- **`notebooks/02_tiger_user_conditioned.ipynb`**: User-conditioned ablation. Evaluates user identity modeling via feature hashing (~19K users mapped into 2,000 slots) with structural offset tokens. Uses a scaled 6-layer Transformer ($d_{model}=512$, 8 heads, FFN=1536) trained with label smoothing 0.1 and decoding length penalty $\alpha=0.0$.

### Module 4: Hybrid Score Fusion (SASRec + EASE)

Trains a 2-block sequential SASRec model (maxlen = 20, hidden = 128, left-padded) with a chronological leave-last-$N$ split and fits a closed-form linear item-item weight matrix $B$ for EASE ($\lambda = 500$) under extreme sparsity:

```bash
python src/hybrid_fusion.py
```

- **Score Integration**: Applies row-wise min-max scaling to both model score distributions and computes weighted consensus recommendations:
  $$S_{\text{hybrid}} = w \cdot \text{Norm}(S_{\text{SASRec}}) + (1 - w) \cdot \text{Norm}(S_{\text{EASE}})$$
  where $w = 0.90$ achieves top test leaderboard performance (Kaggle Recall@10 = 0.01801).

---

## 7. Key Empirical Insights & System Trade-offs

- **Model Capacity vs. Interaction Sparsity (NCF)**: In highly sparse regimes, larger neural networks quickly memorize noise and overfit. The most compact configuration (`[32, 16, 8]`) acted as an implicit regularizer, outperforming deeper variants with Recall@10 of 0.1508.
- **Temporal Locality & Negative Signals (SASRec)**: Truncating user histories to the 50 most recent interactions ($L = 50$) outperformed longer contexts ($L = 100, 200$), showing that recent actions have a higher signal-to-noise ratio. Increasing negative samples from $n = 1$ to $n = 3$ provided more informative contrastive gradients than adding Transformer layers.
- **Representation Granularity vs. User Compression (TIGER)**: Generative retrieval performance is bounded by RQ-VAE codebook expressiveness rather than beam search width. Compressing user IDs via feature hashing introduced representation aliasing, dropping Recall@10 from 0.0275 to 0.0266 despite scaling model capacity.
- **Distribution Shift in Loss Formulation (Hybrid)**: Switching SASRec from BCE to full-softmax cross-entropy raised offline validation Recall@10 (0.0253 $\rightarrow$ 0.0316) but collapsed online test performance to 0.00889 due to temporal distribution shift between static splits and future interactions.

---

## 8. References & Citations

1. Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. *Neural Collaborative Filtering*. In Proceedings of the 26th International Conference on World Wide Web (WWW), 2017.
2. Wang-Cheng Kang and Julian McAuley. *Self-Attentive Sequential Recommendation*. In Proceedings of the IEEE International Conference on Data Mining (ICDM), 2018.
3. Shashank Rajput, Nikhil Mehta, Anima Singh, et al. *Recommender Systems with Generative Retrieval*. In Advances in Neural Information Processing Systems (NeurIPS), 2023.
4. Harald Steck. *Embarrassingly Shallow Autoencoders for Sparse Data*. In The World Wide Web Conference (WWW), 2019.
