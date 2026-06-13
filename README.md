# DL Homework 2 — BloodMNIST CNNs & RNA Binding Affinity Prediction

**Course:** Deep Learning — IST, 2025-26  
**Authors:** José Machado (1106643) · Gil Jorge (110062) · Elias Heimdal (116702)

---

## Overview

This repository contains the code and results for two questions:

- **Question 1:** Image classification with CNNs on the BloodMNIST dataset.
- **Question 2:** RNA–protein binding affinity prediction (regression) using CNN, BiLSTM, and attention-augmented architectures for the RBP RBFOX1.

---

## Repository structure

```
.
├── hw2_q1.py                               # Q1: CNN training (all 4 model variants)
├── hw2_q1_1_plots.py                       # Q1: generates loss and validation accuracy plots
├── model_train_metrics.json                # Q1: saved validation/training metrics per epoch
├── time_measures.json                      # Q1: training time per model
│
├── HW2_Q2_1_t*.ipynb                       # Q2.1: CNN and BiLSTM grid search notebooks
├── plot_full_grid.py                       # Q2.1: plots validation correlations across grid
├── get_bestModels.ipynb                    # Q2.1/2.2: extracts best hyperparameters and plots
├── homework2_q2.3_kaggle_notebook.ipynb    # Q2.2/2.3: attention-LSTM and heatmaps
│
├── Outputs_t*/
│   ├── CNN_search_results_t*.json          # Q2.1 CNN grid search results
│   └── LSTM_search_results_t*.json         # Q2.1 LSTM grid search results
├── Attention_LSTM_search_results_2.2.json  # Q2.2 attention-LSTM results
│
└── utils.py                                # Data loader, masked_mse_loss, evaluation utils
```

---

## Setup

### Dependencies

```bash
pip install torch torchvision medmnist scipy numpy matplotlib
```

### Data

**Question 1 — BloodMNIST:**
```bash
pip install medmnist
```
The dataset is downloaded automatically via `medmnist.BloodMNIST`.

**Question 2 — RNAcompete (RBFOX1):**  
Download from [Google Drive](https://drive.google.com/drive/folders/1b9FfWZqEtPEdsSDu_lWQIJiPP7Z3rBKL?usp=sharing) and place in the project root. Load with:
```python
from utils import load_rnacompete_data
X_train, y_train = load_rnacompete_data('RBFOX1', split='train')
```
Data is pre-processed (one-hot encoded, log-transformed, z-scored). Sequences range from 38 to 41 nucleotides, padded to a fixed length.

---

## Question 1 — BloodMNIST image classification

Four CNN variants are trained on BloodMNIST (17,092 images, 8 classes, split 7:1:2) for 200 epochs with Adam (lr=0.001, batch size 64).

| Model | Max Pooling | Softmax Output |
|-------|-------------|----------------|
| Model 1 | No | No (logits) |
| Model 2 | No | Yes |
| Model 3 | Yes | No (logits) |
| Model 4 | Yes | Yes |

**Run training:**
```bash
python hw2_q1.py
```

**Generate plots:**
```bash
python hw2_q1_1_plots.py
```

### Key results

| Model | Best Epoch | Val. Acc. | Test Acc. | Total Time (s) |
|-------|-----------|-----------|-----------|----------------|
| 1 (no pool, logits) | 23 | 0.939 | 0.935 | 1365 |
| 2 (no pool, softmax) | 9 | 0.430 | 0.435 | 1320 |
| 3 (pool, logits) | 60 | 0.954 | 0.953 | 756 |
| 4 (pool, softmax) | 133 | 0.946 | 0.933 | 768 |

**Logits vs Softmax:** Applying softmax before `nn.CrossEntropyLoss` (which internally applies log-softmax) causes double-softmax and vanishing gradients — Model 2 degrades to ~0.43 accuracy.

**Max pooling:** Halves training time (~1.8× speedup) and improves peak accuracy by reducing spatial resolution of feature maps. Model 3 achieves the best test accuracy (0.953); Model 1 is the best accuracy/speed trade-off.

---

## Question 2 — RNA binding affinity prediction

### Task

Predict the binding affinity (measured as Spearman rank correlation) between RBFOX1 and synthetic RNA sequences (~241,000 sequences of 38–41 nucleotides). Loss: masked MSE (`masked_mse_loss` from `utils.py`, NaN-safe). Primary metric: Spearman rank correlation.

### 2.1 — CNN and BiLSTM

**CNN architecture:** Three conv layers (kernel sizes 5, 7, 9), each followed by batch normalisation. Output passed through max and average pooling, then two FC layers with ReLU.

**BiLSTM architecture:** Two-layer bidirectional LSTM on one-hot encoded input, batch normalisation, dropout, two FC layers with ReLU. Final representation concatenates last hidden states from both directions.

Hyperparameter search uses an iterative grid strategy (successive small grids around promising regions) over learning rate, dropout rate, and first FC layer width. Results saved to `Outputs_t*/`.

**Best hyperparameters:**

| Model | lr | dropout | H | Best Val Corr | Test Corr | Best Epoch |
|-------|----|---------|---|---------------|-----------|------------|
| CNN | 5×10⁻⁴ | 0.1 | 256 | 0.672 | 0.665 | 28 |
| BiLSTM | 5×10⁻⁴ | 0.3 | 256 | 0.682 | 0.681 | 48 |

BiLSTM outperforms CNN by ~1% and shows better generalisation (lower train–val correlation gap at both best and final epochs).

**Run grid search:**
```bash
jupyter nbconvert --to notebook --execute HW2_Q2_1_t1.ipynb
```

**Get best models and plots:**
```bash
jupyter nbconvert --to notebook --execute get_bestModels.ipynb
```

**Plot full grid:**
```bash
python plot_full_grid.py
```

### 2.2 — Attention-augmented BiLSTM

Single-head and two-head multi-head attention (MHA) added to the best vanilla BiLSTM. Evaluated at `dr=0.3`, `H=256`, `lr ∈ {3×10⁻⁴, 5×10⁻⁴}`.

| Heads | LR | Best Val Corr |
|-------|----|---------------|
| 1 | 3×10⁻⁴ | 0.67778 |
| 2 | 3×10⁻⁴ | 0.67118 |
| 1 | 5×10⁻⁴ | 0.67811 |
| **2** | **5×10⁻⁴** | **0.68042** |

Best attention-LSTM: MSE 0.3136, Spearman **0.6794** on test set — marginal improvement over vanilla BiLSTM, consistent with the binding motif being localised (single-head sufficient).

Attention heatmaps confirm the expected biology: high-affinity sequences produce focused attention on the GCA nucleotides of the (U)GCAUG motif; low-affinity sequences show diffuse, near-uniform attention. The second attention head mirrors the first.

**Run attention experiments and generate heatmaps:**
```bash
jupyter nbconvert --to notebook --execute homework2_q2.3_kaggle_notebook.ipynb
```

### 2.3 — Extension to multiple RBPs

See the report (`HW2_GROUP_37_REPORT.pdf`) for a full description of how to generalise the pipeline to multiple RNA-binding proteins simultaneously, covering:
- Data: sequence–protein tuples with balanced sampling across proteins
- Architecture: cross-attention and FiLM (Feature-wise Linear Modulation) conditioning on protein identity
- Evaluation: per-protein Spearman correlation averaged across proteins, with loss weighting for protein frequency

---

## Results summary

| Model | Test Spearman |
|-------|--------------|
| CNN | 0.665 |
| BiLSTM | 0.681 |
| Attention-BiLSTM (2 heads, lr=5×10⁻⁴) | **0.6794** |

---

## References

- Sun et al. (2012). Mechanisms of activation and repression by RBFOX1/2. *RNA* 18(2).
- Ray et al. (2013). A compendium of RNA-binding motifs. *Nature* 499.
- Vaswani et al. (2017). Attention Is All You Need. arXiv:1706.03762.
- Karpathy (2015). The Unreasonable Effectiveness of RNNs.
- Zhang et al. (2018). Understanding deep learning requires rethinking generalization. arXiv:1808.01174.
- Xie et al. (2019). Deep Features Analysis with Attention Networks. AAAI-19 Workshop.
