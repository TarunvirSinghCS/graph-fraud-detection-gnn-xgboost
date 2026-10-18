# Graph-Based Credit Card Fraud Detection: GNN vs. XGBoost

Does a graph neural network (GNN) on a customer–merchant transaction graph beat a tuned XGBoost model at credit card fraud detection? This project compares the two under both a **random** split and a **temporal** split (train on earlier transactions, test on later ones).

Author: Tarunvir Singh. Built as a research project with Inspirit AI (mentor: Mikhail Baranchuk).

## Main findings

PR-AUC is the main metric, since only 0.39% of transactions are fraud.

| Model | Split | PR-AUC | ROC-AUC | Recall | Precision |
|---|---|---|---|---|---|
| XGBoost (tuned) | Random | 0.870 | 0.991 | 0.825 | 0.839 |
| XGBoost (tuned) | Temporal | 0.700 | 0.972 | 0.668 | 0.724 |
| GNN (5-seed mean) | Random | 0.663 | 0.988 | 0.941 | 0.076 |
| GNN (main run) | Temporal | 0.420 | 0.982 | 0.815 | 0.070 |

Precision and recall use a 0.5 threshold. The temporal GNN's PR-AUC ranged from 0.353 to 0.487 across three seeds.

- Tuned XGBoost beat the GNN on both splits and dropped less under temporal shift.
- Message passing over an undirected graph gave no detectable benefit over directed-only.
- Letting the temporal graph include test-period connections made no detectable difference.
- Before XGBoost was tuned, the GNN looked more robust to temporal shift. That advantage disappeared once the baseline was tuned.
- An early version of the random-split GNN evaluation tested on training data because of an index reset after a merge. It was caught and fixed, and all reported GNN random-split numbers come from the corrected split.

The data is simulated, so these results compare the two setups on this dataset and say nothing about real bank data. See the paper for limitations.

## Repository contents

```
.
├── fraud_gnn_xgboost.ipynb     # full notebook: preprocessing, XGBoost, GNN, experiments
├── paper/
│   └── Inspirit_AI_Project_Final.pdf
├── figures/                    # figures used in the paper
├── requirements.txt
└── README.md
```

## Data

The notebook uses `fraudTest.csv` from the Kaggle dataset
[Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
(generated with the Sparkov tool). The data is not included in this repository. Download it from Kaggle and place `fraudTest.csv` where the notebook expects it.

## How to run

**In Colab (easiest):** open the notebook in Colab, upload `fraudTest.csv`, choose a GPU runtime, and use Runtime → Run all.

**Locally:**

```bash
pip install -r requirements.txt
jupyter notebook fraud_gnn_xgboost.ipynb
```

PyTorch Geometric may need a PyTorch-specific install command. See the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

GNN results vary from run to run, even with a fixed seed, because of GPU nondeterminism. The paper reports several seeds where possible.

## Setup details

- XGBoost: `scale_pos_weight` set to the training negative-to-positive ratio; depth 9, 200 trees, learning rate 0.1 (only depth was tuned).
- GNN: two GraphSAGE (`SAGEConv`) layers, 64 hidden units, dropout 0.3, class-weighted cross-entropy, Adam (lr 0.005, weight decay 5e-4), 200 epochs.
- Random split: 80/20 stratified, seed 42. Temporal split: cutoff December 7, 2020.

## Paper

The full write-up is in `paper/`.
