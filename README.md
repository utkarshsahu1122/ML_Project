# SocialEdge: Graph Link Predictor

A graph-based link prediction system that identifies potential connections in a social network using structural similarity features and machine learning classifiers.

## Overview

Built on the Facebook social circles dataset (~22K nodes, ~171K edges), this project manually engineers graph-theoretic features for each node pair and trains a Random Forest classifier to predict whether an edge (friendship) should exist between two nodes.

## Features

- **Graph construction**: Full social graph built with NetworkX from raw edge data
- **Feature engineering**: 8+ structural similarity metrics computed per node pair:
  - Jaccard Coefficient
  - Adamic-Adar Index
  - Common Neighbors
  - Preferential Attachment
  - Resource Allocation Index
  - and more
- **Classifier comparison**: Benchmarked Logistic Regression, Decision Tree, and Random Forest
- **Threshold optimization**: Precision-Recall curve analysis used to shift decision boundary and reduce false positives

## Results

| Metric | Score |
|---|---|
| ROC-AUC | ~0.97 |
| Precision@K | ~92% |
| False positive reduction | ~18% via threshold tuning |

## Tech Stack

- **Python**
- **NetworkX** — graph construction and traversal
- **Scikit-learn** — classifiers, ROC/PR curve analysis
- **Pandas** — feature matrix construction
- **NumPy** — numerical operations

## How It Works

1. Raw edge list is loaded and converted into a NetworkX `Graph`
2. Positive samples (existing edges) and negative samples (non-edges) are collected
3. For each node pair, 8+ structural features are computed to form a feature vector
4. Three classifiers are trained and evaluated; Random Forest achieves best generalization
5. Decision threshold is tuned using the Precision-Recall curve to reduce false positives

## Dataset

Facebook Social Circles dataset — ~22,000 nodes, ~171,000 edges (ego networks from Facebook)

## Usage

```bash
git clone https://github.com/<your-username>/socialedge-link-predictor
cd socialedge-link-predictor
pip install -r requirements.txt
python train.py
python predict.py --node1 <ID> --node2 <ID>
```
