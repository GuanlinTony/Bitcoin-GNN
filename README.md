# Bitcoin Fraud Detection with Graph Convolutional Networks

# Usage:

Open `Bitcoin_(4).ipynb` to see the modeling codes.
The `Bitcoin_GCN_Project_Report.pdf` is a written technical and theoretical-heavy report that illustrates our methodology and results. The original data is from Kaggle.

## Thesis Background and Goals

Traditional fraud detection treats each transaction in isolation. That's a problem for Bitcoin, where illicit activity often shows up as *patterns* — laundering loops, coordinated transfer clusters, suspicious neighborhoods in the transaction graph. This project tackles that by applying a Graph Convolutional Network (GCN) to the Elliptic Bitcoin Dataset, letting the model learn from both transaction features and how transactions are connected to each other.

The short version: graph structure matters for fraud detection, and the numbers back that up.

## Dataset

**Elliptic Bitcoin Dataset** — a labeled snapshot of the Bitcoin network.

- 203,769 transaction nodes, 234,355 directed edges
- 166 engineered features per node
- 49 time steps (~2-week intervals)
- Labels: 2.23% illicit, 20.62% licit, 77.15% unknown

The class imbalance is severe. This shapes basically every design decision in the project — evaluation metric, training strategy, and how results are interpreted.

Unknown-labeled nodes stay in the graph for message passing but are excluded from the supervised loss.

## Approach

### Graph Construction

Each transaction is a node. Directed edges represent fund flows. Node features are the 166 transaction attributes plus a normalized time step. Degree centrality was also analyzed to understand how hub transactions influence fraud propagation through the network.

### Model

A two-layer GCN implemented in PyTorch Geometric. The standard Kipf & Welling formulation:

$$H^{(l+1)} = \sigma\left(\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}H^{(l)}W^{(l)}\right)$$

Deeper networks were tested and consistently hurt performance — over-smoothing collapses node representations into indistinguishable embeddings past 2 layers.

### Data Splitting

Three strategies were compared, each with different tradeoffs:

| Strategy | Description | Key Consideration |
|---|---|---|
| Random Split | 80/10/10 random | Baseline; risks temporal leakage |
| Chronological Split | Train early → test late | Mirrors real AML deployment |
| Rolling Window | Periodic retraining on recent data | Approximates continuous monitoring |


## Training Setup

| Parameter | Value |
|---|---|
| GCN Layers | 2 |
| Training | Mini-batch |
| Dropout | 0.5 |
| L2 Regularization | 0.0005 |
| Optimizer | Adam |
| Loss | Negative Log Likelihood |


## Results

Primary metric: **F1 score on the illicit class** (accuracy is meaningless at 2% prevalence).

| Experiment | Best Illicit F1 |
|---|---|
| Layer depth tuning | 0.632 |
| Mini-batch sampling | 0.751 |
| Regularization tuning | **0.764** |

Best configuration: 2-layer GCN + mini-batch sampling + chronological split.

Key observations:
- A transaction's neighbors are strongly predictive — fraud clusters spatially in the graph
- Mini-batch training reduces temporal leakage compared to full-batch
- Fraud signals decay over time; models trained on old data degrade on newer transactions

## Limitations

- 2% illicit prevalence makes the problem inherently hard; small recall drops have large real-world consequences
- Mini-batch sampling can miss important subgraph structures
- Vanilla GCN has no explicit temporal modeling — it doesn't know that time step 40 is "after" time step 10


## Future Steps

- Hybrid approach: use GCN embeddings as features for a Random Forest or XGBoost classifier
- Temporal encodings to make the model time-aware
- More expressive GNN architectures: Graph Attention Networks (GAT) or Graph Isomorphism Networks (GIN)

---

## References

- Weber et al. (2019) — *Anti-Money Laundering in Bitcoin*
- Kipf & Welling (2016) — *Semi-Supervised Classification with GCNs*
- [Elliptic Dataset on Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)



