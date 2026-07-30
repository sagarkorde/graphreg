# GraphReg

A checkpointed, resumable software pipeline for Bitcoin mainnet transaction graph analysis, wallet-pattern
classification, and cross-dataset evaluation against the [Elliptic](https://www.elliptic.co/) benchmark.
This is the reference implementation for *GraphReg: A Scalable System Implementation for Sparse Regularity
Lemma-Based Bitcoin Graph Analysis and Wallet Profiling* (Korde and Shekokar).

GraphReg is a single-machine pipeline (no distributed compute layer or service interface) organized into
five resumable stages:

1. **Streaming ingestion** (`data_ingestion.py`): memory-mapped, row-group-wise Parquet reading of the
   mainnet ledger, avoiding full-file materialization.
2. **Feature extraction** (`feature_extractor.py`): 23 topological, monetary, and protocol-level features
   per transaction, plus a deterministic structural rule cascade that assigns mainnet pattern labels
   (peer-to-peer, consolidation, distribution, batch payment, CoinJoin-like, other).
3. **Sparse density partitioning** (`regularity_solver.py`): quantile-based clustering of transactions by
   degree, with a k×k bipartite density matrix flagging irregular (high-density) cluster pairs. This is a
   tractable heuristic *inspired by* Szemerédi's Sparse Regularity Lemma, not a formal implementation of it:
   exhaustively verifying epsilon-regularity requires checking an exponential number of subset pairs, which
   this solver does not do.
4. **ML inference engine** (`ml_engine.py`): Logistic Regression, Random Forest, XGBoost, and an MLP for
   six-class pattern classification, plus a dedicated binary XGBoost model for CoinJoin-like anomaly
   detection.
5. **Elliptic benchmark evaluation** (`elliptic_benchmark.py`): the same model family, retrained from
   scratch on the independently-labeled [Elliptic dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
   (203,769 transactions, 46,564 labeled) under a strict temporal train/test split, with no parameter
   transfer from the mainnet stage.

`checkpoint.py` and `logger.py` are cross-cutting: every stage persists its output (as Parquet or pickled
objects) and checks a JSON state file before recomputing, so the pipeline can be interrupted and resumed
without repeating upstream work. The logger records structured, timestamped throughput/latency/memory
metrics for each stage.

## Important caveat on results

Mainnet pattern labels are a **deterministic function of the same structural fields** (input/output counts,
script types) supplied to the classifiers. Near-ceiling mainnet F1 scores therefore demonstrate that the
classifiers correctly recover this known labeling rule, not that they generalize to unseen transaction
structure. The Elliptic benchmark, whose labels come from an independent forensic investigation, is the
primary evidence of real-world classification capability.

Similarly, the ingestion engine's reported throughput is **streaming row-group I/O throughput**, not
full end-to-end pipeline throughput. Feature extraction, partitioning, and model training are separately
timed, slower stages.

## Usage

```bash
pip install -r requirements.txt
```

Download the primary mainnet transaction corpus (5,884,387 Bitcoin transactions, Parquet format) from
[IEEE DataPort](https://dx.doi.org/10.21227/bxmt-mn56) (DOI: 10.21227/bxmt-mn56) and place it at
`datasets/custom/Dataset.parquet`. Place the [Elliptic](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
CSVs under `datasets/elliptic/`. Then run:

```bash
python -m graphreg_system.main
```

Each stage's outputs and metrics are checkpointed under `graphreg_system/checkpoints/`; re-running the
command resumes from the last completed stage rather than recomputing everything.

## Requirements

- Python 3.10+
- See `requirements.txt`. XGBoost will use CUDA histogram construction automatically when a CUDA-capable
  GPU is available (`torch.cuda.is_available()`); it falls back to CPU otherwise.

## Citation

If this code or the accompanying dataset is used, please cite:

```
S. Korde and N. Shekokar, "GraphReg: A Scalable System Implementation for Sparse Regularity
Lemma-Based Bitcoin Graph Analysis and Wallet Profiling," IEEE conference submission, 2026.

S. Korde, N. Shekokar, and I. Siddavatam, "Bitcoin Blockchain Transaction Dataset for Wallet
Address Profiling and Behavioral Analysis (Parquet Format)," IEEE DataPort, 2026,
doi: 10.21227/bxmt-mn56.
```

## Repository scope

This repository contains the GraphReg system implementation only. The manuscript-generation tooling used
to produce the accompanying paper is intentionally excluded.
