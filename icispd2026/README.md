# GraphReg: ICISPD 2026 experiments

Scripts that produce every number, table, and figure in the ICISPD 2026 manuscript
*GraphReg: An Audited Evaluation of Regularity-Partition Graph Context for Bitcoin Transaction Forensics*.

## Requirements

Python 3.12 with numpy, pandas, pyarrow, scipy, scikit-learn, xgboost (≥ 3.0), torch (CUDA optional), psutil, matplotlib.
The runs reported in the paper used Anaconda Python 3.12 on a 13th-generation Intel Core i9 laptop with 64 GB RAM
(50 GB working limit) and an RTX 4060 Laptop GPU.

## Data

- Mainnet corpus: `datasets/custom/Dataset.parquet` from IEEE DataPort, doi:10.21227/bxmt-mn56.
- Elliptic: `elliptic_txs_features.csv`, `elliptic_txs_classes.csv`, and `elliptic_txs_edgelist.csv` under `datasets/elliptic/`.

Edit the `ROOT` constants at the top of each script if your data lives elsewhere.

## Reproduce

```bash
python elliptic_partition_context.py      # Elliptic: partition, spectral certificate, PC features, 5 seeds, GCN
python mainnet_scaling.py audit           # corpus audit (Table 1)
python mainnet_scaling.py leakage         # feature-group ablation (Table 3)
python mainnet_scaling.py scale           # scaling runs, one subprocess per corpus size (Table 2, Fig. 2)
python mainnet_scaling.py naive           # full-load and row-wise labelling baselines
python make_figures.py                    # figures and printed summaries
```

Results are written to `../results/` as JSON or JSONL, and figures to `../figures/`.
