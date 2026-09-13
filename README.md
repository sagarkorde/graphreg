# GraphReg

Code for GraphReg, a single-machine pipeline for Bitcoin transaction forensics. It audits a 5.88-million-transaction
mainnet corpus, builds regularity-style graph partitions, and evaluates illicit-transaction detection on the
[Elliptic](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) benchmark.

## Repository layout

| Folder | Contents |
|---|---|
| `icispd2026/` | Scripts behind the ICISPD 2026 manuscript *GraphReg: An Audited Evaluation of Regularity-Partition Graph Context for Bitcoin Transaction Forensics* (Korde and Shekokar). **Use this folder for all current results.** |
| `graphreg_system/` | The earlier checkpointed pipeline, kept unchanged for traceability. Its reported results are superseded; see [below](#status-of-the-earlier-implementation). |

## ICISPD 2026 experiments (`icispd2026/`)

| Script | Purpose |
|---|---|
| `mainnet_scaling.py audit` | Full-corpus audit: overlapping pattern flags, corrected class distribution, unreliable derived columns |
| `mainnet_scaling.py leakage` | Feature-group ablation showing that mainnet pattern labels are recovered from the fields the labelling rules use |
| `mainnet_scaling.py scale` | Single-machine scaling benchmark: column-projected reading, vectorized features, address tokenization, transaction–address incidence partition, XGBoost |
| `mainnet_scaling.py naive` | Baselines: full 53-column load into pandas and the original row-wise label cascade |
| `elliptic_partition_context.py` | k-way partition of the Elliptic graph, spectral certificate for sparse ε-regularity, partition-context features, five-seed temporal evaluation (LR, RF, XGBoost, MLP, GCN) |
| `make_figures.py` | Figures and summary tables |

Requirements and run order are in [`icispd2026/README.md`](icispd2026/README.md).

Headline results from the manuscript:

- The full corpus is tokenized, partitioned, and classified end to end in 94 s with 7.0 GB of peak memory on one laptop.
- Six-class mainnet pattern scores of 1.000 fall to a macro-F1 of 0.715 once count and size features are removed, so they
  measure recovery of the labelling rules rather than forensic generalization.
- On Elliptic (test time steps 35–49, five seeds), a random forest reaches an illicit F1 of 0.824, XGBoost a PR-AUC of 0.803,
  and a GCN reproduction an F1 of 0.517. Partition-context features add about one F1 point to local-feature models and
  nothing measurable once neighbourhood aggregates are included.
- No cluster pair is certified ε-regular on either the Elliptic graph or the mainnet incidence graph.

## Status of the earlier implementation

An audit carried out for the ICISPD 2026 manuscript found errors in results produced with `graphreg_system/`. Those
numbers should not be cited:

- `regularity_solver.py` never counts edges between clusters, so its density matrix carries no pairwise information.
- The ingestion throughput divides the full corpus size by the time needed to read 10 of the 51 row groups.
- The class distribution plotted in `ml_engine.py` is hard-coded, counts overlapping pattern flags more than once, and
  treats OP_RETURN (an attribute) as a class.
- The corpus's stored address counts and `has_taproot` flag are unreliable; `icispd2026/` recomputes them from raw fields.

To run the earlier pipeline for comparison:

```bash
pip install -r requirements.txt
python -m graphreg_system.main
```

## Data

- **Mainnet corpus** (5,884,387 transactions, Parquet): [IEEE DataPort](https://dx.doi.org/10.21227/bxmt-mn56),
  doi:10.21227/bxmt-mn56. Place it at `datasets/custom/Dataset.parquet`.
- **Elliptic**: [Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set). Place the three CSV files under
  `datasets/elliptic/`.

Datasets, experiment outputs, and manuscript sources are not included in this repository.

## Citation

If you use this code or the dataset, please cite:

```
S. Korde and N. Shekokar, "GraphReg: An Audited Evaluation of Regularity-Partition Graph Context for
Bitcoin Transaction Forensics," ICISPD 2026 submission, 2026.

S. Korde, N. Shekokar, and I. Siddavatam, "Bitcoin Blockchain Transaction Dataset for Wallet
Address Profiling and Behavioral Analysis (Parquet Format)," IEEE DataPort, 2026,
doi: 10.21227/bxmt-mn56.
```
