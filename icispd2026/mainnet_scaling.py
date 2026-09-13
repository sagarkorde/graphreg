"""
Mainnet corpus audit, leakage-controlled pattern classification, and single-machine
scaling benchmark for the ICISPD 2026 GraphReg manuscript.

Usage (Anaconda Python):
    python mainnet_scaling.py smoke      # n = 20,000 sanity run (separate output file)
    python mainnet_scaling.py audit      # full-corpus statistics and data-quality checks
    python mainnet_scaling.py leakage    # feature-group ablation at n = 1,000,000
    python mainnet_scaling.py scale      # each corpus size runs in a fresh subprocess
    python mainnet_scaling.py naive      # full materialisation + row-wise labelling baseline
"""
import os
import sys
import json
import time
import threading
import subprocess

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import scipy.sparse as sp
import psutil
import xgboost as xgb
from sklearn.metrics import f1_score, accuracy_score

ROOT = r"E:\DY Patil IEEE Conf"
DATA = os.path.join(ROOT, "datasets", "custom", "Dataset.parquet")
OUT = os.path.join(ROOT, "PuneCon", "ICISPD_2026", "results")
os.makedirs(OUT, exist_ok=True)

SIZES = [250_000, 500_000, 1_000_000, 2_000_000, 4_000_000, 5_884_387]
FLAGS = [(0, "is_peer_to_peer"), (1, "is_consolidation"), (2, "is_distribution"),
         (3, "is_batch_payment"), (4, "is_coinjoin_like")]  # later entries take priority
CLASS_NAMES = ["P2P", "Consolidation", "Distribution", "Batch payment", "CoinJoin-like", "Other"]
ACTIVITY_EDGES = [1, 2, 3, 6, 21, 101]   # address activity bins: 1, 2, 3-5, 6-20, 21-100, >100
BASE_COLS = ["block_height", "size", "vsize", "weight", "input_count", "output_count",
             "total_input_value", "total_output_value", "fee", "input_output_ratio",
             "value_difference", "has_op_return", "rbf_enabled", "avg_input_value",
             "avg_output_value", "value_concentration_ratio", "input_addresses",
             "output_addresses", "input_script_types", "output_script_types"] + [f for _, f in FLAGS]
COUNT_FEATURES = ["input_count", "output_count", "in_addr_n", "out_addr_n", "total_addr_n",
                  "in_script_n", "out_script_n", "address_reuse", "avg_input_value", "avg_output_value"]
SIZE_FEATURES = ["size", "vsize", "weight"]


class PeakMemory:
    def __init__(self, interval=0.05):
        self.interval, self.peak, self._stop = interval, 0, False
        self.proc = psutil.Process()

    def __enter__(self):
        self.peak = self.proc.memory_info().rss
        self._stop = False
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()
        return self

    def _run(self):
        while not self._stop:
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(self.interval)

    def __exit__(self, *a):
        self._stop = True
        self.t.join()


def emit(name, record):
    with open(os.path.join(OUT, name), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=float) + "\n")
    print(json.dumps(record, default=float)[:3000], flush=True)


# --------------------------------------------------------------------------- stages
def read_rows(n, columns):
    """Column-projected, row-group-wise read of the first n rows. Row groups in the
    corpus are not ordered by time, so a prefix spans the whole collection window."""
    pf = pq.ParquetFile(DATA)
    parts, got = [], 0
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=columns)
        if got + t.num_rows > n:
            t = t.slice(0, n - got)
        parts.append(t)
        got += t.num_rows
        if got >= n:
            break
    return pa.concat_tables(parts)


def tokens(list_col):
    """Explode a chunked list<string> column whose elements may hold several ';'-joined
    values. Returns (list of large_string token chunks, parent row index array)."""
    tok_chunks, parents, base = [], [], 0
    for arr in list_col.chunks:
        offsets = arr.offsets.to_numpy()
        flat = arr.values.slice(offsets[0], offsets[-1] - offsets[0])
        parent = np.repeat(np.arange(len(arr)), np.diff(offsets)) + base
        parts = pc.split_pattern(flat, ";")
        p_off = parts.offsets.to_numpy()
        tok = parts.values.slice(p_off[0], p_off[-1] - p_off[0])
        tok_parent = np.repeat(parent, np.diff(p_off))
        keep = pc.fill_null(pc.and_(pc.is_valid(tok), pc.not_equal(tok, "")), False)
        tok_chunks.append(tok.filter(keep).cast(pa.large_string()))
        parents.append(tok_parent[keep.to_numpy(zero_copy_only=False)])
        base += len(arr)
    return tok_chunks, (np.concatenate(parents) if parents else np.zeros(0, np.int64))


def extract(tb):
    """Vectorised feature and label extraction (replaces the row-wise cascade)."""
    n = tb.num_rows
    col = lambda c: tb[c].to_numpy()
    in_chunks, in_par = tokens(tb["input_addresses"])
    out_chunks, out_par = tokens(tb["output_addresses"])
    all_tok = pa.chunked_array(in_chunks + out_chunks, type=pa.large_string())
    uniq = pc.unique(all_tok)
    ids = pc.index_in(all_tok, value_set=uniq)
    ids = np.concatenate([c.to_numpy(zero_copy_only=False) for c in ids.chunks]).astype(np.int64)
    n_addr = len(uniq)
    n_in = len(in_par)
    in_ids, out_ids = ids[:n_in], ids[n_in:]
    in_keys = np.unique(in_par.astype(np.int64) * n_addr + in_ids)
    out_keys = np.unique(out_par.astype(np.int64) * n_addr + out_ids)
    in_addr_n = np.bincount(in_keys // n_addr, minlength=n)
    out_addr_n = np.bincount(out_keys // n_addr, minlength=n)
    reuse = np.bincount(np.intersect1d(in_keys, out_keys, assume_unique=True) // n_addr, minlength=n)
    total_addr_n = np.bincount(np.union1d(in_keys, out_keys) // n_addr, minlength=n)
    _, in_s_par = tokens(tb["input_script_types"])
    out_s_chunks, out_s_par = tokens(tb["output_script_types"])
    taproot_out = int(sum((pc.sum(pc.equal(c, "witness_v1_taproot")).as_py() or 0) for c in out_s_chunks))
    fee_sat = col("fee").astype(np.float64) * 1e8
    X = pd.DataFrame({
        "size": col("size"), "vsize": col("vsize"), "weight": col("weight"),
        "input_count": col("input_count"), "output_count": col("output_count"),
        "total_input_value": col("total_input_value"), "total_output_value": col("total_output_value"),
        "fee": col("fee"), "input_output_ratio": col("input_output_ratio"),
        "value_difference": col("value_difference"),
        "fee_rate_sat_per_byte": fee_sat / np.maximum(col("size"), 1),
        "fee_rate_sat_per_vbyte": fee_sat / np.maximum(col("vsize"), 1),
        "in_addr_n": in_addr_n, "out_addr_n": out_addr_n, "total_addr_n": total_addr_n,
        "in_script_n": np.bincount(in_s_par, minlength=n), "out_script_n": np.bincount(out_s_par, minlength=n),
        "address_reuse": reuse, "has_op_return": col("has_op_return").astype(np.int8),
        "rbf_enabled": col("rbf_enabled").astype(np.int8), "avg_input_value": col("avg_input_value"),
        "avg_output_value": col("avg_output_value"),
        "value_concentration_ratio": col("value_concentration_ratio"),
    }).astype(np.float32)
    y = np.full(n, 5, dtype=np.int8)
    for code, flag in FLAGS:
        y[col(flag)] = code
    return X, y, {"in_par": in_par, "in_ids": in_ids, "out_par": out_par, "out_ids": out_ids,
                  "n_addr": n_addr, "n_tokens": int(len(ids)), "taproot_output_tokens": taproot_out}


def sigma1_power(blk, iters=30, seed=0):
    ni, nj = blk.shape
    d = blk.nnz / (ni * nj)
    v = np.random.default_rng(seed).standard_normal(nj)
    v /= np.linalg.norm(v)
    s = 0.0
    for _ in range(iters):
        u = blk @ v - d * v.sum()
        u /= np.linalg.norm(u) + 1e-12
        v = blk.T @ u - d * u.sum()
        s = np.linalg.norm(v)
        v /= s + 1e-12
    return float(s)


def partition(y, g, n):
    """Transaction-class x address-activity partition of the bipartite incidence graph."""
    rows = np.concatenate([g["in_par"], g["out_par"]])
    cols = np.concatenate([g["in_ids"], g["out_ids"]])
    B = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, g["n_addr"]))
    B.data[:] = 1.0
    activity = np.diff(B.tocsc().indptr)
    abin = np.searchsorted(ACTIVITY_EDGES, activity, side="right") - 1
    ro, co = np.argsort(y, kind="stable"), np.argsort(abin, kind="stable")
    rank_r = np.empty(n, np.int64)
    rank_r[ro] = np.arange(n)
    rank_c = np.empty(g["n_addr"], np.int64)
    rank_c[co] = np.arange(g["n_addr"])
    C = B.tocoo()
    Bp = sp.csr_matrix((C.data, (rank_r[C.row], rank_c[C.col])), shape=B.shape)
    rs = np.concatenate([[0], np.cumsum(np.bincount(y, minlength=6))])
    cs = np.concatenate([[0], np.cumsum(np.bincount(abin, minlength=6))])
    p = B.nnz / (n * g["n_addr"])
    K = 6
    e, Dp, eta = np.zeros((K, K)), np.zeros((K, K)), np.zeros((K, K))
    for i in range(K):
        Bi = Bp[rs[i]:rs[i + 1]]
        for j in range(K):
            ni, nj = rs[i + 1] - rs[i], cs[j + 1] - cs[j]
            if ni == 0 or nj == 0:
                continue
            blk = Bi[:, cs[j]:cs[j + 1]].tocsr()
            e[i, j] = blk.nnz
            Dp[i, j] = blk.nnz / (p * ni * nj)
            if blk.nnz and min(ni, nj) > 1:
                eta[i, j] = sigma1_power(blk) / (p * np.sqrt(ni * nj))
    return {"n_addr": int(g["n_addr"]), "incidences": int(B.nnz), "p": p,
            "tx_per_class": np.diff(rs).tolist(), "addr_per_bin": np.diff(cs).tolist(),
            "edges": e.tolist(), "Dp": Dp.tolist(), "eta": eta.tolist()}


def temporal_split(block_height, frac=0.8):
    cut = np.quantile(block_height, frac)
    return block_height <= cut, block_height > cut


def train_xgb(X, y, tr, te, device):
    mdl = xgb.XGBClassifier(n_estimators=100, max_depth=8, learning_rate=0.1, tree_method="hist",
                            device=device, n_jobs=-1, random_state=42, eval_metric="mlogloss")
    t0 = time.time()
    mdl.fit(X[tr], y[tr])
    t_fit = time.time() - t0
    pred = mdl.predict(X[te])
    return mdl, t_fit, pred, {"accuracy": accuracy_score(y[te], pred),
                              "macro_f1": f1_score(y[te], pred, average="macro"),
                              "weighted_f1": f1_score(y[te], pred, average="weighted")}


# ---------------------------------------------------------------------------- modes
def run_size(n, out_name="mainnet_scaling.jsonl"):
    rec = {"n": n}
    with PeakMemory() as pm:
        t0 = time.time()
        tb = read_rows(n, BASE_COLS)
        t1 = time.time()
        X, y, g = extract(tb)
        t2 = time.time()
        part = partition(y, g, n)
        t3 = time.time()
        bh = tb["block_height"].to_numpy()
        del tb
        tr, te = temporal_split(bh)
        _, t_gpu, _, m_gpu = train_xgb(X, y, tr, te, "cuda")
        t4 = time.time()
    rec.update({"read_s": t1 - t0, "extract_s": t2 - t1, "partition_s": t3 - t2, "xgb_gpu_fit_s": t_gpu,
                "total_s": t4 - t0, "peak_rss_gb": pm.peak / 2**30, "n_tokens": g["n_tokens"],
                "throughput_read_to_partition_tx_s": n / (t3 - t0), "throughput_end_to_end_tx_s": n / (t4 - t0),
                "class_counts": np.bincount(y, minlength=6).tolist(), "xgb_gpu_metrics": m_gpu,
                "partition": part, "taproot_output_tokens": g["taproot_output_tokens"]})
    if n <= 1_000_000:
        _, t_cpu, _, m_cpu = train_xgb(X, y, tr, te, "cpu")
        rec.update({"xgb_cpu_fit_s": t_cpu, "xgb_cpu_metrics": m_cpu})
    emit(out_name, rec)


def run_leakage(n=1_000_000):
    tb = read_rows(n, BASE_COLS)
    X, y, _ = extract(tb)
    tr, te = temporal_split(tb["block_height"].to_numpy())
    groups = {
        "all_23": list(X.columns),
        "count_features_only": COUNT_FEATURES,
        "without_count_features": [c for c in X.columns if c not in COUNT_FEATURES],
        "value_fee_only": [c for c in X.columns if c not in COUNT_FEATURES + SIZE_FEATURES],
    }
    for name, cols in groups.items():
        Xg = X[cols].to_numpy()
        mdl, t_fit, pred, m = train_xgb(Xg, y, tr, te, "cuda")
        gain = pd.Series(mdl.get_booster().get_score(importance_type="gain"))
        gain.index = [cols[int(k[1:])] for k in gain.index]
        per_class = f1_score(y[te], pred, average=None, labels=list(range(6)), zero_division=0)
        emit("mainnet_leakage.jsonl", {"n": n, "group": name, "n_features": len(cols), "fit_s": t_fit, **m,
                                        "test_class_counts": np.bincount(y[te], minlength=6).tolist(),
                                        "per_class_f1": dict(zip(CLASS_NAMES, per_class.tolist())),
                                        "top_gain": (gain / gain.sum()).sort_values(ascending=False).head(5).to_dict()})


def run_audit():
    cols = ["block_height", "timestamp", "year", "input_count", "output_count", "has_op_return", "has_coinbase",
            "has_taproot", "fee", "fee_rate_sat_per_vbyte", "sample_size"] + [f for _, f in FLAGS]
    tb = pq.read_table(DATA, columns=cols)
    d = {c: tb[c].to_numpy() for c in cols if c != "timestamp"}
    n = tb.num_rows
    y = np.full(n, 5, dtype=np.int8)
    for code, flag in FLAGS:
        y[d[flag]] = code
    M = np.vstack([d[f] for _, f in FLAGS]).astype(int)
    overlap = {}
    for i, (_, a) in enumerate(FLAGS):
        for _, b in FLAGS[i + 1:]:
            c = int((d[a] & d[b]).sum())
            if c:
                overlap[f"{a}&{b}"] = c
    ts = tb["timestamp"]
    rec = {
        "rows": n, "block_height_min": int(d["block_height"].min()), "block_height_max": int(d["block_height"].max()),
        "time_min": str(pc.min(ts).as_py()), "time_max": str(pc.max(ts).as_py()),
        "rows_by_year": {int(k): int(v) for k, v in zip(*np.unique(d["year"], return_counts=True))},
        "raw_flag_counts": {f: int(d[f].sum()) for _, f in FLAGS},
        "rows_with_multiple_flags": int((M.sum(0) > 1).sum()), "flag_overlaps": overlap,
        "class_counts": dict(zip(CLASS_NAMES, np.bincount(y, minlength=6).tolist())),
        "op_return_by_class": dict(zip(CLASS_NAMES, [int(d["has_op_return"][y == k].sum()) for k in range(6)])),
        "op_return_total": int(d["has_op_return"].sum()), "coinbase_total": int(d["has_coinbase"].sum()),
        "has_taproot_flag_total": int(d["has_taproot"].sum()),
        "fee_rate_column_zero_frac": float(np.mean(d["fee_rate_sat_per_vbyte"] == 0)),
        "fee_positive_frac": float(np.mean(d["fee"] > 0)),
        "distinct_sample_size_values": np.unique(d["sample_size"]).tolist(),
    }
    del tb, d
    tb2 = read_rows(1_000_000, BASE_COLS + ["input_address_count", "output_address_count"])
    X, _, g = extract(tb2)
    in_list_len = np.diff(tb2["input_addresses"].combine_chunks().offsets.to_numpy())
    rec["address_check_rows"] = tb2.num_rows
    rec["stored_input_address_count_eq_list_len_frac"] = float(np.mean(tb2["input_address_count"].to_numpy() == in_list_len))
    rec["stored_input_address_count_mismatch_frac"] = float(np.mean(tb2["input_address_count"].to_numpy() != X["in_addr_n"].to_numpy()))
    rec["stored_output_address_count_mismatch_frac"] = float(np.mean(tb2["output_address_count"].to_numpy() != X["out_addr_n"].to_numpy()))
    rec["taproot_output_tokens_first_1M"] = g["taproot_output_tokens"]
    rec["distinct_addresses_first_1M"] = g["n_addr"]
    emit("mainnet_audit.jsonl", rec)


def run_naive():
    """Baseline: full 53-column materialisation into pandas plus the original row-wise cascade."""
    rec = {}
    with PeakMemory() as pm:
        t0 = time.time()
        df = pq.read_table(DATA).to_pandas()
        rec["full_load_s"] = time.time() - t0
    rec["full_load_peak_rss_gb"] = pm.peak / 2**30
    sub = df.iloc[:200_000].copy()
    del df

    def cascade(row):
        if row["is_coinjoin_like"]:
            return 4
        if row["is_batch_payment"]:
            return 3
        if row["is_distribution"]:
            return 2
        if row["is_consolidation"]:
            return 1
        if row["is_peer_to_peer"]:
            return 0
        return 5

    t0 = time.time()
    y_row = sub.apply(cascade, axis=1).to_numpy()
    rec["rowwise_label_200k_s"] = time.time() - t0
    t0 = time.time()
    y_vec = np.full(len(sub), 5, dtype=np.int8)
    for code, flag in FLAGS:
        y_vec[sub[flag].to_numpy()] = code
    rec["vectorised_label_200k_s"] = time.time() - t0
    rec["labels_identical"] = bool(np.array_equal(y_row, y_vec))
    emit("mainnet_naive.jsonl", rec)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "scale"
    if mode == "smoke":
        run_size(20_000, "mainnet_smoke.jsonl")
    elif mode == "audit":
        run_audit()
    elif mode == "leakage":
        run_leakage()
    elif mode == "naive":
        run_naive()
    elif mode == "size":
        run_size(int(sys.argv[2]))
    elif mode == "scale":
        for n in SIZES:
            subprocess.run([sys.executable, "-u", __file__, "size", str(n)], check=True)
