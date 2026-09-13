"""
Elliptic experiments for the ICISPD 2026 GraphReg manuscript.

Builds a k-way partition of the Elliptic transaction graph, measures each cluster
pair with a spectral discrepancy certificate for sparse epsilon-regularity, derives
partition-context (PC) node features, and evaluates illicit-transaction detection
under the standard temporal split (train: time steps 1-34, test: 35-49).

Run with:  C:\\ProgramData\\anaconda3\\python.exe elliptic_partition_context.py
"""
import os
import json
import time
import threading
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds, LinearOperator
from scipy.stats import mannwhitneyu, spearmanr
import psutil
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, accuracy_score)
import xgboost as xgb
import torch

warnings.filterwarnings("ignore")

ROOT = r"E:\DY Patil IEEE Conf"
ELL = os.path.join(ROOT, "datasets", "elliptic")
OUT = os.path.join(ROOT, "PuneCon", "ICISPD_2026", "results")
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "elliptic_log.txt")

N_LOCAL = 93          # local features after the time-step column
SEEDS = [0, 1, 2, 3, 4]
K_GRID = [8, 16, 32]
EPS_GRID = [0.1, 0.25, 0.5, 1.0]
TRAIN_LAST, VAL_LAST = 34, 27


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


class PeakMemory:
    """Samples process RSS in a background thread."""
    def __init__(self, interval=0.05):
        self.interval, self.peak, self._stop = interval, 0, False
        self.proc = psutil.Process()

    def __enter__(self):
        self._stop = False
        self.peak = self.proc.memory_info().rss
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


# ----------------------------------------------------------------------------- data
def load_elliptic():
    feats = pd.read_csv(os.path.join(ELL, "elliptic_txs_features.csv"), header=None)
    classes = pd.read_csv(os.path.join(ELL, "elliptic_txs_classes.csv"), dtype={"class": str})
    edges = pd.read_csv(os.path.join(ELL, "elliptic_txs_edgelist.csv"))
    tx = feats[0].to_numpy(np.int64)
    ts = feats[1].to_numpy(np.int32)
    X = feats.iloc[:, 2:].to_numpy(np.float64)
    pos = pd.Series(np.arange(len(tx)), index=tx)
    cls = classes.set_index("txId").loc[tx, "class"].to_numpy()
    y = np.where(cls == "1", 1, np.where(cls == "2", 0, -1)).astype(np.int8)
    src = pos.loc[edges["txId1"].to_numpy()].to_numpy()
    dst = pos.loc[edges["txId2"].to_numpy()].to_numpy()
    return tx, ts, X, y, src, dst


# ------------------------------------------------------------------ regularity analysis
def residual_sigma1(blk, ni, nj):
    """Largest singular value of A - dJ for one cluster pair, computed matrix-free."""
    d = blk.nnz / (ni * nj)
    if min(ni, nj) <= 2 or ni * nj <= 400:
        return float(np.linalg.norm(blk.toarray() - d, 2))
    ones_i, ones_j = np.ones(ni), np.ones(nj)

    def mv(x):
        x = np.asarray(x).ravel()
        return blk @ x - d * ones_i * x.sum()

    def rmv(v):
        v = np.asarray(v).ravel()
        return blk.T @ v - d * ones_j * v.sum()

    op = LinearOperator((ni, nj), matvec=mv, rmatvec=rmv, dtype=np.float64)
    try:
        return float(svds(op, k=1, return_singular_vectors=False, maxiter=20000, tol=1e-6)[0])
    except Exception:
        v = np.random.default_rng(0).standard_normal(nj)
        s = 0.0
        for _ in range(300):
            u = mv(v)
            u /= np.linalg.norm(u) + 1e-12
            v = rmv(u)
            s = np.linalg.norm(v)
            v /= s + 1e-12
        return float(s)


def partition_stats(c, K, src, dst, node_mask):
    """Edge densities and spectral irregularity for every ordered cluster pair,
    using only nodes in node_mask and edges among them."""
    m = node_mask[src] & node_mask[dst]
    s, t = src[m], dst[m]
    nodes = np.flatnonzero(node_mask)
    n, E = len(nodes), len(s)
    p = E / (n * (n - 1))
    sizes = np.bincount(c[nodes], minlength=K).astype(np.int64)
    order = nodes[np.argsort(c[nodes], kind="stable")]
    rank = np.full(len(c), -1, dtype=np.int64)
    rank[order] = np.arange(n)
    A = sp.csr_matrix((np.ones(E), (rank[s], rank[t])), shape=(n, n))
    A.sum_duplicates()
    A.data[:] = 1.0
    b = np.concatenate([[0], np.cumsum(sizes)])
    e = np.zeros((K, K))
    Dp = np.zeros((K, K))
    sigma = np.zeros((K, K))
    eta = np.zeros((K, K))
    for i in range(K):
        Ai = A[b[i]:b[i + 1]]
        for j in range(K):
            ni, nj = int(sizes[i]), int(sizes[j])
            if ni == 0 or nj == 0:
                continue
            blk = Ai[:, b[j]:b[j + 1]].tocsr()
            e[i, j] = blk.nnz
            Dp[i, j] = blk.nnz / (p * ni * nj)
            if blk.nnz == 0:
                continue
            sigma[i, j] = residual_sigma1(blk, ni, nj)
            eta[i, j] = sigma[i, j] / (p * np.sqrt(ni * nj))
    return {"n": n, "E": E, "p": p, "sizes": sizes, "e": e, "Dp": Dp, "sigma": sigma, "eta": eta}


def certificate_summary(st):
    """A pair is certified (eps, p)-regular when eta <= eps**2 (see Proposition 1)."""
    nz = st["e"] > 0
    out = {}
    for eps in EPS_GRID:
        cert = nz & (st["eta"] <= eps ** 2)
        out[str(eps)] = {
            "pairs_certified_frac": float(cert.sum() / max(nz.sum(), 1)),
            "edges_in_certified_pairs_frac": float(st["e"][cert].sum() / max(st["e"].sum(), 1)),
        }
    return out


def fit_partition(X, ts, y_mask_fit, K, seed, method, src, dst):
    """Cluster assignment for all nodes. The k-means variant is fitted on the fitting
    period only (local features, no labels); the degree variant uses rank quantiles."""
    if method == "kmeans":
        loc = X[:, :N_LOCAL]
        sc = StandardScaler().fit(loc[y_mask_fit])
        Z = sc.transform(loc)
        km = MiniBatchKMeans(n_clusters=K, random_state=seed, batch_size=8192, n_init=3,
                             max_iter=300).fit(Z[y_mask_fit])
        return km.predict(Z).astype(np.int64)
    deg = np.bincount(src, minlength=len(ts)) + np.bincount(dst, minlength=len(ts))
    rng = np.random.default_rng(seed)
    key = deg + rng.uniform(0, 0.5, len(ts))
    edges_q = np.quantile(key[y_mask_fit], np.linspace(0, 1, K + 1)[1:-1])
    return np.searchsorted(edges_q, key, side="right").astype(np.int64)


def pc_features(c, K, src, dst, n_all, st):
    """Partition-context features for every node from its own edges and the
    block statistics of the fitting-period graph."""
    A = sp.csr_matrix((np.ones(len(src)), (src, dst)), shape=(n_all, n_all))
    C = sp.csr_matrix((np.ones(n_all), (np.arange(n_all), c)), shape=(n_all, K))
    outdeg = np.asarray(A.sum(1)).ravel()
    indeg = np.asarray(A.sum(0)).ravel()
    out_prof = (A @ C).toarray() / np.maximum(outdeg, 1)[:, None]
    in_prof = (A.T @ C).toarray() / np.maximum(indeg, 1)[:, None]
    le, ld = np.log1p(st["eta"]), np.log1p(st["Dp"])
    e_eta, e_d = le[c[src], c[dst]], ld[c[src], c[dst]]
    deg = outdeg + indeg
    sum_eta = np.bincount(src, e_eta, n_all) + np.bincount(dst, e_eta, n_all)
    sum_d = np.bincount(src, e_d, n_all) + np.bincount(dst, e_d, n_all)
    max_eta = np.zeros(n_all)
    np.maximum.at(max_eta, src, e_eta)
    np.maximum.at(max_eta, dst, e_eta)
    extra = np.column_stack([np.log1p(indeg), np.log1p(outdeg), sum_eta / np.maximum(deg, 1),
                             max_eta, sum_d / np.maximum(deg, 1), le[c, c], ld[c, c]])
    names = ([f"pc_out_c{j}" for j in range(K)] + [f"pc_in_c{j}" for j in range(K)] +
             ["pc_log_indeg", "pc_log_outdeg", "pc_mean_eta", "pc_max_eta", "pc_mean_logDp",
              "pc_self_eta", "pc_self_logDp"])
    return np.hstack([out_prof, in_prof, extra]), names, max_eta


# ------------------------------------------------------------------------ models
def make_model(name, seed):
    if name == "LR":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    if name == "RF":
        return RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=seed)
    if name == "XGB":
        return xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, tree_method="hist",
                                 n_jobs=-1, random_state=seed, eval_metric="logloss")
    if name == "MLP":
        return make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=seed))
    raise ValueError(name)


def scores(y, prob):
    pred = (prob >= 0.5).astype(int)
    return {"precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "roc_auc": roc_auc_score(y, prob),
            "pr_auc": average_precision_score(y, prob),
            "micro_f1": accuracy_score(y, pred)}


def fit_predict(name, seed, Xtr, ytr, Xte):
    mdl = make_model(name, seed)
    t0 = time.time()
    mdl.fit(Xtr, ytr)
    return mdl, mdl.predict_proba(Xte)[:, 1], time.time() - t0


def run_gcn(X, y, ts, src, dst, seed, train_last, eval_mask, epochs=1000, hidden=100, lr=1e-3):
    """Two-layer GCN with the configuration reported by Weber et al. (2019)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = X.shape[0]
    tr = ts <= train_last
    Xt = torch.tensor(StandardScaler().fit(X[tr]).transform(X), dtype=torch.float32, device=dev)
    rows = np.concatenate([src, dst, np.arange(n)])
    cols = np.concatenate([dst, src, np.arange(n)])
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A.data[:] = 1.0
    dinv = 1.0 / np.sqrt(np.asarray(A.sum(1)).ravel())
    A = (sp.diags(dinv) @ A @ sp.diags(dinv)).tocoo()
    At = torch.sparse_coo_tensor(np.vstack([A.row, A.col]), A.data, (n, n),
                                 dtype=torch.float32, device=dev).coalesce()
    lin1, lin2 = torch.nn.Linear(X.shape[1], hidden), torch.nn.Linear(hidden, 2)
    net = torch.nn.ModuleList([lin1, lin2]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    idx_tr = torch.tensor(np.flatnonzero(tr & (y >= 0)), device=dev)
    yt = torch.tensor(np.maximum(y, 0).astype(np.int64), device=dev)
    w = torch.tensor([0.3, 0.7], dtype=torch.float32, device=dev)

    def forward():
        h = torch.relu(torch.sparse.mm(At, lin1(Xt)))
        return torch.sparse.mm(At, lin2(h))

    t0 = time.time()
    for _ in range(epochs):
        net.train()
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(forward()[idx_tr], yt[idx_tr], weight=w)
        loss.backward()
        opt.step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t_train = time.time() - t0
    net.eval()
    with torch.no_grad():
        prob = torch.softmax(forward(), 1)[:, 1].cpu().numpy()
    return prob[eval_mask], t_train


def summarize(runs):
    df = pd.DataFrame(runs)
    metric_cols = ["precision", "recall", "f1", "roc_auc", "pr_auc", "micro_f1", "train_time_s"]
    g = df.groupby(["stage", "model", "features"], sort=False)[metric_cols]
    return g.mean().round(4).join(g.std().round(4), rsuffix="_std").join(g.size().rename("n_runs"))


# -------------------------------------------------------------------------- main
def main():
    open(LOG, "w").close()
    results = {"meta": {}, "runs": [], "partition": {}, "enrichment": {}, "per_timestep": {}, "importance": {}}
    tx, ts, X, y, src, dst = load_elliptic()
    n = len(tx)
    same_step = float(np.mean(ts[src] == ts[dst]))
    results["meta"] = {"nodes": int(n), "edges": int(len(src)), "features": int(X.shape[1]),
                       "labeled": int((y >= 0).sum()), "illicit": int((y == 1).sum()),
                       "licit": int((y == 0).sum()), "edges_within_same_timestep": same_step,
                       "mean_degree": float(2 * len(src) / n),
                       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"}
    log(f"Loaded Elliptic: {results['meta']}")
    LF = X[:, :N_LOCAL]
    AF = X
    lab = y >= 0

    # ------------------------------------------------ 1. choose K on a validation split
    fit_v = ts <= VAL_LAST
    tr_v = lab & (ts <= VAL_LAST)
    va_v = lab & (ts > VAL_LAST) & (ts <= TRAIN_LAST)
    val_rows = []
    for K in K_GRID:
        c = fit_partition(X, ts, fit_v, K, 0, "kmeans", src, dst)
        st = partition_stats(c, K, src, dst, fit_v)
        PC, _, _ = pc_features(c, K, src, dst, n, st)
        for fname, base in [("LF+PC", LF), ("AF+PC", AF)]:
            F = np.hstack([base, PC])
            for mname in ["RF", "XGB"]:
                _, prob, _ = fit_predict(mname, 0, F[tr_v], y[tr_v], F[va_v])
                sc = scores(y[va_v], prob)
                val_rows.append({"K": K, "features": fname, "model": mname, **sc})
                log(f"[val] K={K} {fname} {mname} F1={sc['f1']:.4f}")
    val_df = pd.DataFrame(val_rows)
    K_best = int(val_df.groupby("K")["f1"].mean().idxmax())
    results["meta"]["K_validation"] = val_df.to_dict(orient="records")
    results["meta"]["K_selected"] = K_best
    log(f"Selected K={K_best} on validation (time steps {VAL_LAST + 1}-{TRAIN_LAST})")

    # ------------------------------------------------ 2. test protocol
    fit_t = ts <= TRAIN_LAST
    tr = lab & fit_t
    te = lab & (ts > TRAIN_LAST)
    base_sets = {"LF": LF, "AF": AF}
    for seed in SEEDS:
        feat_sets = dict(base_sets)
        timing = {}
        for method in ["kmeans", "degree"]:
            t0 = time.time()
            c = fit_partition(X, ts, fit_t, K_best, seed, method, src, dst)
            t1 = time.time()
            with PeakMemory() as pm:
                st = partition_stats(c, K_best, src, dst, fit_t)
            t2 = time.time()
            PC, pc_names, max_eta = pc_features(c, K_best, src, dst, n, st)
            t3 = time.time()
            timing[method] = {"cluster_s": t1 - t0, "regularity_s": t2 - t1, "pc_features_s": t3 - t2,
                              "regularity_peak_rss_mb": pm.peak / 2**20}
            tag = "PC" if method == "kmeans" else "PCdeg"
            feat_sets[f"LF+{tag}"] = np.hstack([LF, PC])
            feat_sets[f"AF+{tag}"] = np.hstack([AF, PC])
            if method == "kmeans":
                feat_sets["PC"] = PC
                if seed == 0:
                    results["partition"] = {
                        "K": K_best, "n_train_nodes": st["n"], "train_edges": st["E"], "p": st["p"],
                        "sizes": st["sizes"].tolist(), "edges": st["e"].tolist(), "Dp": st["Dp"].tolist(),
                        "sigma": st["sigma"].tolist(), "eta": st["eta"].tolist(),
                        "certificate": certificate_summary(st), "pc_feature_names": pc_names}
                    np.save(os.path.join(OUT, "elliptic_cluster_assign_seed0.npy"), c)
                    # irregularity versus label on both periods
                    enr = {}
                    for period, msk in [("train", tr), ("test", te)]:
                        me, yy = max_eta[msk], y[msk]
                        rho, p_rho = spearmanr(me, yy)
                        u, p_u = mannwhitneyu(me[yy == 1], me[yy == 0], alternative="two-sided")
                        q = pd.qcut(pd.Series(me).rank(method="first"), 4, labels=False)
                        rate = pd.Series(yy).groupby(q.values).mean().tolist()
                        enr[period] = {"spearman": float(rho), "spearman_p": float(p_rho),
                                       "mwu_p": float(p_u), "auc_max_eta": float(roc_auc_score(yy, me)),
                                       "illicit_rate_by_quartile": rate,
                                       "median_max_eta_illicit": float(np.median(me[yy == 1])),
                                       "median_max_eta_licit": float(np.median(me[yy == 0]))}
                    cl_rate = pd.Series(y[tr]).groupby(c[tr]).agg(["mean", "size"])
                    enr["cluster_illicit_rate"] = cl_rate.reset_index().to_dict(orient="records")
                    results["enrichment"] = enr
                    log(f"Enrichment: {json.dumps({k: v for k, v in enr.items() if k != 'cluster_illicit_rate'})}")
        results["meta"].setdefault("timing_by_seed", {})[str(seed)] = timing
        log(f"seed {seed} timing {timing}")

        plans = [("LR", ["LF", "AF", "LF+PC", "AF+PC"]),
                 ("RF", list(feat_sets.keys())),
                 ("XGB", list(feat_sets.keys())),
                 ("MLP", ["LF", "AF", "LF+PC", "AF+PC"])]
        for mname, fsets in plans:
            if mname == "LR" and seed != SEEDS[0]:
                continue
            for fname in fsets:
                F = feat_sets[fname]
                mdl, prob, t_fit = fit_predict(mname, seed, F[tr], y[tr], F[te])
                sc = scores(y[te], prob)
                results["runs"].append({"stage": "test", "model": mname, "features": fname, "seed": seed,
                                        "K": K_best, "n_features": int(F.shape[1]), "train_time_s": t_fit, **sc})
                log(f"[test] seed={seed} {mname:4s} {fname:9s} F1={sc['f1']:.4f} AUC={sc['roc_auc']:.4f} ({t_fit:.1f}s)")
                if seed == 0 and mname in ("RF", "XGB") and fname in ("AF", "AF+PC"):
                    pred = (prob >= 0.5).astype(int)
                    steps = ts[te]
                    results["per_timestep"][f"{mname}|{fname}"] = {
                        int(s): float(f1_score(y[te][steps == s], pred[steps == s], zero_division=0))
                        for s in np.unique(steps)}
                if seed == 0 and mname == "XGB" and fname == "AF+PC":
                    names = [f"local_{i}" for i in range(N_LOCAL)] + [f"agg_{i}" for i in range(72)] + pc_names
                    gain = mdl.get_booster().get_score(importance_type="gain")
                    imp = pd.Series({names[int(k[1:])]: v for k, v in gain.items()}).sort_values(ascending=False)
                    results["importance"] = {"top20": imp.head(20).to_dict(),
                                             "pc_gain_share": float(imp[imp.index.str.startswith("pc_")].sum() / imp.sum())}

        te_idx = np.flatnonzero(te)
        prob, t_fit = run_gcn(AF, y, ts, src, dst, seed, TRAIN_LAST, te_idx)
        sc = scores(y[te], prob)
        results["runs"].append({"stage": "test", "model": "GCN", "features": "AF", "seed": seed, "K": None,
                                "n_features": int(AF.shape[1]), "train_time_s": t_fit, **sc})
        log(f"[test] seed={seed} GCN  AF        F1={sc['f1']:.4f} AUC={sc['roc_auc']:.4f} ({t_fit:.1f}s)")

        with open(os.path.join(OUT, "elliptic_results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1, default=float)

    summ = summarize(results["runs"])
    summ.to_csv(os.path.join(OUT, "elliptic_summary.csv"))
    log("\n" + summ.to_string())
    with open(os.path.join(OUT, "elliptic_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, default=float)
    log("done")


if __name__ == "__main__":
    main()
