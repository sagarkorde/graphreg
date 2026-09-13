"""
Builds the manuscript figures and prints the summary numbers used in the tables.

Run with:  C:\\ProgramData\\anaconda3\\python.exe make_figures.py
"""
import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"E:\DY Patil IEEE Conf\PuneCon\ICISPD_2026"
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
COLORS = {"RF|AF": "#1b6ca8", "RF|AF+PC": "#6fb1e0", "XGB|AF": "#b5452c", "XGB|AF+PC": "#e6997f", "GCN|AF": "#555555"}


def read_jsonl(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def elliptic():
    path = os.path.join(RES, "elliptic_results.json")
    if not os.path.exists(path):
        print("elliptic_results.json missing")
        return
    R = json.load(open(path, encoding="utf-8"))
    runs = pd.DataFrame(R["runs"])
    g = runs.groupby(["model", "features"], sort=False)
    summ = g[["precision", "recall", "f1", "roc_auc", "pr_auc", "micro_f1", "train_time_s"]].agg(["mean", "std"]).round(4)
    summ["n"] = g.size()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 100)
    print("=== Elliptic summary (mean, std over seeds)\n", summ)
    summ.to_csv(os.path.join(RES, "elliptic_summary_table.csv"))

    # Paired differences AF+PC - AF and LF+PC - LF per seed
    for model in ["RF", "XGB", "MLP"]:
        for base in ["LF", "AF"]:
            a = runs[(runs.model == model) & (runs.features == base)].set_index("seed")["f1"]
            b = runs[(runs.model == model) & (runs.features == base + "+PC")].set_index("seed")["f1"]
            dlt = (b - a).dropna()
            if len(dlt):
                print(f"paired dF1 {model} {base}+PC-{base}: mean {dlt.mean():+.4f} sd {dlt.std():.4f} min {dlt.min():+.4f} max {dlt.max():+.4f} (n={len(dlt)})")

    part = R["partition"]
    eta = np.array(part["eta"])
    e = np.array(part["edges"])
    nz = e > 0
    print("=== Partition K", part["K"], "train nodes", part["n_train_nodes"], "edges", part["train_edges"], "p", part["p"])
    print("sizes", part["sizes"])
    for eps in [0.1, 0.25, 0.5, 1.0]:
        cert = nz & (eta <= eps ** 2)
        print(f"eps={eps}: pairs certified {cert.sum()}/{nz.sum()}  edge share {e[cert].sum() / e.sum():.4f}")
    print("eta over nonempty pairs: min %.3f median %.3f max %.3f" % (eta[nz].min(), np.median(eta[nz]), eta[nz].max()))
    print("fraction of nonempty pairs", nz.mean(), "diagonal edge share", np.trace(e) / e.sum())
    print("=== Enrichment", json.dumps({k: v for k, v in R["enrichment"].items() if k != "cluster_illicit_rate"}, indent=1))
    cl = pd.DataFrame(R["enrichment"]["cluster_illicit_rate"])
    print("cluster illicit rate (train):\n", cl.sort_values("mean", ascending=False).to_string())
    print("=== Importance", json.dumps(R["importance"], indent=1))
    tim = pd.DataFrame({s: {f"{m}.{k}": v for m, d in t.items() for k, v in d.items()}
                        for s, t in R["meta"]["timing_by_seed"].items()}).T
    print("=== Timing (mean over seeds)\n", tim.mean().round(3))
    print("K validation:\n", pd.DataFrame(R["meta"]["K_validation"]).groupby("K")["f1"].mean())

    # Figure: per-time-step F1 and eta heatmap
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 1.9), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    for key, series in R["per_timestep"].items():
        steps = sorted(int(s) for s in series)
        ax.plot(steps, [series[str(s)] for s in steps], marker="o", ms=2, lw=0.9,
                color=COLORS.get(key, "k"), label=key.replace("|", " "))
    ax.set_xticks(range(35, 50, 2))
    ax.axvline(43, color="grey", lw=0.6, ls="--")
    ax.text(43.3, 0.93, "dark-market\nshutdown", fontsize=5.5, color="grey", va="top")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Illicit F1")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=5.5, ncol=1, loc="lower left", handlelength=1.2)
    ax.set_title("(a) Test-period F1 by time step", fontsize=7)
    ax = axes[1]
    im = ax.imshow(np.log10(1 + eta), cmap="viridis")
    ax.set_xlabel("Receiver cluster $j$")
    ax.set_ylabel("Sender cluster $i$")
    ax.set_xticks(range(0, eta.shape[0], 4))
    ax.set_yticks(range(0, eta.shape[0], 4))
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"$\log_{10}(1+\eta_{ij})$", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    ax.set_title(r"(b) Spectral irregularity, $K$=%d" % eta.shape[0], fontsize=7)
    fig.savefig(os.path.join(FIG, "elliptic_timestep_eta.pdf"))
    fig.savefig(os.path.join(FIG, "elliptic_timestep_eta.png"), dpi=300)
    plt.close(fig)


def mainnet():
    audit = read_jsonl("mainnet_audit.jsonl")
    if audit:
        print("=== Audit\n", json.dumps(audit[-1], indent=1))
    leak = read_jsonl("mainnet_leakage.jsonl")
    if leak:
        df = pd.DataFrame(leak)
        print("=== Leakage\n", df[["group", "n_features", "accuracy", "macro_f1", "weighted_f1", "fit_s"]].to_string())
        for r in leak:
            print(r["group"], "per-class F1", {k: round(v, 4) for k, v in r["per_class_f1"].items()}, "top gain", {k: round(v, 3) for k, v in r["top_gain"].items()})
        print("test class counts", leak[0]["test_class_counts"])
    naive = read_jsonl("mainnet_naive.jsonl")
    if naive:
        print("=== Naive\n", json.dumps(naive[-1], indent=1))
    scale = read_jsonl("mainnet_scaling.jsonl")
    if not scale:
        return
    df = pd.DataFrame(scale).drop_duplicates("n", keep="last").sort_values("n")
    cols = ["n", "n_tokens", "read_s", "extract_s", "partition_s", "xgb_gpu_fit_s", "total_s", "peak_rss_gb",
            "throughput_read_to_partition_tx_s", "throughput_end_to_end_tx_s"]
    print("=== Scaling\n", df[cols].round(3).to_string())
    print("GPU macro-F1:", [round(m["macro_f1"], 4) for m in df["xgb_gpu_metrics"]])
    if "xgb_cpu_fit_s" in df:
        print("CPU fit s:", df[["n", "xgb_cpu_fit_s"]].dropna().to_string())
    big = df.iloc[-1]
    part = big["partition"]
    print("=== Full-corpus partition: n_addr", part["n_addr"], "incidences", part["incidences"], "p", part["p"])
    print("tx per class", part["tx_per_class"], "addr per bin", part["addr_per_bin"])
    print("log10 Dp:\n", np.round(np.log10(np.array(part["Dp"]) + 1e-12), 2))
    print("eta:\n", np.round(np.array(part["eta"]), 2))
    # log-log slope of total time
    x, yv = np.log(df["n"].to_numpy()), np.log(df["total_s"].to_numpy())
    print("log-log slope total_s vs n: %.3f" % np.polyfit(x, yv, 1)[0])
    x2 = np.log(df["n"].to_numpy()); y2 = np.log((df["read_s"] + df["extract_s"] + df["partition_s"]).to_numpy())
    print("log-log slope read+extract+partition vs n: %.3f" % np.polyfit(x2, y2, 1)[0])

    fig, axes = plt.subplots(1, 3, figsize=(4.8, 1.55), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    nm = df["n"].to_numpy() / 1e6
    ax = axes[0]
    bottom = np.zeros(len(df))
    for c, lab, colr in [("read_s", "read", "#9ecae1"), ("extract_s", "extract", "#3182bd"),
                         ("partition_s", "partition", "#fd8d3c"), ("xgb_gpu_fit_s", "XGBoost (GPU)", "#636363")]:
        ax.bar(range(len(df)), df[c], bottom=bottom, color=colr, label=lab, width=0.7)
        bottom += df[c].to_numpy()
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([f"{v:.2f}".rstrip("0").rstrip(".") for v in nm], fontsize=5.5, rotation=45)
    ax.set_xlabel("Transactions (M)")
    ax.set_ylabel("Wall time (s)")
    ax.legend(fontsize=5, loc="upper left", handlelength=1.0, borderaxespad=0.1)
    ax.set_ylim(0, bottom.max() * 1.55)
    ax.set_title("(a) Stage time", fontsize=7)
    ax = axes[1]
    ax.plot(nm, df["peak_rss_gb"], marker="o", ms=2.5, lw=0.9, color="#3182bd")
    ax.axhline(50, color="grey", lw=0.6, ls="--")
    ax.text(nm[0], 47, "50 GB limit", fontsize=5.5, color="grey", va="top")
    ax.set_xlabel("Transactions (M)")
    ax.set_ylabel("Peak RSS (GB)")
    ax.set_title("(b) Peak memory", fontsize=7)
    ax = axes[2]
    E = np.array(part["edges"])
    lift = E * E.sum() / (E.sum(1, keepdims=True) * E.sum(0, keepdims=True))
    print("log2 lift:\n", np.round(np.log2(lift), 2))
    im = ax.imshow(np.log2(lift), cmap="RdBu_r", vmin=-2.3, vmax=2.3)
    ax.set_yticks(range(6))
    ax.set_yticklabels(["P2P", "Cons.", "Distr.", "Batch", "CoinJ.", "Other"], fontsize=5.5)
    ax.set_xticks(range(6))
    ax.set_xticklabels(["1", "2", "3-5", "6-20", "21-100", ">100"], fontsize=5, rotation=45)
    ax.set_xlabel("Address activity")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"$\log_2$ lift", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    ax.set_title("(c) Class $\\times$ activity lift", fontsize=7)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(os.path.join(FIG, "mainnet_scaling_partition.pdf"))
    fig.savefig(os.path.join(FIG, "mainnet_scaling_partition.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    elliptic()
    mainnet()
