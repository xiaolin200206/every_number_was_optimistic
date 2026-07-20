"""Regenerate Figs. 1, 3 and 4 from the released data (run from repo root).

Fig. 1  FLOPs vs latency / LLC vs latency        -> figures/fig1_flops_vs_latency.png
Fig. 3  FLOP vs weight-size vs LLC predictor     -> figures/fig3_predictor_comparison.png
Fig. 4  Cross-platform traffic ratio             -> figures/fig4_traffic_ratio.png

GFLOPs are the fused-graph values (= Ultralytics model cards).
Weights are weight_MB from data/worksets.json.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams.update({"font.family": "serif", "font.size": 12})

A, B = "data/platform_a/", "data/platform_b/"
GFLOPS = np.array([6.5, 21.5, 68.0, 86.9, 194.9])
FILES = [("yolo11n_fp32", "cachebench_jetson_11n"),
         ("yolo11s_fp32", "cachebench_jetson_11s"),
         ("yolo11m_fp32", "cachebench_jetson_11m"),
         ("yolo11l_fp32", "cachebench_jetson_11l"),
         ("yolo11x_n50", "cachebench_jetson_11x")]
LABELS = [f"yolo11{m}" for m in "nsmlx"]


def read_b(path):
    """Platform B CSVs carry a trailing free-text governor field."""
    rows = []
    with open(path) as f:
        head = f.readline().strip().split(",")
        for line in f:
            fl = line.strip().split(",")
            rows.append({c: (float(fl[i]) if i < len(fl) else np.nan)
                         for i, c in enumerate(head[:14])})
    return pd.DataFrame(rows)


la, ll_a, lb, ll_b = [], [], [], []
for fa, fb in FILES:
    d = pd.read_csv(A + fa + ".csv")
    la.append(d.lat_mean.mean())
    ll_a.append((d.ll_cache_miss_rd / 50).mean() / 1e6)
    d = read_b(B + fb + ".csv")
    lb.append(d.lat_mean.mean())
    ll_b.append((d.ll_cache_miss_rd / 150).mean() / 1e6)
la, ll_a, lb, ll_b = map(np.array, (la, ll_a, lb, ll_b))


def fit(x, y):
    k = (x * y).sum() / (x * x).sum()
    return k, np.abs(k * x - y) / y * 100


# ---------------- Fig. 1 ----------------
kfA, _ = fit(GFLOPS, la)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.475, 3.495), dpi=200)
a1.plot(GFLOPS, la, "-o", color="#d62728", ms=9, lw=2, label="Platform A (RPi5)")
a1.plot(GFLOPS, lb, "--s", color="#1f77b4", ms=9, lw=2, label="Platform B (Jetson)")
xs = np.array([0, 200])
a1.plot(xs, kfA * xs, ":", color="#d62728", alpha=0.4, lw=2, label="Proportional (A)")
a1.set_xlabel("GFLOPs"); a1.set_ylabel("Latency (ms)")
a1.set_title("(a) FLOPs vs. Latency"); a1.legend(loc="upper left", fontsize=10)
a2.plot(ll_a, la, "-o", color="#d62728", ms=9, lw=2, label="Platform A")
a2.plot(ll_b, lb, "--s", color="#1f77b4", ms=9, lw=2, label="Platform B")
a2.set_xlabel("LLC Misses per Image (M)"); a2.set_ylabel("Latency (ms)")
a2.set_title("(b) LLC Misses vs. Latency"); a2.legend(loc="upper left", fontsize=10)
plt.tight_layout(); plt.savefig("figures/fig1_flops_vs_latency.png"); plt.close()

# ---------------- Fig. 3 ----------------
ws = json.load(open("data/worksets.json"))
wt = np.array([ws[m]["weight_MB"] for m in LABELS])
panels = []
for lat in (la, lb):
    panels.append((fit(GFLOPS, lat)[1], fit(wt, lat)[1], fit(ll_a if lat is la else ll_b, lat)[1]))
fig, axes = plt.subplots(1, 2, figsize=(9.645, 3.645), dpi=200)
xpos, w = np.arange(5), 0.26
for ax, (ef, ew, el), title in zip(axes, panels,
                                   ["(a) Platform A (RPi5)", "(b) Platform B (Jetson)"]):
    ax.bar(xpos - w, ef, w, color="#d62728", alpha=0.85, edgecolor="black",
           lw=0.8, label=f"FLOP (max {ef.max():.1f}%)")
    ax.bar(xpos, ew, w, color="#ff7f0e", edgecolor="black",
           lw=0.8, label=f"Weight-size (max {ew.max():.1f}%)")
    ax.bar(xpos + w, el, w, color="#2ca02c", edgecolor="black",
           lw=0.8, label=f"LLC-miss (max {el.max():.1f}%)")
    ax.set_xticks(xpos); ax.set_xticklabels(LABELS, fontsize=10)
    ax.set_ylim(0, 44); ax.set_ylabel("Prediction Error (%)")
    ax.set_xlabel("Model"); ax.set_title(title)
    ax.legend(fontsize=9.5, loc="upper right")
plt.tight_layout(); plt.savefig("figures/fig3_predictor_comparison.png"); plt.close()

# ---------------- Fig. 4 ----------------
ratio = ll_b * 150 / (ll_a * 50) * 50 / 150  # = ll_b / ll_a (both already per-image)
ratio = ll_b / ll_a
frac = np.array([ws[m]["frac_layers_over_L3"] for m in LABELS])
rho, _ = stats.spearmanr(ratio, frac)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.76, 3.34), dpi=200)
ax1.bar(LABELS, ratio, width=0.55, color="#1f77b4", edgecolor="black", linewidth=0.8)
for i, v in enumerate(ratio):
    ax1.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=9)
ax1.axhline(1.0, ls=":", color="0.7", lw=2)
ax1.set_ylim(0, 1.08); ax1.set_ylabel("B/A Traffic Ratio"); ax1.set_xlabel("Model")
ax1.set_title("(a) Traffic Ratio (B/A)"); ax1.tick_params(axis="x", labelsize=9)
order = np.argsort(frac)
ax2.plot(frac[order], ratio[order], "-o", color="#1f77b4", ms=8)
offsets = {"yolo11n": (6, -3), "yolo11s": (6, -3), "yolo11m": (2, -13),
           "yolo11l": (6, 2), "yolo11x": (6, -3)}
for f, r, lab in zip(frac, ratio, LABELS):
    ax2.annotate(lab, (f, r), textcoords="offset points",
                 xytext=offsets[lab], fontsize=9)
ax2.set_xlabel("Fraction of Layers > 2 MB Working Set")
ax2.set_ylabel("B/A Traffic Ratio")
ax2.set_title(f"(b) vs. Layer Working Set (\u03c1={rho:.2f})")
ax2.set_xlim(0.26, 0.84)
plt.tight_layout(); plt.savefig("figures/fig4_traffic_ratio.png"); plt.close()

print("wrote figures/fig1_flops_vs_latency.png, fig3_predictor_comparison.png, "
      "fig4_traffic_ratio.png")
