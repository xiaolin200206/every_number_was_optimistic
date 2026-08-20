"""Regenerate every display item to Nature Sensors specification.
Width <= 180 mm, 300 dpi, sans-serif labels at 6-7 pt, editable text."""
import json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") + "/"
OUT = os.path.join(R, "figures") + "/"
os.makedirs(OUT, exist_ok=True)
MM = 1/25.4
W2 = 180*MM          # double-column
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "lines.linewidth": 1.1, "lines.markersize": 3.5,
    "savefig.dpi": 300, "figure.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})
CA, CB, CG, CR, CO = "#0F6FC6", "#D1495B", "#2E933C", "#B5179E", "#E8871A"

def read_b(path):
    rows = []
    with open(R+path) as f:
        header = f.readline().strip().split(",")
        for line in f:
            fl = line.strip().split(",")
            row = {}
            for i, c in enumerate(header[:14]):
                try: row[c] = float(fl[i]) if fl[i] else np.nan
                except (ValueError, IndexError): row[c] = fl[i]
            rows.append(row)
    return pd.DataFrame(rows)

def load(p):
    return read_b(p) if "platform_b" in p else pd.read_csv(R+p)

def agg(p, n):
    d = load(p)
    return dict(lat=d.lat_mean.mean(), sd=d.lat_mean.std(ddof=1),
                llc=d.ll_cache_miss_rd.mean()/n,
                llc_sd=d.ll_cache_miss_rd.std(ddof=1)/n,
                l3=d.l3d_cache.mean()/n, d=d)

MODELS = ["yolo11n","yolo11s","yolo11m","yolo11l","yolo11x"]
LBL = ["n","s","m","l","x"]
GF = dict(zip(MODELS,[6.5,21.5,68.0,86.9,194.9]))
A = {m: agg(f"data/platform_a/{f}",50) for m,f in zip(MODELS,
     ["yolo11n_fp32.csv","yolo11s_fp32.csv","yolo11m_fp32.csv","yolo11l_fp32.csv","yolo11x_n50.csv"])}
B = {m: agg(f"data/platform_b/cachebench_jetson_{k}.csv",150) for m,k in
     zip(MODELS,["11n","11s","11m","11l","11x"])}
TA = {t: agg(f"data/platform_a/yolo11m_t{t}.csv",50) for t in (1,2,3,4)}
TB = {t: agg(f"data/platform_b/cachebench_orin_t{t}.csv",150) for t in (1,2,3,4)}
PW = {t: pd.read_csv(R+f"data/platform_a/yolo11m_pwr_t{t}.csv") for t in (1,2,3,4)}
WS = json.load(open(R+"data/worksets.json"))
Wt = {m: WS[m]["weight_MB"] for m in MODELS}

def panel(ax, s, x=-0.20, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")

def fitk(x, y):
    x, y = np.asarray(x,float), np.asarray(y,float)
    return (x*y).sum()/(x*x).sum()

# ================= Fig 1: provisioning proxies and endurance =================
E_BATT, T_ROUND = 72.0, 5.7          # 72 Wh pack; 5.7 h round used only as a marker
th = np.array([1,2,3,4])
Pact = np.array([PW[t].p_active_w.mean() for t in th])
Psd  = np.array([PW[t].p_active_w.std(ddof=1) for t in th])
lat_p = np.array([PW[t].lat_mean.mean() for t in th])
endur = E_BATT/Pact

fig, ax = plt.subplots(1, 3, figsize=(W2, 55*MM))
a = ax[0]
a.axhspan(6.8, 8.8, color="0.85", zorder=0)
a.text(1.05, 7.8, "published board-level range", fontsize=5.5, color="0.35", va="center")
a.errorbar(th, Pact, yerr=Psd, marker="o", color=CA, capsize=2, elinewidth=0.7)
a.set_xticks(th); a.set_xlabel("Threads"); a.set_ylabel("Power at battery (W)")
a.set_ylim(0, 14)
a.annotate("1.40–1.81×", xy=(4, Pact[3]), xytext=(2.6, 13.0), fontsize=6,
           arrowprops=dict(arrowstyle="->", lw=0.6, color="0.3"))
panel(a, "a")

a = ax[1]
bars = ["Published\n6.8 W", "Published\n8.8 W", "Measured\n12.34 W"]
vals = [E_BATT/6.8, E_BATT/8.8, endur[3]]
a.bar(bars, vals, color=["0.75", "0.75", CR], width=0.62, edgecolor="none")
a.axhline(T_ROUND, color=CG, lw=0.9, ls="--")
a.text(-0.52, T_ROUND+0.28, "5.7 h round", fontsize=5.8, color=CG, ha="left")
for i, v in enumerate(vals):
    a.text(i, v+0.22, f"{v:.1f} h", ha="center", fontsize=6)
a.set_ylabel("Endurance on 72 Wh pack (h)"); a.set_ylim(0, 12.5); a.set_xlim(-0.6, 3.9)
a.plot([2.32, 3.05], [vals[2], vals[2]], lw=0.6, color=CR)
a.plot([2.32, 3.05], [T_ROUND, T_ROUND], lw=0.6, color=CR)
a.annotate("", xy=(3.02, vals[2]), xytext=(3.02, T_ROUND),
           arrowprops=dict(arrowstyle="<->", lw=0.7, color=CR, shrinkA=0, shrinkB=0))
a.text(3.12, T_ROUND-1.15, "8 min\nmargin", fontsize=5.8, color=CR, va="top", ha="left")
a.annotate("", xy=(1.30, vals[1]), xytext=(1.30, T_ROUND),
           arrowprops=dict(arrowstyle="<->", lw=0.7, color="0.45"))
a.text(1.42, (vals[1]+T_ROUND)/2, "2.5 h\nmargin", fontsize=5.8, color="0.45",
       va="center", ha="left")
panel(a, "b")

a = ax[2]
a.bar(th, endur, color=CA, width=0.6, edgecolor="none")
a.axhline(T_ROUND, color="0.35", lw=0.8, ls="--")
a.text(0.62, T_ROUND-0.85, "5.7 h round", fontsize=5.8, color="0.35", ha="left")
for t, v in zip(th, endur):
    a.text(t, v+0.22, f"{v:.1f}", ha="center", fontsize=6)
a.set_xticks(th); a.set_xlabel("Threads provisioned")
a.set_ylabel("Endurance on 72 Wh pack (h)"); a.set_ylim(0, 12.5); a.set_xlim(0.4, 4.6)
a.annotate("", xy=(1.45, endur[0]+0.75), xytext=(4.0, endur[0]+0.75),
           arrowprops=dict(arrowstyle="<->", lw=0.7, color=CR))
a.text(2.72, endur[0]+1.1, f"−{(1-endur[3]/endur[0])*100:.0f}% endurance", fontsize=5.8,
       color=CR, ha="center")
a2 = a.twinx(); a2.spines["top"].set_visible(False)
a2.plot(th, 3*lat_p/1000, marker="s", ms=3, color="0.35", lw=0.9, ls="--")
a2.set_ylabel("Time for 3 frames (s)", fontsize=7); a2.set_ylim(0, 26)
a2.tick_params(labelsize=6)
a2.text(1.12, 3*lat_p[0]/1000-3.4, "perception time", fontsize=5.5, color="0.35", ha="left")
panel(a, "c")
fig.tight_layout(w_pad=2.4)
fig.savefig(OUT+"Fig1_provisioning_endurance.png", bbox_inches="tight")
fig.savefig(OUT+"Fig1_provisioning_endurance.pdf", bbox_inches="tight"); plt.close(fig)

# ================= Fig 2: memory traffic, not arithmetic =================
fig, ax = plt.subplots(1, 3, figsize=(W2, 55*MM))
gf = np.array([GF[m] for m in MODELS])
latA = np.array([A[m]["lat"] for m in MODELS]); latB = np.array([B[m]["lat"] for m in MODELS])
llcA = np.array([A[m]["llc"] for m in MODELS]); llcB = np.array([B[m]["llc"] for m in MODELS])

a = ax[0]
xs = np.linspace(0, 210, 50)
a.plot(xs, fitk(gf, latA)*xs, ls=":", color="0.4", lw=0.9)
a.plot(gf, latA, "o-", color=CA, label="Platform A (Cortex-A76)")
a.plot(gf, latB, "s-", color=CB, label="Platform B (Cortex-A78AE)")
kf = fitk(gf, latA)
i = int(np.argmax(np.abs((kf*gf-latA)/latA)))
a.annotate(f"{abs((kf*gf-latA)/latA)[i]*100:.1f}% error", xy=(gf[i], latA[i]),
           xytext=(gf[i]+18, latA[i]-430), fontsize=6,
           arrowprops=dict(arrowstyle="->", lw=0.6, color="0.3"))
for j, l in enumerate(LBL):
    a.annotate(l, (gf[j], latA[j]), textcoords="offset points", xytext=(2, -7), fontsize=5.5, color=CA)
a.set_xlabel("Arithmetic (GFLOPs)"); a.set_ylabel("Perception time (ms)")
a.legend(frameon=False, loc="upper left", handlelength=1.4)
panel(a, "a")

a = ax[1]
a.plot(llcA/1e6, latA, "o-", color=CA); a.plot(llcB/1e6, latB, "s-", color=CB)
for j, l in enumerate(LBL):
    a.annotate(l, (llcA[j]/1e6, latA[j]), textcoords="offset points", xytext=(2, -7), fontsize=5.5, color=CA)
a.set_xlabel("Cache misses per frame (millions)"); a.set_ylabel("Perception time (ms)")
panel(a, "b")

a = ax[2]
wt = np.array([Wt[m] for m in MODELS])
def errs(x, y): 
    k = fitk(x, y); return np.abs((k*np.asarray(x,float)-y)/y)*100
w = 0.26; idx = np.arange(5)
eA = [errs(gf, latA), errs(wt, latA), errs(llcA, latA)]
eB = [errs(gf, latB), errs(wt, latB), errs(llcB, latB)]
for k, (lab, col) in enumerate([("Arithmetic", CR), ("Model size", CO), ("Cache misses", CG)]):
    a.bar(idx-w+k*w, eA[k], w*0.92, color=col, edgecolor="none", label=lab)
    a.bar(idx-w+k*w, -eB[k], w*0.92, color=col, edgecolor="none", alpha=0.45)
a.axhline(0, color="0.3", lw=0.6)
a.set_xticks(idx); a.set_xticklabels(LBL); a.set_xlabel("Perception model")
a.set_ylabel("Prediction error (%)")
yl = max(max(max(e) for e in eA), max(max(e) for e in eB))*1.35
a.set_ylim(-yl, yl)
a.set_yticks(a.get_yticks()); a.set_yticklabels([f"{abs(v):.0f}" for v in a.get_yticks()])
a.text(0.02, 0.97, "Platform A", transform=a.transAxes, fontsize=5.5, va="top", color="0.3")
a.text(0.02, 0.03, "Platform B", transform=a.transAxes, fontsize=5.5, va="bottom", color="0.3")
a.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16),
         handlelength=1.0, columnspacing=1.0)
a.spines["bottom"].set_visible(False)
panel(a, "c")
fig.tight_layout(w_pad=2.2)
fig.savefig(OUT+"Fig2_traffic_not_arithmetic.png", bbox_inches="tight")
fig.savefig(OUT+"Fig2_traffic_not_arithmetic.pdf", bbox_inches="tight"); plt.close(fig)

# ================= Fig 3: thread sweep, the reversal =================
fig, ax = plt.subplots(1, 3, figsize=(W2, 55*MM))
spA = np.array([TA[1]["lat"]/TA[t]["lat"] for t in th])
spB = np.array([TB[1]["lat"]/TB[t]["lat"] for t in th])
a = ax[0]
a.plot(th, th, ls=":", color="0.5", lw=0.9); a.text(3.4, 3.75, "ideal", fontsize=5.5, color="0.5")
a.plot(th, spA, "o-", color=CA, label="Platform A"); a.plot(th, spB, "s-", color=CB, label="Platform B")
a.set_xticks(th); a.set_xlabel("Threads"); a.set_ylabel("Speedup")
a.legend(frameon=False, loc="upper left", handlelength=1.4)
panel(a, "a")

a = ax[1]
nA = np.array([TA[t]["llc"] for t in th])/TA[1]["llc"]
nB = np.array([TB[t]["llc"] for t in th])/TB[1]["llc"]
eA_ = np.array([TA[t]["llc_sd"] for t in th])/TA[1]["llc"]
eB_ = np.array([TB[t]["llc_sd"] for t in th])/TB[1]["llc"]
a.axhline(1.0, color="0.4", lw=0.7, ls="--")
a.errorbar(th, nA, yerr=eA_, marker="o", color=CA, capsize=2, elinewidth=0.7)
a.errorbar(th, nB, yerr=eB_, marker="s", color=CB, capsize=2, elinewidth=0.7)
a.set_ylim(0.86, 1.19)
a.text(3.92, nA[-1]+0.022, f"+{(nA[-1]-1)*100:.1f}%", fontsize=6, color=CA, ha="right")
a.text(3.92, nB[-1]-0.030, f"{(nB[-1]-1)*100:.1f}%", fontsize=6, color=CB, ha="right")
a.text(1.08, 1.155, "Platform A", fontsize=6, color=CA)
a.text(1.08, 0.878, "Platform B", fontsize=6, color=CB)
a.set_xticks(th); a.set_xlabel("Threads")
a.set_ylabel("Memory traffic per frame\n(normalized to 1 thread)")
panel(a, "b")

a = ax[2]
a.bar(th-0.19, spA/th*100, 0.36, color=CA, edgecolor="none", label="Platform A")
a.bar(th+0.19, spB/th*100, 0.36, color=CB, edgecolor="none", label="Platform B")
a.set_xticks(th); a.set_xlabel("Threads"); a.set_ylabel("Parallel efficiency (%)")
a.set_ylim(0, 118); a.axhline(100, color="0.5", lw=0.7, ls=":")
a.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.16),
         handlelength=1.0, columnspacing=1.2)
panel(a, "c")
fig.tight_layout(w_pad=2.2)
fig.savefig(OUT+"Fig3_scaling_reversal.png", bbox_inches="tight")
fig.savefig(OUT+"Fig3_scaling_reversal.pdf", bbox_inches="tight"); plt.close(fig)

# ================= Fig 4: core pinning =================
P1 = agg("data/platform_b/cachebench_pin_0123.csv",150)
P2 = agg("data/platform_b/cachebench_pin_0145.csv",150)
UN = B["yolo11l"]
fig, ax = plt.subplots(1, 2, figsize=(130*MM, 56*MM))
names = ["One block\ncores 0–3", "Two blocks\ncores 0,1,4,5", "Unpinned\nscheduler"]
tr = [P1["llc"]/1e6, P2["llc"]/1e6, UN["llc"]/1e6]
trsd = [P1["llc_sd"]/1e6, P2["llc_sd"]/1e6, UN["llc_sd"]/1e6]
a = ax[0]
a.bar(names, tr, yerr=trsd, color=[CG, CB, "0.72"], width=0.6, edgecolor="none",
      capsize=2, error_kw=dict(elinewidth=0.7))
a.set_ylabel("Memory traffic per frame (millions)"); a.set_ylim(0, 52)
a.annotate("", xy=(0, tr[0]+2.2), xytext=(1, tr[1]+2.2),
           arrowprops=dict(arrowstyle="<->", lw=0.7, color="0.3"))
a.text(0.5, tr[1]+4.6, f"−{(1-tr[0]/tr[1])*100:.1f}%\n$P$ = 4.7 × 10$^{{-6}}$",
       ha="center", fontsize=5.8)
panel(a, "a", x=-0.24)
a = ax[1]
lt = [P1["lat"], P2["lat"], UN["lat"]]
ltsd = [P1["sd"], P2["sd"], UN["sd"]]
a.bar(names, lt, yerr=ltsd, color=[CG, CB, "0.72"], width=0.6, edgecolor="none",
      capsize=2, error_kw=dict(elinewidth=0.7))
a.set_ylabel("Perception time (ms)"); a.set_ylim(0, 1600)
panel(a, "b", x=-0.24)
fig.tight_layout(w_pad=2.4)
fig.savefig(OUT+"Fig4_core_pinning.png", bbox_inches="tight")
fig.savefig(OUT+"Fig4_core_pinning.pdf", bbox_inches="tight"); plt.close(fig)

# ================= Fig 5: energy and thermal operating point =================
fig, ax = plt.subplots(1, 3, figsize=(W2, 55*MM))
en = np.array([PW[t].energy_per_img_j.mean() for t in th])
ensd = np.array([PW[t].energy_per_img_j.std(ddof=1) for t in th])
tp = np.array([PW[t].peak_temp.mean() for t in th])
tpsd = np.array([PW[t].peak_temp.std(ddof=1) for t in th])
a = ax[0]
a.errorbar(th, en, yerr=ensd, marker="o", color=CA, capsize=2, elinewidth=0.7)
j = int(np.argmin(en))
a.plot(th[j], en[j], "o", ms=6.5, mfc="none", mec=CG, mew=1.1)
a.annotate("energy minimum", xy=(th[j], en[j]), xytext=(2.35, 14.4), fontsize=5.8, color=CG,
           arrowprops=dict(arrowstyle="->", lw=0.6, color=CG))
a.set_xticks(th); a.set_xlabel("Threads"); a.set_ylabel("Energy per frame (J)")
panel(a, "a")
a = ax[1]
a.errorbar(th, tp, yerr=tpsd, marker="^", color=CR, capsize=2, elinewidth=0.7)
a.axhline(80, color="0.4", ls="--", lw=0.8)
a.text(1.05, 81, "vendor throttling band, 80–85 °C (never entered)", fontsize=5.5, color="0.35")
a.set_xticks(th); a.set_xlabel("Threads"); a.set_ylabel("Peak die temperature (°C)")
a.set_ylim(50, 88)
panel(a, "b")
a = ax[2]
a.bar(th, 3*lat_p/1000, color="0.65", width=0.6, edgecolor="none")
a.axhline(60, color=CO, lw=1.0, ls="--")
a.text(2.5, 60*0.55, "one-minute interval between survey units", fontsize=5.8, color=CO, ha="center")
for t, v in zip(th, 3*lat_p/1000):
    a.text(t, v*1.25, f"{v:.1f} s", ha="center", fontsize=5.8)
a.set_yscale("log"); a.set_ylim(1, 400)
a.set_xticks(th); a.set_xlabel("Threads"); a.set_ylabel("Time for 3 frames (s)")
panel(a, "c")
fig.tight_layout(w_pad=2.2)
fig.savefig(OUT+"Fig5_energy_thermal.png", bbox_inches="tight")
fig.savefig(OUT+"Fig5_energy_thermal.pdf", bbox_inches="tight"); plt.close(fig)

# ================= Extended Data =================
fig, a = plt.subplots(figsize=(88*MM, 58*MM))
bwA = llcA*64/(latA/1000)/1e9; bwB = llcB*64/(latB/1000)/1e9
a.plot(gf, bwA, "o-", color=CA, label="Platform A"); a.plot(gf, bwB, "s-", color=CB, label="Platform B")
a.set_xscale("log"); a.set_xlabel("Arithmetic (GFLOPs)")
a.set_ylabel("Effective bandwidth (GB s$^{-1}$)"); a.set_ylim(0, 5)
a.text(8, 4.4, f"spread: {(bwA.max()/bwA.min()-1)*100:.1f}% (A), {(bwB.max()/bwB.min()-1)*100:.1f}% (B)", fontsize=6)
a.legend(frameon=False, loc="lower right", handlelength=1.4)
fig.tight_layout(); fig.savefig(OUT+"ExtFig1_bandwidth.png", bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(W2*0.78, 55*MM))
ratio = llcB/llcA
frac = np.array([WS[m]["frac_layers_over_L3"] for m in MODELS])
a = ax[0]
a.bar(LBL, ratio, color=CB, width=0.6, edgecolor="none")
a.axhline(1.0, color="0.4", lw=0.7, ls="--")
a.set_ylim(0.7, 1.05); a.set_xlabel("Perception model")
a.set_ylabel("Traffic ratio, Platform B / Platform A")
panel(a, "a", x=-0.22)
a = ax[1]
a.plot(frac*100, ratio, "o", color=CB)
for j, l in enumerate(LBL):
    a.annotate(l, (frac[j]*100, ratio[j]), textcoords="offset points", xytext=(3, 3), fontsize=6)
a.set_xlabel("Layers with working set > 2 MB (%)")
a.set_ylabel("Traffic ratio, Platform B / Platform A")
a.text(0.04, 0.06, "Spearman $\\rho$ = −1.0\nexact two-sided $P$ = 0.017, $n$ = 5",
       transform=a.transAxes, fontsize=6)
panel(a, "b", x=-0.22)
fig.tight_layout(w_pad=2.4); fig.savefig(OUT+"ExtFig2_traffic_ratio.png", bbox_inches="tight"); plt.close(fig)

fig, a = plt.subplots(figsize=(88*MM, 58*MM))
ts = np.array([2,3,4]); s = np.array([(t/(TA[1]["lat"]/TA[t]["lat"])-1)/(t-1) for t in ts])
a.plot(ts, s, "o-", color=CR)
for t, v in zip(ts, s):
    a.annotate(f"{v:.3f}", (t, v), textcoords="offset points", xytext=(4, -3), fontsize=6)
a.set_xticks(ts); a.set_xlabel("Threads")
a.set_ylabel("Back-solved Amdahl serial fraction")
a.set_ylim(0, 0.20)
a.text(2.05, 0.175, "constant under Amdahl's law;\nmonotonic rise indicates contention", fontsize=6)
fig.tight_layout(); fig.savefig(OUT+"ExtFig3_amdahl.png", bbox_inches="tight"); plt.close(fig)

print("written:", sorted(os.listdir(OUT)))
print("endurance (h):", [f"{v:.2f}" for v in endur], " 3-frame (s):", [f"{v:.1f}" for v in 3*lat_p/1000])
