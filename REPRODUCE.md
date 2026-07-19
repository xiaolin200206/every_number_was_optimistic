# Reproducing every table and figure

This document maps each table and figure in the paper to the exact data files
and computations that produced it. No hardware is required — the raw CSVs are
included.

All paths are relative to the repository root. Platform A = Raspberry Pi 5,
Platform B = Jetson Orin Nano Super.

---

## Table I — Model-size sweep (5 detectors, 640×640, 4 threads)

**Data files:**

| Model | Platform A | Platform B |
|---|---|---|
| yolo11n | `data/platform_a/yolo11n_fp32.csv` | `data/platform_b/cachebench_jetson_11n.csv` |
| yolo11s | `data/platform_a/yolo11s_fp32.csv` | `data/platform_b/cachebench_jetson_11s.csv` |
| yolo11m | `data/platform_a/yolo11m_fp32.csv` | `data/platform_b/cachebench_jetson_11m.csv` |
| yolo11l | `data/platform_a/yolo11l_fp32.csv` | `data/platform_b/cachebench_jetson_11l.csv` |
| yolo11x | `data/platform_a/yolo11x_n50.csv` | `data/platform_b/cachebench_jetson_11x.csv` |

**Computation:**
- Latency: `mean(lat_mean)` across trials, `±` is `std(lat_mean)`
- LLC/img: `mean(ll_cache_miss_rd / n_inf)` across trials
  - Platform A: `n_inf` = 50 (not in CSV, constant)
  - Platform B: `n_inf` = 150 (column present)
- Effective BW: `(LLC/img × 64 bytes) / (latency_seconds)` in GB/s

**Note on Platform B CSV parsing:** the `cores` field contains unquoted commas.
Parse only the first 14 columns, or use a custom reader.

---

## Table II — Iso-FLOP pairs (Platform A)

**Data files:**
- Pair A: `yolo11x_n50.csv` (196.0 GFLOPs) vs `yolo11m_1088.csv` (196.5 GFLOPs)
- Pair B: `yolo11s_fp32.csv` (21.5 GFLOPs) vs `yolo11n_1152.csv` (21.1 GFLOPs)

**Computation:** same as Table I. GFLOPs are from the YOLO11 model cards
(resolution-scaled: GFLOPs × (new_res / 640)²).

---

## Table III — Thread sweep, Platform A (yolo11m, 640×640)

**Data files:**
- `data/platform_a/yolo11m_t1.csv` (1 thread)
- `data/platform_a/yolo11m_t2.csv` (2 threads)
- `data/platform_a/yolo11m_t3.csv` (3 threads)
- `data/platform_a/yolo11m_t4.csv` (4 threads)

**Computation:**
- Speedup: `lat_t1 / lat_tN`
- Efficiency: `speedup / N × 100`
- L3 miss rate: `mean(ll_cache_miss_rd / l3d_cache) × 100`
- Traffic change 1→4: `(LLC4 − LLC1) / LLC1 × 100` = **+13.5%**

---

## Table IV — Thread sweep, Platform B (yolo11l, cores 0–3)

**Data files:**
- `data/platform_b/cachebench_orin_t1.csv` through `cachebench_orin_t4.csv`

**Computation:** same as Table III.
- Traffic change 1→4: **−10.8%** (opposite direction)

---

## Table V — Core-pinning experiment (Platform B, yolo11l, 4 threads)

**Data files:**
- `data/platform_b/cachebench_pin_0123.csv` (cores 0,1,2,3 — one L3 block)
- `data/platform_b/cachebench_pin_0145.csv` (cores 0,1,4,5 — two L3 blocks)
- Baseline: `data/platform_b/cachebench_jetson_11l.csv` (unpinned)

**Computation:**
- LLC/img as above
- Traffic difference: `(LLC_one − LLC_two) / LLC_two × 100` = **−6.3%**
- Statistical test: two-sample t-test, `p ≈ 0`, Cohen's d = −7.66

---

## Table VII — Power sweep, Platform A (yolo11m, 640×640, on battery)

**Data files:**
- `data/platform_a/yolo11m_pwr_t1.csv` through `yolo11m_pwr_t4.csv`

**Computation:**
- Active power: `mean(p_active_w)`
- Idle power: `mean(p_idle_w)`
- Energy/image: `mean(energy_per_img_j)`
- Peak temp: `max(peak_temp)`

**Replication check** (thread sweep vs power sweep, same configuration):
- Compare `yolo11m_t1.csv` vs `yolo11m_pwr_t1.csv`: latency agrees to 0.008%
- Compare `yolo11m_t4.csv` vs `yolo11m_pwr_t4.csv`: latency agrees to 0.41%

---

## Quantisation (Section V)

**Data files:**
- `data/platform_a/fp32_smoke.csv` (FP32 baseline, yolo11s)
- `data/platform_a/int8_smoke.csv` (dynamic INT8, yolo11s)

**Computation:**
- Slowdown: `mean(int8.lat_mean) / mean(fp32.lat_mean)` = **1.87×**

---

## Figures

All figures are generated from the data above. The exact values used:

### Fig. 1 — FLOPs vs. latency / LLC vs. latency
Source: Table I data. Two panels: (a) GFLOPs on x-axis, (b) LLC/img on x-axis.

### Fig. 2 — Thread sweep comparison (three panels)
Source: Tables III and IV. (a) Speedup, (b) normalised DRAM traffic, (c) efficiency.

### Fig. 3 — Core-pinning experiment
Source: Table V. Bar charts of LLC/img and latency for three configurations.

### Fig. 4 — Power and energy (three panels)
Source: Table VII. (a) Active power with published range shaded, (b) energy/image,
(c) peak temperature.

### Fig. 5 — Effective bandwidth
Source: Table I. BW = (LLC/img × 64) / latency for each model on each platform.

### Fig. 6 — FLOP vs LLC predictor error
Source: Table I. Through-origin fits: `lat = k × GFLOP` vs `lat = k × LLC/img`.
Per-model percentage error shown as grouped bars.

### Fig. 7 — Cross-platform traffic ratio
Source: Table I. (a) `LLC_B / LLC_A` per model. (b) vs fraction of layers > 2 MB
working set (from `worksets.py`).

### Fig. 8 — Amdahl serial fraction
Source: Table III. Back-solved: `s = (1/S − 1/N) / (1 − 1/N)` at N = 2, 3, 4.

---

## Verification script

```python
import pandas as pd
import numpy as np

# Example: verify Table I, Platform A, yolo11m
df = pd.read_csv('data/platform_a/yolo11m_fp32.csv')
N = 50  # inferences per trial

lat = df['lat_mean'].mean()
lat_sd = df['lat_mean'].std()
llc = df['ll_cache_miss_rd'].mean() / N
bw = (llc * 64) / (lat / 1000) / 1e9

print(f"yolo11m: {lat:.1f} ± {lat_sd:.1f} ms, "
      f"LLC/img = {llc/1e6:.2f} M, BW = {bw:.2f} GB/s")
# Expected: 963.6 ± 2.7 ms, LLC/img = 46.25 M, BW = 3.07 GB/s
```
