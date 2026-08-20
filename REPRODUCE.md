# Reproducing every claim

This document walks through each claim in the README's claim index, naming the
exact files and the exact computation. No hardware is required — the raw CSVs
are included.

Claims are identified by **ID (C1–C18)**, not by table or section number, so
this document stays valid regardless of how the accompanying manuscript is
organised.

All paths are relative to the repository root.
Platform A = Raspberry Pi 5 (Cortex-A76). Platform B = Jetson Orin Nano Super
(Cortex-A78AE).

Run everything at once:

```bash
python3 verify_tables.py
```

---

## Conventions used throughout

| Quantity | Computation |
|---|---|
| Latency | `mean(lat_mean)` across trials; `±` is `std(lat_mean, ddof=1)` |
| Cache misses per frame | `mean(ll_cache_miss_rd) / n_inf` |
| `n_inf` | Platform A: 50 (constant, not in CSV). Platform B: 150 (column `n_inf`) |
| Effective bandwidth | `(misses_per_frame × 64 bytes) / (latency_ms / 1000)`, in GB/s |
| L3 miss rate | `ll_cache_miss_rd / l3d_cache` |
| Through-origin fit | `k = Σ(x·y) / Σ(x²)`; error `= (k·x − y) / y` |

Cache line size is 64 B on both platforms, so one last-level-cache miss
corresponds to 64 bytes moved from DRAM.

**Parsing.** All CSVs are RFC 4180 compliant and load with a plain
`pd.read_csv()`. (Earlier revisions had an unquoted comma-separated `cores`
field on Platform B that caused silent column misalignment; see
`fix_cores_field.py`.)

---

## Model sweep — source data for C1–C5, C7

| Model | GFLOPs @640 | Platform A | Platform B |
|---|---|---|---|
| yolo11n | 6.5 | `data/platform_a/yolo11n_fp32.csv` | `data/platform_b/cachebench_jetson_11n.csv` |
| yolo11s | 21.5 | `data/platform_a/yolo11s_fp32.csv` | `data/platform_b/cachebench_jetson_11s.csv` |
| yolo11m | 68.0 | `data/platform_a/yolo11m_fp32.csv` | `data/platform_b/cachebench_jetson_11m.csv` |
| yolo11l | 86.9 | `data/platform_a/yolo11l_fp32.csv` | `data/platform_b/cachebench_jetson_11l.csv` |
| yolo11x | 194.9 | `data/platform_a/yolo11x_n50.csv` | `data/platform_b/cachebench_jetson_11x.csv` |

GFLOP values are those of the fused, deployed ONNX graphs and match the
published model cards. All runs are 640×640, 4 threads.

### C1–C3 — predictor comparison

Fit each predictor through the origin on all five models, then take the worst
per-model relative error:

```python
import numpy as np
def max_error(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    k = (x * y).sum() / (x * x).sum()
    return np.abs((k * x - y) / y).max() * 100
```

| Predictor | x | Platform A | Platform B |
|---|---|---|---|
| Operation count (C1) | GFLOPs | 38.8% | 32.8% |
| Model file size (C2) | `worksets.json → weight_MB` | 19.4% | 27.1% |
| Cache misses (C3) | misses per frame | 10.0% | 20.3% |

Coefficients of determination for the through-origin fits: 0.9964 (cache) vs
0.9918 (arithmetic) on A; 0.9990 vs 0.9966 on B.

**Leave-one-out cross-validation (C3).** For each model in turn, refit `k` on
the other four and predict the held-out model. Max error becomes 10.1% (A) and
20.3% (B); the arithmetic baseline is unchanged at 38.8% and 32.8%, because a
single-parameter through-origin model has almost no capacity to overfit five
points.

### C4 — parameter-free roofline variant

Compute effective bandwidth for each of the five models on Platform A, take
the mean (2.98 GB/s), then predict
`latency = (misses_per_frame × 64) / BW_eff`. Max error 7.9%.

### C5 — near-constant bandwidth

Spread is `(max/min − 1) × 100` over the five models: 13.5% on A, 21.8% on B,
across a 30× range in arithmetic. This is what places the workload in the
memory-bounded regime and licenses C4.

### C7 — cross-platform traffic scaling

Per-model ratio of misses per frame, B over A. The advantage widens from 7.3%
on yolo11n to 24.5% on yolo11x. Correlate that ratio against
`frac_layers_over_L3` from `data/worksets.json`:

```python
from scipy.stats import spearmanr
spearmanr(ratio, frac)     # rho = -1.0
```

With *n* = 5 the exact two-sided *P* for |ρ| = 1 is 2/120 = 0.017. Note that
`frac_layers_over_L3` is **not** monotonic in model size (yolo11m 0.748 exceeds
yolo11l 0.730) and the traffic ratio tracks that same non-monotonic ordering —
which is the point of the test.

---

## C6 — matched-arithmetic pairs

| Pair | Reference (640) | Comparison | GFLOPs | Match | Latency difference |
|---|---|---|---|---|---|
| 1 | `yolo11x_n50.csv` | `yolo11m_1088.csv` | 194.90 vs 196.52 | 0.83% | 18.2% |
| 2 | `yolo11s_fp32.csv` | `yolo11n_1152.csv` | 21.50 vs 21.06 | 2.05% | 44.9% |

**GFLOP precision matters here.** The comparison values are exact quadratic
scalings of the 640 baselines:

```
yolo11m @1088 = 68.0 × (1088/640)² = 196.52
yolo11n @1152 =  6.5 × (1152/640)² =  21.06
```

Both match percentages are relative to the 640-resolution reference model
(194.90 and 21.50). Rounding the comparison values to one decimal place
(196.5, 21.1) does **not** reproduce 0.83% and 2.05% — report two decimals
wherever these percentages appear.

Supporting figures: weights differ 2.8× and 3.6× while traffic differs only
1.24× and 1.65×, so the traffic is activation-dominated, not weight-dominated.

---

## C8–C10 — Platform A thread sweep

Files: `data/platform_a/yolo11m_t{1,2,3,4}.csv` (yolo11m @640, 3 trials each).

| Threads | Speedup | Efficiency | Misses/frame | L3 miss rate |
|---|---|---|---|---|
| 1 | 1.000 | 100.0% | 40.85 M | 24.5% |
| 2 | 1.855 | 92.7% | 38.27 M | 26.0% |
| 3 | 2.407 | 80.2% | 39.73 M | 28.6% |
| 4 | 2.681 | 67.0% | 46.38 M | 34.4% |

Traffic change 1→4 threads: **+13.5%**. L2 refills rise 23.2%. The L3 miss
rate rise from 24.5% to 34.4% is a 40% relative increase.

**C9** — back-solve Amdahl's law, `s = (t/speedup − 1)/(t − 1)`, giving 0.078,
0.123, 0.164 at 2, 3, 4 threads. Under Amdahl's law `s` is a constant; a
monotonic rise indicates contention rather than a fixed serial section.

**C10** — the 4-thread workload sustains 3.08 GB/s, well below the streaming
ceiling, so bandwidth saturation is excluded. (The ceiling itself is C18 and
is not reproducible from these logs.)

---

## C11 — Platform B thread sweep

Files: `data/platform_b/cachebench_orin_t{1,2,3,4}.csv` (yolo11l, threads
pinned to cores 0–3 = one 2 MB block, 5 trials each).

| Threads | Speedup | Efficiency | Misses/frame |
|---|---|---|---|
| 1 | 1.000 | 100.0% | 46.95 M |
| 2 | 1.984 | 99.2% | 43.01 M |
| 3 | 2.895 | 96.5% | 42.14 M |
| 4 | 3.789 | 94.7% | 41.88 M |

Traffic change 1→4 threads: **−10.8%** — opposite in sign to C8, on the same
cache size. `l2d_cache` (total work) differs by 0.04% between 1 and 4 threads,
confirming the workload is identical.

---

## C12 — core pinning

Files: `cachebench_pin_0123.csv` (one block), `cachebench_pin_0145.csv` (two
blocks), and `cachebench_jetson_11l.csv` (unpinned baseline). yolo11l,
4 threads, 5 trials each.

| Configuration | Cache available | Latency | Misses/frame |
|---|---|---|---|
| cores 0,1,2,3 | 2 MB | 1357.3 ms | 42.46 M |
| cores 0,1,4,5 | 2 + 2 MB | 1336.3 ms | 45.32 M |
| unpinned | scheduler | 1326.3 ms | 44.60 M |

One block generates **6.3% less** traffic than two — opposite to what a
capacity mechanism predicts.

```python
from scipy.stats import ttest_ind
ttest_ind(one_block, two_block)     # P = 4.7e-6
```

Cohen's *d* = −6.9 using the pooled sample standard deviation.

---

## C13 — compression

Files: `data/platform_a/fp32_smoke.csv`, `int8_smoke.csv` (yolo11s, 2 trials
each — fewer than elsewhere; the effect is large relative to the trial SDs of
3.6 and 1.0 ms).

| | Size | Latency |
|---|---|---|
| FP32 | 37.9 MB | 350.6 ± 3.6 ms |
| Dynamic INT8 | 9.9 MB | 657.2 ± 1.0 ms |

Compression 3.83×; slowdown 1.87×. Cache misses rise 10%, L2 refills 26%.
This is a property of the runtime's kernel coverage for this format on this
target, not of the numerical format in general.

---

## C14–C16 — power and thermal

Files: `data/platform_a/yolo11m_pwr_t{1,2,3,4}.csv`. Measured at the battery
pack, upstream of the buck converter, so the figures include conversion
losses, the active cooler and the power module's quiescent draw.

| Threads | Latency | Active W | Idle W | Energy/frame | Peak °C |
|---|---|---|---|---|---|
| 1 | 2587.9 ms | 6.881 | 4.672 | 17.729 J | 57.3 |
| 2 | 1399.2 ms | 9.119 | 4.788 | 12.684 J | 64.5 |
| 3 | 1080.1 ms | 10.979 | 4.767 | 11.707 J | 70.7 |
| 4 | 969.3 ms | 12.339 | 4.782 | 11.846 J | 74.3 |

**C14** — 12.339 W against a published bare-board range of 6.8–8.8 W is
1.40–1.81×.

**C15** — energy per frame is minimised at 3 threads. The 4th thread: latency
−10.3%, power +12.4%, energy +1.19%, peak temperature +3.6 °C.

Peak temperature is the **mean of per-trial peak die temperatures**, not the
maximum across trials.

**C16 — replication.** The thread sweep (C8) and this power sweep were run
hours apart with no shared state. Latency agrees to 0.008% at 1 thread and
0.41% at 4; cache traffic to 0.4%.

---

## C17 — endurance

Not a measurement. Computed from C14's measured power and the pack capacity:

```
T = E_batt / P_node          E_batt = 72 Wh
```

| Threads | Power | Endurance |
|---|---|---|
| 1 | 6.881 W | 10.46 h |
| 2 | 9.119 W | 7.90 h |
| 3 | 10.979 W | 6.56 h |
| 4 | 12.339 W | 5.84 h |

For comparison, provisioning from the published bare-board range gives 10.59 h
(6.8 W) or 8.18 h (8.8 W) on the same pack — the overstatement factor being
exactly the power ratio of C14, 1.40–1.81×.

**What depends on what.** Only *P*_node is measured. Endurance is linear in
both inputs, so the ratio between configurations (four threads deliver 56% of
one thread's endurance) is independent of pack capacity, and the overstatement
factor is independent of both. Substitute your own pack capacity by
proportion.

The model assumes the node draws active power continuously while operating.
Perception time enters endurance only through *P*_node; whether it *also*
constrains a given deployment depends on the survey, and is checked separately
against the measured frame times (3 frames take 2.9 s at four threads, 7.8 s
at one).

## C18 — the one number not reproducible here

The streaming-copy bandwidth of 11.49 GB/s comes from a separate memory
microbenchmark, not from the inference harness, and no log for it is included.
It appears only in the argument that bandwidth saturation is excluded (C10),
where the operative measured number is the 3.08 GB/s the workload actually
sustains. Every other reported quantity is recomputed from the released CSVs.
