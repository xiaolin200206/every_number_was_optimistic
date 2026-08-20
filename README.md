# Cross-platform cache, power and endurance measurement for battery-powered edge inference nodes

Measurement harness, raw hardware-counter logs, analysis scripts and a
verification script for a study of on-node neural inference running from a
battery on two Arm platforms.

This repository is **self-describing and venue-neutral**. Every quantity is
indexed by the claim it supports, stated in full, rather than by a table or
section number — so it does not need to be edited when the accompanying
manuscript is reformatted, retitled or submitted elsewhere.

---

## What was measured

Three quantities routinely used to provision a battery-powered inference node
were compared against measurements on the deployed hardware:

| Provisioning proxy | Expectation | Measured |
|---|---|---|
| Operation count → latency | proportional | up to **38.8%** error |
| Published board power | 6.8–8.8 W | **12.34 W** at the battery |
| Reduced-precision compression | faster | **1.87× slower** under this runtime |

A shared-cache mechanism proposed on the first platform was then stated as a
prediction and tested on the second, where it **reversed**:

| | Platform A (Cortex-A76) | Platform B (Cortex-A78AE) |
|---|---|---|
| 1→4 thread memory traffic | **+13.5%** | **−10.8%** |
| 4-thread speedup | 2.68× | 3.79× |
| Parallel efficiency | 67.0% | 94.7% |
| Dominant effect | capacity contention | shared-data reuse |

Both platforms have a 2 MB last-level-cache block; they differ in core
microarchitecture. A core-pinning experiment confirms the direction is a
property of the microarchitecture rather than of cache size.

---

## Claim index

Each row states a claim in full, names the files it comes from, and gives the
computation. `verify_tables.py` checks all of them and reports the count.

### Latency prediction

| ID | Claim | Data | Computation |
|---|---|---|---|
| **C1** | A predictor proportional to operation count has max error 38.8% (A) and 32.8% (B) over a 30× arithmetic range | model-sweep CSVs, both platforms | through-origin least squares on GFLOPs vs `lat_mean` |
| **C2** | A predictor proportional to model file size has max error 19.4% (A) and 27.1% (B) | as C1 + `data/worksets.json` (`weight_MB`) | through-origin least squares |
| **C3** | A predictor proportional to last-level-cache misses per frame has max error 10.0% (A) and 20.3% (B); under leave-one-out cross-validation 10.1% and 20.3% | as C1 | through-origin least squares on `ll_cache_miss_rd / n_inf` |
| **C4** | Fixing effective bandwidth at its measured mean (2.98 GB/s on A) gives a parameter-free roofline predictor with max error 7.9% | as C1 | `latency = (LLC/frame × 64 B) / BW_eff` |
| **C5** | Effective bandwidth varies by only 13.5% (A) and 21.8% (B) across a 30× range in arithmetic — the memory-bounded regime | as C1 | `(LLC/frame × 64 B) / latency_s` |

### Arithmetic is not the operative variable

| ID | Claim | Data | Computation |
|---|---|---|---|
| **C6** | With operation counts matched to within 0.83% and 2.05%, latency still differs by 18.2% and 44.9% | `yolo11x_n50` vs `yolo11m_1088`; `yolo11s_fp32` vs `yolo11n_1152` | see note on GFLOP precision below |
| **C7** | Platform B's traffic advantage widens from 7.3% to 24.5% with model size, and tracks the fraction of layers exceeding a 2 MB working set (Spearman ρ = −1.0, exact two-sided *P* = 0.017, *n* = 5) | model-sweep CSVs + `worksets.json` (`frac_layers_over_L3`) | ratio of `LLC/frame`; `scipy.stats.spearmanr` |

### Parallel scaling and its reversal

| ID | Claim | Data | Computation |
|---|---|---|---|
| **C8** | On Platform A, 4 threads give 2.68× speedup at 67.0% efficiency while memory traffic rises 13.5%; the L3 miss rate rises 24.5% → 34.4% and L2 refills rise 23.2% | `yolo11m_t{1..4}.csv` | ratios of trial means |
| **C9** | The back-solved Amdahl serial fraction rises monotonically (0.078, 0.123, 0.164) — the signature of contention, not of a fixed serial section | as C8 | `s = (t/speedup − 1)/(t − 1)` |
| **C10** | Bandwidth is not saturated: the 4-thread workload sustains 3.08 GB/s | as C8 | as C5 |
| **C11** | On Platform B, 4 threads give 3.79× at 94.7% efficiency while traffic **falls** 10.8%; L2 accesses differ by 0.04%, confirming identical work | `cachebench_orin_t{1..4}.csv` | as C8 |
| **C12** | Four threads confined to one cache block generate 6.3% **less** traffic than four spread across two blocks — opposite to the capacity prediction (two-sample *t*-test *P* = 4.7 × 10⁻⁶, Cohen's *d* = −6.9, *n* = 5 per configuration) | `cachebench_pin_0123.csv`, `cachebench_pin_0145.csv` | `scipy.stats.ttest_ind`; pooled-SD effect size |

### Compression, power and thermal

| ID | Claim | Data | Computation |
|---|---|---|---|
| **C13** | Dynamic INT8 compresses the model 3.83× (37.9 → 9.9 MB) and runs 1.87× slower under this runtime; cache misses rise 10% and L2 refills 26% | `fp32_smoke.csv`, `int8_smoke.csv` | ratios of trial means (2 trials per condition) |
| **C14** | Power at the battery is 12.339 W at 4 threads — 1.40–1.81× the 6.8–8.8 W published for the bare board | `yolo11m_pwr_t4.csv` | `mean(p_active_w)` |
| **C15** | Energy per frame is minimised at 3 threads; the 4th thread buys 10.3% lower latency for 12.4% more power, raising energy 1.19% and peak die temperature 3.6 °C (70.7 → 74.3 °C) | `yolo11m_pwr_t{1..4}.csv` | `mean(energy_per_img_j)`, `mean(peak_temp)` |
| **C16** | The thread sweep and the power-instrumented sweep, run hours apart, agree to 0.008% (1 thread) and 0.41% (4 threads) in latency and 0.4% in traffic | `yolo11m_t*.csv` vs `yolo11m_pwr_t*.csv` | pairwise relative difference |

### Derived quantity

| ID | Claim | Data | Computation |
|---|---|---|---|
| **C17** | On the node's 72 Wh pack, endurance is 10.5, 7.9, 6.6 and 5.8 h at 1–4 threads; provisioning from published board power (6.8–8.8 W) instead promises 10.6–8.2 h | `yolo11m_pwr_t{1..4}.csv` | `T = E_batt / P_node` |

**C17 is computed, not measured.** It has exactly two inputs: the measured
power at the battery (C14) and the pack capacity, 72 Wh. Endurance is linear
in both, so the **ratio** between configurations — four threads deliver 56% of
the endurance of one — depends on the measured power alone and is independent
of pack capacity entirely. Likewise, the factor by which a published power
figure overstates endurance is exactly the factor by which it understates
power (C14), so no deployment model intervenes. Where a survey length appears
in the accompanying manuscript it is a marker for reading these hours against,
not an input to them.

### Not reproducible from these logs

| ID | Quantity | Why |
|---|---|---|
| **C18** | Streaming-copy bandwidth, 11.49 GB/s | Produced by a separate memory microbenchmark, not by the inference harness. It is used only to rule out bandwidth saturation in C10, where the measured 3.08 GB/s is the operative number. |

C18 is the **only** reported quantity not recomputed from the released logs.

---

## Data availability

**Released:** all measurement code, raw CSV counter logs (29 files across both
platforms), analysis scripts, verification script, figure-generation code and
the two-point probe.

**Not released:** the 299 field images used as inference input. They are part
of a commercial deployment. They are not required to verify any claim above —
every claim except C18 is recomputed from the released CSVs by
`verify_tables.py`. Anyone running the harness on their own hardware may
substitute any image set of equivalent size; the harness makes no assumption
about image content.

---

## Repository layout

```
.
├── README.md              # this file — claim index and quick start
├── REPRODUCE.md           # per-claim reproduction walkthrough
├── CITATION.cff           # citation metadata
├── verify_tables.py       # recomputes the key numbers (44 checks)
├── verify_claims.py       # independent recomputation of every claim (167 checks)
├── fix_cores_field.py     # one-off data-hygiene repair (see note below)
│
├── code/
│   ├── cache_benchmark_pi.py       # harness, Platform A
│   ├── cache_benchmark_jetson.py   # harness, Platform B
│   ├── jetson_port.diff            # exact diff between the two (worker untouched)
│   ├── collect.py                  # aggregate CSVs into summary tables
│   ├── cross_platform.py           # cross-platform comparison
│   ├── worksets.py                 # per-layer working sets from ONNX graphs
│   ├── export_models.py            # export .pt → .onnx
│   ├── quantize.py                 # dynamic INT8 quantisation
│   ├── make_paper_figs.py          # regenerate every figure from data/
│   └── probe.py                    # two-point contention/sharing probe
│
├── data/
│   ├── worksets.json               # pre-computed per-layer working sets
│   ├── platform_a/                 # 17 CSVs
│   └── platform_b/                 # 12 CSVs
│
└── figures/                        # Fig1–Fig5 (PNG + PDF), ExtFig1–ExtFig3 (PNG)
```

### Note on CSV parsing

The `cores` column of the Platform B logs holds a comma-separated core list.
In an earlier revision it was written **unquoted**, so affected rows carried
more fields than the header and `pandas.read_csv()` misaligned every column to
its right — silently, with no error raised. The files are now quoted per
RFC 4180 and parse correctly with a plain `pd.read_csv()`.
`fix_cores_field.py` is the idempotent repair that was applied; it is retained
so the change is auditable, and reports "no malformed rows found" on the
current data.

---

## Platforms

| | Platform A | Platform B |
|---|---|---|
| Board | Raspberry Pi 5 | Jetson Orin Nano Super |
| SoC | BCM2712 | Tegra (Orin) |
| CPU | 4× Cortex-A76 @ 2.4 GHz | 6× Cortex-A78AE @ 1.344 GHz |
| L2 | 512 kB per core | 256 kB per core |
| L3 | 1 × 2 MB shared | 2 × 2 MB independent blocks |
| DRAM | LPDDR4X-4267 | LPDDR5-3199 |
| UPS | Waveshare UPS HAT (E) | Waveshare UPS Module (C) |
| Power sensing | UPS HAT (E) power gauge via I²C @ 0x2D | not logged (battery supply only) |
| ONNX Runtime | 1.27.0 | 1.23.0 |

Both use `CPUExecutionProvider` only — no GPU, no NPU. Identical ONNX model
files and identical input frames on both. Frequency is pinned and verified on
every trial; every trial starts below 55 °C and carries a clean throttle
record.

---

## Quick start

### Verify the claims (no hardware required)

```bash
pip install pandas scipy matplotlib

python3 verify_tables.py     # 44 checks on the key numbers
python3 verify_claims.py     # 167 checks, independent recomputation of C1–C17
python3 code/make_paper_figs.py                 # regenerate every figure
python3 code/collect.py --pi-dir data/platform_a/ --jetson-dir data/platform_b/
```

`verify_tables.py` and `verify_claims.py` were written separately and share no
code, so their agreement is evidence rather than a shared assumption. Both are
expected to pass in full; `verify_claims.py` additionally covers the
cross-validated predictors, the effective-bandwidth spread, the Amdahl serial
fractions, the iso-FLOP derived ratios and the endurance figures.

### Probe your own hardware

The two-point probe classifies a target as contention-dominated or
sharing-dominated by comparing cache misses per frame at 1 thread and at 4.
It runs in under five minutes.

```bash
python3 code/probe.py --model yolo11m.onnx --imgs imgs/ --size 640
```

Traffic rises → contention-dominated: expect diminishing returns from extra
cores, and provision on power. Traffic falls → sharing-dominated: additional
cores are close to free.

### Run the full harness

**Platform A** — lock frequency first; the harness refuses otherwise.

```bash
for c in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor; do
    echo performance | sudo tee $c > /dev/null
done

# model sweep (one model at a time)
python3 code/cache_benchmark_pi.py --mode combined --combined yolo11s.onnx \
    --imgs imgs/ --size 640 --threads 4 --trials 3 --n 50 --tag yolo11s_fp32

# thread sweep
for t in 1 2 3 4; do
  python3 code/cache_benchmark_pi.py --mode combined --combined yolo11m.onnx \
      --imgs imgs/ --size 640 --threads $t --trials 3 --n 50 --tag yolo11m_t${t}
done

# power sweep — disconnect external supply first, or the gauge reads the wall
for t in 1 2 3 4; do
  python3 code/cache_benchmark_pi.py --mode combined --combined yolo11m.onnx \
      --imgs imgs/ --size 640 --threads $t --trials 3 --n 50 --power \
      --tag yolo11m_pwr_t${t}
done
```

**Platform B** — lock clocks with `sudo jetson_clocks`, then invoke
`code/cache_benchmark_jetson.py` with the same arguments plus `--trials 5`.
`jetson_port.diff` shows the exact difference between the two harnesses; the
worker function that performs inference is byte-identical.

---

## Requirements

**Measurement:** Python 3.10+, ONNX Runtime, `linux-tools` / `perf` for PMU
counters, root for counter access, a frequency-pinned board.

**Analysis:** Python 3.10+, pandas, scipy, matplotlib. No hardware.

---

## Revision note

Figure legends read **Platform A / Platform B**, matching the terminology used
throughout this README and `REPRODUCE.md`. An earlier revision of
`code/make_paper_figs.py` labelled them "Node A / Node B"; only the labels
changed, and every plotted value is unchanged. Both verification scripts pass
in full against the current figures and data.

A superseded figure set (`fig1_*` … `fig8_*`) and the script that produced it
have been removed. `code/make_paper_figs.py` regenerates every figure the
repository ships.

## Citation

See `CITATION.cff`. If your tooling does not read CFF:

```bibtex
@software{shan_edge_measurement,
  author  = {Shan, Lin Ding},
  title   = {Cross-platform cache, power and endurance measurement
             for battery-powered edge inference nodes},
  year    = {2026},
  url     = {https://github.com/xiaolin200206/every_number_was_optimistic}
}
```

If you are citing the findings rather than the artefact, please cite the
accompanying manuscript; contact the author for its current status.

## License

Code: MIT. Data: CC BY 4.0.
