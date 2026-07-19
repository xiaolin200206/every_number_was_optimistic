# Every Number Was Optimistic — Including Ours

**Cross-Platform Measurement of Edge Detection on Battery-Powered Agricultural Nodes**

Code, data, and reproduction scripts for the paper submitted to
*IEEE Transactions on Industrial Cyber-Physical Systems*.

---

## What this paper found

Three numbers commonly used to plan edge-AI deployments — FLOP counts,
published board power, and the expectation that quantisation buys speed —
were each optimistic on a battery-powered durian-orchard inspection node.

| Proxy | Expected | Measured | Gap |
|---|---|---|---|
| FLOPs → latency | Proportional | Up to 39.1% error | LLC-miss predictor: 10.0% max error |
| Published board power | 6.8–8.8 W | 12.34 W at battery | 1.40–1.81× |
| INT8 quantisation | Faster | 1.87× slower | No int8 conv kernel on ARM |
| Energy minimum | 4 threads | 3 threads | 4th core is a net energy loss |

We identified shared-cache contention as the mechanism behind the latency
result on Platform A, then tested it on Platform B — and the mechanism
**reversed**:

| | Platform A (RPi5) | Platform B (Jetson Orin Nano) |
|---|---|---|
| 1→4 thread DRAM traffic | **+13.5%** | **−10.8%** |
| 4-thread speedup | 2.68× | 3.79× |
| Parallel efficiency | 67.0% | 94.7% |
| Dominant effect | Cache contention | Data sharing |

A core-pinning experiment confirmed that the net direction depends on how the
microarchitecture's replacement policy handles shared vs. private cache lines —
not on cache size.

## Repository structure

```
.
├── README.md                          # this file
├── REPRODUCE.md                       # step-by-step: every table & figure
│
├── code/
│   ├── cache_benchmark_pi.py          # measurement harness, Platform A (RPi5)
│   ├── cache_benchmark_jetson.py      # measurement harness, Platform B (Jetson)
│   ├── jetson_port.diff               # exact diff (worker function untouched)
│   ├── collect.py                     # aggregate CSVs into summary tables
│   ├── cross_platform.py              # cross-platform comparison analysis
│   └── worksets.py                    # per-layer working sets from ONNX graphs
│
├── data/
│   ├── platform_a/                    # all Platform A (RPi5) CSVs
│   │   ├── yolo11n_fp32.csv           # model sweep: yolo11n, 640×640, 4 threads
│   │   ├── yolo11s_fp32.csv
│   │   ├── yolo11m_fp32.csv
│   │   ├── yolo11l_fp32.csv
│   │   ├── yolo11x_n50.csv            # model sweep: yolo11x, 640×640, 4 threads
│   │   ├── yolo11m_t1.csv             # thread sweep: 1 thread
│   │   ├── yolo11m_t2.csv             # thread sweep: 2 threads
│   │   ├── yolo11m_t3.csv             # thread sweep: 3 threads
│   │   ├── yolo11m_t4.csv             # thread sweep: 4 threads
│   │   ├── yolo11m_pwr_t1.csv         # power sweep: 1 thread (on battery)
│   │   ├── yolo11m_pwr_t2.csv         # power sweep: 2 threads
│   │   ├── yolo11m_pwr_t3.csv         # power sweep: 3 threads
│   │   ├── yolo11m_pwr_t4.csv         # power sweep: 4 threads
│   │   ├── yolo11m_1088.csv           # iso-FLOP: yolo11m at 1088×1088
│   │   ├── yolo11n_1152.csv           # iso-FLOP: yolo11n at 1152×1152
│   │   ├── fp32_smoke.csv             # quantisation: FP32 baseline
│   │   └── int8_smoke.csv             # quantisation: INT8
│   │
│   └── platform_b/                    # all Platform B (Jetson) CSVs
│       ├── cachebench_jetson_11n.csv   # model sweep: yolo11n
│       ├── cachebench_jetson_11s.csv
│       ├── cachebench_jetson_11m.csv
│       ├── cachebench_jetson_11l.csv
│       ├── cachebench_jetson_11x.csv
│       ├── cachebench_jetson_smoke.csv
│       ├── cachebench_orin_t1.csv      # thread sweep: 1 thread, cores 0-3
│       ├── cachebench_orin_t2.csv      # thread sweep: 2 threads
│       ├── cachebench_orin_t3.csv      # thread sweep: 3 threads
│       ├── cachebench_orin_t4.csv      # thread sweep: 4 threads
│       ├── cachebench_pin_0123.csv     # core pinning: cores 0,1,2,3 (one L3)
│       └── cachebench_pin_0145.csv     # core pinning: cores 0,1,4,5 (two L3)
│
└── figures/                           # generated from data/
    ├── fig1_flops_vs_latency.png
    ├── fig2_thread_sweep.png
    ├── fig3_core_pinning.png
    ├── fig4_power_energy.png
    ├── fig5_bandwidth.png
    ├── fig6_predictor_comparison.png
    ├── fig7_traffic_ratio.png
    └── fig8_amdahl.png
```

## Platforms

| | Platform A | Platform B |
|---|---|---|
| Board | Raspberry Pi 5 | Jetson Orin Nano Super |
| SoC | BCM2712 | Tegra Orin |
| CPU | 4× Cortex-A76 @ 2.4 GHz | 6× Cortex-A78AE @ 1.344 GHz |
| L2 | 512 kB per core | 256 kB per core |
| L3 | 1× 2 MB shared | 2× 2 MB independent blocks |
| DRAM | LPDDR4X-4267 | LPDDR5-3199 |
| UPS | Waveshare UPS HAT (E) | Waveshare UPS Module (C) |
| Power sensor | INA219 via I2C @ 0x2D | INA219 via I2C @ 0x2D |
| ORT version | 1.27.0 | 1.23.0 |

Both platforms use ONNX Runtime with `CPUExecutionProvider` only (no GPU, no
NPU). The same ONNX model files and the same 299 field images are used on both.

## Quick start

### Reproduce the tables from raw data

No hardware required. The CSVs in `data/` are the complete raw output of the
measurement harness.

```bash
# Install dependencies
pip install pandas scipy matplotlib

# Verify every number in the paper
python3 code/collect.py --pi-dir data/platform_a/ --jetson-dir data/platform_b/
```

### Run the harness yourself

#### Platform A (Raspberry Pi 5)

```bash
# Lock CPU frequency (required — the harness refuses otherwise)
for c in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor; do
    echo performance | sudo tee $c > /dev/null
done

# Model sweep (one model at a time)
python3 code/cache_benchmark_pi.py \
    --mode combined --combined yolo11s.onnx \
    --imgs imgs/ --size 640 --threads 4 \
    --trials 5 --n 50 --tag yolo11s_fp32

# Thread sweep
for t in 1 2 3 4; do
    python3 code/cache_benchmark_pi.py \
        --mode combined --combined yolo11m.onnx \
        --imgs imgs/ --size 640 --threads $t \
        --trials 5 --n 50 --tag yolo11m_t${t}
done

# Power sweep (unplug Type-C first!)
for t in 1 2 3 4; do
    python3 code/cache_benchmark_pi.py \
        --mode combined --combined yolo11m.onnx \
        --imgs imgs/ --size 640 --threads $t \
        --trials 5 --n 50 --power --tag yolo11m_pwr_t${t}
done
```

#### Platform B (Jetson Orin Nano)

```bash
# Lock CPU frequency
sudo jetson_clocks

# Model sweep
sudo -E PYTHONPATH=$HOME/.local/lib/python3.10/site-packages \
    PERF_BIN=/usr/lib/linux-tools-5.15.0-186/perf \
    python3 code/cache_benchmark_jetson.py \
    --mode combined --combined yolo11l.onnx \
    --imgs imgs/ --size 640 --threads 4 \
    --trials 5 --n 50 --tag jetson_11l

# Thread sweep (pinned to cores 0-3, one L3 block)
for t in 1 2 3 4; do
    sudo -E PYTHONPATH=$HOME/.local/lib/python3.10/site-packages \
        PERF_BIN=/usr/lib/linux-tools-5.15.0-186/perf \
        python3 code/cache_benchmark_jetson.py \
        --mode combined --combined yolo11l.onnx \
        --imgs imgs/ --size 640 --threads $t --cores 0,1,2,3 \
        --trials 5 --n 50 --tag orin_t${t}
done

# Core-pinning experiment
for cores in 0,1,2,3 0,1,4,5; do
    sudo -E PYTHONPATH=$HOME/.local/lib/python3.10/site-packages \
        PERF_BIN=/usr/lib/linux-tools-5.15.0-186/perf \
        python3 code/cache_benchmark_jetson.py \
        --mode combined --combined yolo11l.onnx \
        --imgs imgs/ --size 640 --threads 4 --cores $cores \
        --trials 5 --n 50 --tag pin_${cores//,/}
done
```

## CSV schema

Every CSV row is one trial. The columns are self-documenting:

| Column | Unit | Description |
|---|---|---|
| `trial` | — | Trial number (1-indexed) |
| `lat_mean` | ms | Mean per-image latency |
| `lat_med` | ms | Median per-image latency |
| `lat_p95` | ms | 95th percentile latency |
| `lat_max` | ms | Maximum single-image latency |
| `l2d_cache` | count | L2 data cache accesses (PMU) |
| `l2d_cache_refill` | count | L2 data cache refills (PMU) |
| `l3d_cache` | count | L3 data cache accesses (PMU) |
| `ll_cache_miss_rd` | count | Last-level cache read misses (PMU) |
| `peak_temp` | °C | Peak die temperature during trial |
| `throttle_bits` | bitmask | Live throttle state (0 = clean) |
| `n_inf` | count | Total inferences in this trial |
| `p_idle_w` | W | Idle board power (before trial) |
| `p_active_w` | W | Mean board power during inference |
| `energy_j` | J | Total energy (trapezoidal integral) |
| `energy_per_img_j` | J | Gross energy per image |
| `vbat_start_v` | V | Pack voltage at trial start |
| `vbat_end_v` | V | Pack voltage at trial end |
| `on_battery` | 0/1 | Both witnesses confirmed battery |
| `governor` | string | CPU frequency governor in effect |
| `cores` | string | Core pinning (Platform B only) |
| `ort_threads` | int | ONNX Runtime intra_op_num_threads |
| `ort_version` | string | ONNX Runtime version |

## Methodological safeguards

The harness enforces the following, documented in the code comments:

1. **Frequency lock** — refuses to start unless every core is pinned
   (`performance` governor on Pi; `jetson_clocks` on Jetson)
2. **Thermal equalisation** — cooldown loop before each trial (target < 55 °C)
3. **Live throttle monitoring** — current state only, not sticky history bits
4. **Power isolation** — I2C sampling runs in the parent process, outside the
   `perf stat` wrapper, so I2C traffic does not contaminate the counted region
5. **Battery verification** — two independent witnesses (current < 0 AND VBUS = 0 mW)
6. **Per-row provenance** — governor, thread count, ORT version, pack voltage,
   and SoC% are written into every CSV row

## Known issue in the raw data

The `cores` field in Platform B CSVs contains unquoted commas (e.g. `0,1,2,3`),
which breaks standard CSV parsers. Read with a fixed-width parser or quote the
field before loading. The first 14 columns (through `rounds`) are unaffected.

## Dependencies

**Measurement** (on the target platform):
- Python 3.10+
- onnxruntime (1.23+ on Jetson, 1.27+ on Pi)
- opencv-python, numpy
- smbus2 (Pi only, for power measurement)
- linux-tools / perf (for PMU counters)

**Analysis** (any machine):
- Python 3.10+
- pandas, scipy, matplotlib

## Citation

```bibtex
@article{shan2026optimistic,
  author  = {Shan, Lin Ding},
  title   = {Every Number Was Optimistic --- Including Ours:
             Cross-Platform Measurement of Edge Detection
             on Battery-Powered Agricultural Nodes},
  journal = {IEEE Trans. Ind. Cyber-Phys. Syst.},
  year    = {2026},
  note    = {Submitted}
}
```

## License

Code: MIT. Data: CC BY 4.0.
