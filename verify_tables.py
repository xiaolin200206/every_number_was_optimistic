#!/usr/bin/env python3
"""
verify_tables.py — Reproduce every key number from the paper.

Run from the repository root:
    python3 verify_tables.py

Requires: pandas, scipy, numpy
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

DIR_A = os.path.join(os.path.dirname(__file__), "data", "platform_a")
DIR_B = os.path.join(os.path.dirname(__file__), "data", "platform_b")

N_A = 50    # inferences per trial, Platform A
N_B = 150   # inferences per trial, Platform B


def read_b(path):
    """Read Platform B CSV, handling unquoted commas in 'cores' field."""
    rows = []
    with open(path) as f:
        header = f.readline().strip().split(",")
        for line in f:
            fields = line.strip().split(",")
            row = {}
            for i, col in enumerate(header[:14]):
                try:
                    row[col] = float(fields[i]) if fields[i] else np.nan
                except (ValueError, IndexError):
                    row[col] = fields[i]
            rows.append(row)
    return pd.DataFrame(rows)


def section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def check(label, computed, expected, tol_pct=0.5):
    diff_pct = abs(float(computed) - float(expected)) / max(abs(float(expected)), 1e-9) * 100
    ok = diff_pct < tol_pct
    mark = "✓" if ok else "✗"
    print(f"  {mark} {label:40s}  computed={computed:>10.3f}  paper={expected:>10.3f}")
    return 1 if ok else 0


def main():
    passed = 0
    total = 0

    # ============================================================
    section("TABLE I: Model-size sweep (640×640, 4 threads)")
    # ============================================================
    models_a = {"n": "yolo11n_fp32", "s": "yolo11s_fp32", "m": "yolo11m_fp32",
                "l": "yolo11l_fp32", "x": "yolo11x_n50"}
    models_b = {"n": "cachebench_jetson_11n", "s": "cachebench_jetson_11s",
                "m": "cachebench_jetson_11m", "l": "cachebench_jetson_11l",
                "x": "cachebench_jetson_11x"}

    expected_lat_a = {"n": 139.0, "s": 353.4, "m": 963.6, "l": 1229.4, "x": 2475.1}
    expected_lat_b = {"n": 140.4, "s": 367.7, "m": 1034.4, "l": 1326.3, "x": 2774.6}

    for m in ["n", "s", "m", "l", "x"]:
        df = pd.read_csv(os.path.join(DIR_A, f"{models_a[m]}.csv"))
        lat = df["lat_mean"].mean()
        r = check(f"A yolo11{m} latency", lat, expected_lat_a[m])
        passed += r; total += 1

        df = read_b(os.path.join(DIR_B, f"{models_b[m]}.csv"))
        lat = df["lat_mean"].mean()
        r = check(f"B yolo11{m} latency", lat, expected_lat_b[m])
        passed += r; total += 1

    # ============================================================
    section("TABLE III: Thread sweep, Platform A (yolo11m)")
    # ============================================================
    expected_sp = {1: 1.000, 2: 1.855, 3: 2.407, 4: 2.681}
    expected_llc = {1: 40.85, 2: 38.27, 3: 39.73, 4: 46.38}

    lat_t1 = None
    for t in [1, 2, 3, 4]:
        df = pd.read_csv(os.path.join(DIR_A, f"yolo11m_t{t}.csv"))
        lat = df["lat_mean"].mean()
        llc = df["ll_cache_miss_rd"].mean() / N_A / 1e6
        if t == 1:
            lat_t1 = lat
        sp = lat_t1 / lat

        r1 = check(f"A t={t} speedup", sp, expected_sp[t])
        r2 = check(f"A t={t} LLC/img (M)", llc, expected_llc[t])
        passed += r1 + r2; total += 2

    # Traffic change
    df1 = pd.read_csv(os.path.join(DIR_A, "yolo11m_t1.csv"))
    df4 = pd.read_csv(os.path.join(DIR_A, "yolo11m_t4.csv"))
    llc1 = df1["ll_cache_miss_rd"].mean() / N_A
    llc4 = df4["ll_cache_miss_rd"].mean() / N_A
    pct = (llc4 - llc1) / llc1 * 100
    r = check("A traffic change 1→4 (%)", pct, 13.5, tol_pct=1.0)
    passed += r; total += 1

    # ============================================================
    section("TABLE IV: Thread sweep, Platform B (yolo11l, cores 0-3)")
    # ============================================================
    expected_sp_b = {1: 1.000, 2: 1.984, 3: 2.895, 4: 3.789}

    lat_t1 = None
    for t in [1, 2, 3, 4]:
        df = read_b(os.path.join(DIR_B, f"cachebench_orin_t{t}.csv"))
        lat = df["lat_mean"].mean()
        if t == 1:
            lat_t1 = lat
        sp = lat_t1 / lat
        r = check(f"B t={t} speedup", sp, expected_sp_b[t])
        passed += r; total += 1

    # Traffic change
    df1 = read_b(os.path.join(DIR_B, "cachebench_orin_t1.csv"))
    df4 = read_b(os.path.join(DIR_B, "cachebench_orin_t4.csv"))
    llc1 = df1["ll_cache_miss_rd"].mean() / N_B
    llc4 = df4["ll_cache_miss_rd"].mean() / N_B
    pct = (llc4 - llc1) / llc1 * 100
    r = check("B traffic change 1→4 (%)", pct, -10.8, tol_pct=1.0)
    passed += r; total += 1

    # ============================================================
    section("TABLE V: Core pinning, Platform B")
    # ============================================================
    df_one = read_b(os.path.join(DIR_B, "cachebench_pin_0123.csv"))
    df_two = read_b(os.path.join(DIR_B, "cachebench_pin_0145.csv"))
    llc_one = df_one["ll_cache_miss_rd"].mean() / N_B / 1e6
    llc_two = df_two["ll_cache_miss_rd"].mean() / N_B / 1e6
    r1 = check("one-block LLC/img (M)", llc_one, 42.46)
    r2 = check("two-block LLC/img (M)", llc_two, 45.32)
    passed += r1 + r2; total += 2

    # Statistical significance
    v1 = df_one["ll_cache_miss_rd"].values / N_B
    v2 = df_two["ll_cache_miss_rd"].values / N_B
    _, pval = stats.ttest_ind(v1, v2)
    print(f"  → t-test p-value: {pval:.2e} (paper: 4.7e-6)")
    sp_pool = np.sqrt(((len(v1) - 1) * v1.std(ddof=1) ** 2
                       + (len(v2) - 1) * v2.std(ddof=1) ** 2)
                      / (len(v1) + len(v2) - 2))
    cohen_d = (v1.mean() - v2.mean()) / sp_pool
    r = check("Cohen's d (pooled sample SD)", cohen_d, -6.85, tol_pct=1.0)
    passed += r; total += 1

    # ============================================================
    section("TABLE VII: Power sweep, Platform A")
    # ============================================================
    expected_pwr = {1: 6.881, 2: 9.119, 3: 10.979, 4: 12.339}
    expected_epj = {1: 17.729, 2: 12.684, 3: 11.707, 4: 11.846}

    for t in [1, 2, 3, 4]:
        df = pd.read_csv(os.path.join(DIR_A, f"yolo11m_pwr_t{t}.csv"))
        pwr = df["p_active_w"].mean()
        epj = df["energy_per_img_j"].mean()
        r1 = check(f"t={t} active power (W)", pwr, expected_pwr[t])
        r2 = check(f"t={t} energy/img (J)", epj, expected_epj[t])
        passed += r1 + r2; total += 2

    # ============================================================
    section("QUANTISATION: yolo11s FP32 vs INT8")
    # ============================================================
    fp = pd.read_csv(os.path.join(DIR_A, "fp32_smoke.csv"))["lat_mean"].mean()
    i8 = pd.read_csv(os.path.join(DIR_A, "int8_smoke.csv"))["lat_mean"].mean()
    r = check("INT8 slowdown", i8 / fp, 1.87, tol_pct=1.0)
    passed += r; total += 1

    # ============================================================
    section("LLC-MISS PREDICTOR (Section III-C)")
    # ============================================================
    gflops = np.array([6.5, 21.5, 68.0, 86.9, 194.9])
    lat_a = np.array([139.0, 353.4, 963.6, 1229.4, 2475.1])
    llc_a = np.array([6.20, 15.66, 46.25, 56.27, 124.48])

    k_flop = np.sum(gflops * lat_a) / np.sum(gflops ** 2)
    err_flop = np.max(np.abs(k_flop * gflops - lat_a) / lat_a * 100)
    k_llc = np.sum(llc_a * lat_a) / np.sum(llc_a ** 2)
    err_llc = np.max(np.abs(k_llc * llc_a - lat_a) / lat_a * 100)

    r1 = check("FLOP predictor max error (%)", err_flop, 38.8)
    r2 = check("LLC predictor max error (%)", err_llc, 10.0)
    passed += r1 + r2; total += 2

    # ============================================================
    section("TABLE VII: mean of per-trial peak temperatures")
    # ============================================================
    expected_temp = {1: 57.3, 2: 64.5, 3: 70.7, 4: 74.3}
    for t in [1, 2, 3, 4]:
        df = pd.read_csv(os.path.join(DIR_A, f"yolo11m_pwr_t{t}.csv"))
        r = check(f"t={t} mean peak temp (°C)", df["peak_temp"].mean(),
                  expected_temp[t], tol_pct=1.0)
        passed += r; total += 1

    # ============================================================
    section("FIG. 4b: traffic ratio vs working-set fraction (worksets.json)")
    # ============================================================
    import json
    ws = json.load(open(os.path.join(os.path.dirname(__file__),
                                     "data", "worksets.json")))
    models_pairs = [("n", "yolo11n_fp32", "cachebench_jetson_11n"),
                    ("s", "yolo11s_fp32", "cachebench_jetson_11s"),
                    ("m", "yolo11m_fp32", "cachebench_jetson_11m"),
                    ("l", "yolo11l_fp32", "cachebench_jetson_11l"),
                    ("x", "yolo11x_n50", "cachebench_jetson_11x")]
    ratios, fracs = [], []
    for m, fa, fb in models_pairs:
        la = pd.read_csv(os.path.join(DIR_A, f"{fa}.csv"))["ll_cache_miss_rd"].mean() / N_A
        lb = read_b(os.path.join(DIR_B, f"{fb}.csv"))["ll_cache_miss_rd"].mean() / N_B
        ratios.append(lb / la)
        fracs.append(ws[f"yolo11{m}"]["frac_layers_over_L3"])
    rho, _ = stats.spearmanr(ratios, fracs)
    r = check("Spearman rho (ratio vs fraction)", rho, -1.0, tol_pct=0.5)
    passed += r; total += 1
    print("  → exact two-sided p for |rho|=1, n=5: 2/120 = 0.017")

    # ============================================================
    section("REPLICATION CHECK: thread sweep vs power sweep")
    # ============================================================
    df_t1 = pd.read_csv(os.path.join(DIR_A, "yolo11m_t1.csv"))
    df_p1 = pd.read_csv(os.path.join(DIR_A, "yolo11m_pwr_t1.csv"))
    diff = abs(df_t1["lat_mean"].mean() - df_p1["lat_mean"].mean()) / df_t1["lat_mean"].mean() * 100
    r = check("t=1 latency agreement (%)", diff, 0.008, tol_pct=100)
    passed += r; total += 1
    print(f"  → {diff:.3f}% (paper: 0.008%)")

    # ============================================================
    section(f"SUMMARY: {passed}/{total} checks passed")
    # ============================================================
    if passed == total:
        print("  All numbers verified. ✓")
    else:
        print(f"  {total - passed} check(s) failed. Investigate above.")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
