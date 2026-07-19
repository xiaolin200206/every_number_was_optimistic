#!/usr/bin/env python3
"""
collect.py — Aggregate all Paper 7 measurement CSVs into a single summary.

Run from the directory containing the CSV files (~/int8_exp/ on either platform).
Reads yolo11{n,s,m,l}_fp32.csv, yolo11x_n50.csv, yolo11m_t{1,2,3,4}.csv,
yolo11m_pwr_t{1,2,3,4}.csv, fp32_smoke.csv, int8_smoke.csv, worksets.json,
and any cachebench_jetson_*.csv files.

Usage:
    python3 collect.py
"""

import csv
import glob
import json
import os
import statistics as st


def agg(f, n=50):
    """Load a benchmark CSV and compute means/stdev for all numeric columns."""
    if not os.path.exists(f):
        return None
    r = list(csv.DictReader(open(f, newline='')))
    if not r:
        return None
    d = {'file': os.path.basename(f), 'trials': len(r)}
    for k, nm in [('lat_mean', 'lat'), ('ll_cache_miss_rd', 'll'),
                  ('l2d_cache_refill', 'l2'), ('l3d_cache', 'l3'),
                  ('l2d_cache', 'l2a'), ('peak_temp', 'temp'),
                  ('p_active_w', 'pw'), ('p_idle_w', 'pi'),
                  ('energy_per_img_j', 'ej')]:
        v = [float(x[k]) for x in r if x.get(k) not in (None, '', 'None')]
        if v:
            d[nm] = st.mean(v)
            d[nm + '_sd'] = st.stdev(v) if len(v) > 1 else 0.0
    for k in ('governor', 'on_battery', 'ort_version', 'cores'):
        if r[0].get(k):
            d[k] = r[0][k]
    # Determine n_inf: use recorded field if available, else try energy ratio, else assume n
    if r[0].get('n_inf'):
        d['n_inf'] = int(float(r[0]['n_inf']))
    elif 'ej' in d and d['ej'] > 0:
        try:
            e = float(r[0].get('energy_j', 0))
            if e > 0:
                d['n_inf'] = round(e / d['ej'])
        except Exception:
            pass
    if 'n_inf' not in d:
        d['n_inf'] = n
    if 'll' in d:
        d['ll_img'] = d['ll'] / d['n_inf']
    return d


def main():
    W = 114
    print("=" * W)
    print("PAPER 7 — MEASUREMENT SUMMARY")
    print("=" * W)

    # [1] Model sweep (Pi)
    print("\n[1] MODEL SWEEP — Pi 5 (640, 4 threads)")
    print(f"{'model':<10}{'GFLOP':>7}{'lat ms':>11}{'sd':>7}{'LLC/img':>11}"
          f"{'MB/img':>9}{'GB/s':>8}{'GF/s':>8}{'F/B':>7}{'degC':>7}")
    bw = []
    for nm, fl, f in [('yolo11n', 6.5, 'yolo11n_fp32.csv'),
                       ('yolo11s', 21.5, 'yolo11s_fp32.csv'),
                       ('yolo11m', 68.0, 'yolo11m_fp32.csv'),
                       ('yolo11l', 87.6, 'yolo11l_fp32.csv'),
                       ('yolo11x', 196.0, 'yolo11x_n50.csv')]:
        d = agg(f)
        if not d:
            print(f"{nm:<10}  [missing {f}]")
            continue
        mb = d['ll_img'] * 64 / 1e6
        b = mb / 1e3 / (d['lat'] / 1e3)
        bw.append(b)
        print(f"{nm:<10}{fl:>7.1f}{d['lat']:>11.1f}{d.get('lat_sd', 0):>7.1f}"
              f"{d['ll_img'] / 1e6:>10.2f}M{mb:>9.0f}{b:>8.3f}"
              f"{fl / (d['lat'] / 1e3):>8.1f}"
              f"{fl * 1e9 / (d['ll_img'] * 64):>7.1f}{d.get('temp', 0):>7.1f}")
    if bw:
        print(f"  bandwidth {min(bw):.2f}-{max(bw):.2f} GB/s"
              f" = {(max(bw) / min(bw) - 1) * 100:.0f}% spread")

    # [2] Thread sweep (Pi)
    print("\n[2] THREAD SWEEP — Pi 5 (yolo11m 640)")
    print(f"{'thr':>4}{'lat ms':>11}{'speedup':>9}{'eff':>7}{'LLC/img':>11}"
          f"{'L3miss':>9}{'GB/s':>8}{'per-thr':>9}{'degC':>7}")
    base = None
    for T in (1, 2, 3, 4):
        d = agg(f'yolo11m_t{T}.csv')
        if not d:
            print(f"{T:>4}  [missing]")
            continue
        base = base or d['lat']
        b = d['ll_img'] * 64 / 1e9 / (d['lat'] / 1e3)
        mr = d['ll'] / d['l3'] * 100 if 'l3' in d else 0
        print(f"{T:>4}{d['lat']:>11.1f}{base / d['lat']:>9.3f}"
              f"{base / d['lat'] / T * 100:>6.1f}%{d['ll_img'] / 1e6:>10.2f}M"
              f"{mr:>8.1f}%{b:>8.2f}{b / T:>9.3f}{d.get('temp', 0):>7.1f}")

    # [3] Power sweep (Pi)
    print("\n[3] THREAD SWEEP + POWER — Pi 5 (battery)")
    print(f"{'thr':>4}{'lat ms':>11}{'activeW':>10}{'idleW':>9}{'J/img':>9}"
          f"{'margW':>9}{'margJ':>9}{'degC':>7}{'batt':>7}")
    best = None
    for T in (1, 2, 3, 4):
        d = agg(f'yolo11m_pwr_t{T}.csv')
        if not d:
            print(f"{T:>4}  [missing]")
            continue
        mw = d.get('pw', 0) - d.get('pi', 0)
        print(f"{T:>4}{d['lat']:>11.1f}{d.get('pw', 0):>10.3f}"
              f"{d.get('pi', 0):>9.3f}{d.get('ej', 0):>9.3f}"
              f"{mw:>9.3f}{mw * d['lat'] / 1000:>9.3f}"
              f"{d.get('temp', 0):>7.1f}"
              f"{str(d.get('on_battery', '?'))[:5]:>7}")
        if d.get('ej') and (best is None or d['ej'] < best[1]):
            best = (T, d['ej'])
    if best:
        print(f"  -> minimum energy at {best[0]} threads ({best[1]:.3f} J/img)")

    # [4] ISO-FLOP (Pi)
    print("\n[4] ISO-FLOP PAIRS — Pi 5")
    print(f"{'pair':<5}{'config':<16}{'GFLOP':>8}{'wtMB':>8}{'lat ms':>11}"
          f"{'LLC/img':>11}{'GB/s':>8}")
    got = {}
    for p, nm, fl, wt, f in [
        ('A', 'yolo11x @640', 196.0, 227.8, 'yolo11x_n50.csv'),
        ('A', 'yolo11m @1088', 196.5, 80.5, 'yolo11m_1088.csv'),
        ('B', 'yolo11s @640', 21.5, 37.9, 'yolo11s_fp32.csv'),
        ('B', 'yolo11n @1152', 21.1, 10.6, 'yolo11n_1152.csv')]:
        d = agg(f)
        if not d:
            print(f"{p:<5}{nm:<16}  [missing {f}]")
            continue
        got[nm] = (d['lat'], d['ll_img'], fl)
        print(f"{p:<5}{nm:<16}{fl:>8.1f}{wt:>8.1f}{d['lat']:>11.1f}"
              f"{d['ll_img'] / 1e6:>10.2f}M"
              f"{d['ll_img'] * 64 / 1e9 / (d['lat'] / 1e3):>8.3f}")
    for a, b in (('yolo11x @640', 'yolo11m @1088'),
                 ('yolo11s @640', 'yolo11n @1152')):
        if a in got and b in got:
            lr = got[b][0] / got[a][0]
            mr = got[b][1] / got[a][1]
            print(f"  {a} vs {b}: FLOP {abs(got[b][2] / got[a][2] - 1) * 100:.2f}%"
                  f" apart, lat x{lr:.3f}, traffic x{mr:.3f},"
                  f" agree {abs(lr - mr) / mr * 100:.1f}%")

    # [5] INT8 (Pi)
    print("\n[5] INT8 — Pi 5")
    f32, i8 = agg('fp32_smoke.csv'), agg('int8_smoke.csv')
    for tag, d in (('FP32', f32), ('INT8', i8)):
        if d:
            print(f"  {tag:<6}{d['lat']:>9.1f} +/- {d.get('lat_sd', 0):.1f} ms"
                  f"   LLC/img {d['ll_img'] / 1e6:.2f}M")
        else:
            print(f"  {tag:<6}[missing]")
    if f32 and i8:
        print(f"  -> INT8 {i8['lat'] / f32['lat']:.2f}x"
              f" {'SLOWER' if i8['lat'] > f32['lat'] else 'faster'}"
              f"   LLC {(i8['ll_img'] / f32['ll_img'] - 1) * 100:+.0f}%"
              f"   L2 {(i8['l2'] / f32['l2'] - 1) * 100:+.0f}%")

    # [6] Working sets
    print("\n[6] WORKING SETS")
    if os.path.exists('worksets.json'):
        w = json.load(open('worksets.json'))
        print(f"{'model':<10}{'wtMB':>9}{'worksetMB':>12}{'layers':>8}"
              f"{'>L3':>6}{'>L3%':>8}{'measured':>10}{'re-stream':>11}")
        for k, v in w.items():
            print(f"{k:<10}{v['weight_MB']:>9.1f}{v['total_workset_MB']:>12.1f}"
                  f"{v['n_compute_layers']:>8}{v['layers_over_L3']:>6}"
                  f"{v['frac_layers_over_L3'] * 100:>7.1f}%"
                  f"{v['measured_MB_per_img']:>10}"
                  f"{v['measured_MB_per_img'] / v['total_workset_MB']:>10.2f}x")
    else:
        print("  [worksets.json missing]")

    # [7] Jetson model sweep
    jfiles = sorted(glob.glob('cachebench_jetson_11?.csv'))
    if jfiles:
        print("\n[7] MODEL SWEEP — Jetson Orin Nano (640, 4 threads)")
        print(f"{'model':<10}{'GFLOP':>7}{'lat ms':>11}{'sd':>7}{'LLC/img':>11}"
              f"{'MB/img':>9}{'GB/s':>8}{'degC':>7}")
        jbw = []
        for nm, fl, f in [('yolo11n', 6.5, 'cachebench_jetson_11n.csv'),
                           ('yolo11s', 21.5, 'cachebench_jetson_11s.csv'),
                           ('yolo11m', 68.0, 'cachebench_jetson_11m.csv'),
                           ('yolo11l', 87.6, 'cachebench_jetson_11l.csv'),
                           ('yolo11x', 196.0, 'cachebench_jetson_11x.csv')]:
            d = agg(f, n=150)
            if not d or 'lat' not in d:
                print(f"{nm:<10}  [missing or no latency in {f}]")
                continue
            mb = d['ll_img'] * 64 / 1e6
            b = mb / 1e3 / (d['lat'] / 1e3)
            jbw.append(b)
            print(f"{nm:<10}{fl:>7.1f}{d['lat']:>11.1f}{d.get('lat_sd', 0):>7.1f}"
                  f"{d['ll_img'] / 1e6:>10.2f}M{mb:>9.0f}{b:>8.3f}"
                  f"{d.get('temp', 0):>7.1f}")
        if jbw:
            print(f"  bandwidth {min(jbw):.2f}-{max(jbw):.2f} GB/s"
                  f" = {(max(jbw) / min(jbw) - 1) * 100:.0f}% spread")

    # [8] Jetson thread sweep
    tfiles = sorted(glob.glob('cachebench_orin_t?.csv'))
    if tfiles:
        print("\n[8] THREAD SWEEP — Jetson Orin Nano (yolo11l 640, cores 0-3)")
        print(f"{'thr':>4}{'lat ms':>11}{'speedup':>9}{'eff':>7}{'LLC/img':>11}")
        tbase = None
        for T in (1, 2, 3, 4):
            d = agg(f'cachebench_orin_t{T}.csv', n=150)
            if not d or 'lat' not in d:
                continue
            tbase = tbase or d['lat']
            print(f"{T:>4}{d['lat']:>11.1f}{tbase / d['lat']:>9.3f}"
                  f"{tbase / d['lat'] / T * 100:>6.1f}%"
                  f"{d['ll_img'] / 1e6:>10.2f}M")

    # [9] Jetson pinning
    pfiles = [f for f in glob.glob('cachebench_pin_*.csv')]
    if pfiles:
        print("\n[9] CORE-PINNING — Jetson Orin Nano (yolo11l 640, 4 threads)")
        print(f"{'config':<20}{'lat ms':>11}{'LLC/img':>11}{'cores':>12}")
        for tag, f in [('pin_0123', 'cachebench_pin_0123.csv'),
                       ('pin_0145', 'cachebench_pin_0145.csv')]:
            d = agg(f, n=150)
            if not d or 'lat' not in d:
                continue
            print(f"{tag:<20}{d['lat']:>11.1f}{d['ll_img'] / 1e6:>10.2f}M"
                  f"{d.get('cores', '?'):>12}")

    print("\n" + "=" * W)
    # Provenance
    prov = agg('yolo11m_pwr_t4.csv') or agg('cachebench_jetson_11x.csv', n=150) or {}
    if prov:
        print("provenance:", ", ".join(f"{k}={v}" for k, v in prov.items()
              if k in ('governor', 'ort_version', 'on_battery', 'cores')))
    print("=" * W)


if __name__ == '__main__':
    main()
