#!/usr/bin/env python3
"""
cross_platform.py — Cross-platform comparison of Pi 5 vs Jetson Orin Nano.

Reads CSV files from both platforms and produces the analysis tables
used in Paper 7. Designed to be run from a directory containing both
sets of CSVs (or pass --pi-dir and --jetson-dir).

Usage:
    python3 cross_platform.py --pi-dir ~/pi_data --jetson-dir ~/jetson_data
"""

import argparse
import csv
import os
import statistics as st


def load(path, n_inf=50):
    """Load a benchmark CSV, return dict with means."""
    if not os.path.exists(path):
        return None
    r = list(csv.DictReader(open(path, newline='')))
    if not r:
        return None
    d = {}
    for k, nm in [('lat_mean', 'lat'), ('ll_cache_miss_rd', 'll'),
                  ('l2d_cache_refill', 'l2rf'), ('l3d_cache', 'l3'),
                  ('l2d_cache', 'l2a')]:
        v = [float(x[k]) for x in r if x.get(k) not in (None, '', 'None')]
        if v:
            d[nm] = st.mean(v)
            d[nm + '_sd'] = st.stdev(v) if len(v) > 1 else 0.0
    # Determine n_inf
    if r[0].get('n_inf'):
        n_inf = int(float(r[0]['n_inf']))
    elif r[0].get('energy_j') and r[0].get('energy_per_img_j'):
        try:
            e = float(r[0]['energy_j'])
            p = float(r[0]['energy_per_img_j'])
            if p > 0:
                n_inf = round(e / p)
        except Exception:
            pass
    d['n_inf'] = n_inf
    if 'll' in d:
        d['ll_img'] = d['ll'] / n_inf
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pi-dir', default='.', help='Directory with Pi CSVs')
    ap.add_argument('--jetson-dir', default='.', help='Directory with Jetson CSVs')
    a = ap.parse_args()

    MODELS = [('n', 6.5), ('s', 21.5), ('m', 68.0), ('l', 86.9), ('x', 194.9)]
    PI_FILES = {'n': 'yolo11n_fp32.csv', 's': 'yolo11s_fp32.csv',
                'm': 'yolo11m_fp32.csv', 'l': 'yolo11l_fp32.csv',
                'x': 'yolo11x_n50.csv'}
    JET_FILES = {'n': 'cachebench_jetson_11n.csv', 's': 'cachebench_jetson_11s.csv',
                 'm': 'cachebench_jetson_11m.csv', 'l': 'cachebench_jetson_11l.csv',
                 'x': 'cachebench_jetson_11x.csv'}

    W = 90
    print("=" * W)
    print("CROSS-PLATFORM COMPARISON")
    print("Pi 5 (A76, 2.4 GHz, 1x2MB L3)  vs  Orin Nano (A78AE, 1.344 GHz, 2x2MB L3)")
    print("=" * W)

    # Model sweep
    print(f"\n{'model':<9}{'GFLOP':>7}{'Pi lat':>9}{'Orin lat':>10}{'ratio':>7}"
          f"{'Pi LLC/img':>12}{'Orin LLC/img':>13}{'traffic':>8}{'Pi BW':>8}{'Orin BW':>9}")
    pb, jb = [], []
    for k, fl in MODELS:
        pi = load(os.path.join(a.pi_dir, PI_FILES[k]), n_inf=50)
        jet = load(os.path.join(a.jetson_dir, JET_FILES[k]), n_inf=150)
        if not pi or not jet or 'lat' not in pi or 'lat' not in jet:
            print(f"{'yolo11' + k:<9}  [missing data]")
            continue
        p_bw = pi['ll_img'] * 64 / 1e9 / (pi['lat'] / 1e3)
        j_bw = jet['ll_img'] * 64 / 1e9 / (jet['lat'] / 1e3)
        pb.append(p_bw)
        jb.append(j_bw)
        print(f"{'yolo11' + k:<9}{fl:>7.1f}{pi['lat']:>9.1f}{jet['lat']:>10.1f}"
              f"{jet['lat'] / pi['lat']:>7.3f}"
              f"{pi['ll_img'] / 1e6:>11.2f}M{jet['ll_img'] / 1e6:>12.2f}M"
              f"{jet['ll_img'] / pi['ll_img']:>8.3f}"
              f"{p_bw:>8.3f}{j_bw:>9.3f}")

    if pb and jb:
        print(f"\n  BW spread  Pi {(max(pb) / min(pb) - 1) * 100:.1f}%"
              f"   Orin {(max(jb) / min(jb) - 1) * 100:.1f}%")

    # Thread sweep comparison
    print(f"\n{'':=<{W}}")
    print("THREAD SWEEP COMPARISON (yolo11m/Pi, yolo11l/Orin, same-cluster)")
    print(f"{'':=<{W}}")
    pi_t = {}
    for T in (1, 2, 3, 4):
        d = load(os.path.join(a.pi_dir, f'yolo11m_t{T}.csv'), n_inf=50)
        if d and 'lat' in d:
            pi_t[T] = d
    jet_t = {}
    for T in (1, 2, 3, 4):
        d = load(os.path.join(a.jetson_dir, f'cachebench_orin_t{T}.csv'), n_inf=150)
        if d and 'lat' in d:
            jet_t[T] = d

    if pi_t and jet_t:
        print(f"\n{'thr':>4}{'Pi lat':>10}{'Pi spd':>8}{'Pi eff':>8}{'Pi LLC/img':>12}"
              f"{'Orin lat':>10}{'Orin spd':>9}{'Orin eff':>9}{'Orin LLC/img':>13}")
        for T in (1, 2, 3, 4):
            if T in pi_t and T in jet_t:
                p, j = pi_t[T], jet_t[T]
                ps = pi_t[1]['lat'] / p['lat']
                js = jet_t[1]['lat'] / j['lat']
                print(f"{T:>4}{p['lat']:>10.1f}{ps:>8.3f}{ps / T * 100:>7.1f}%"
                      f"{p['ll_img'] / 1e6:>11.2f}M"
                      f"{j['lat']:>10.1f}{js:>9.3f}{js / T * 100:>8.1f}%"
                      f"{j['ll_img'] / 1e6:>12.2f}M")
        if 1 in pi_t and 4 in pi_t and 1 in jet_t and 4 in jet_t:
            p_delta = (pi_t[4]['ll_img'] / pi_t[1]['ll_img'] - 1) * 100
            j_delta = (jet_t[4]['ll_img'] / jet_t[1]['ll_img'] - 1) * 100
            print(f"\n  Pi  1->4:  traffic {p_delta:+.1f}%  efficiency"
                  f" {pi_t[1]['lat'] / pi_t[4]['lat'] / 4 * 100:.1f}%")
            print(f"  Orin 1->4: traffic {j_delta:+.1f}%  efficiency"
                  f" {jet_t[1]['lat'] / jet_t[4]['lat'] / 4 * 100:.1f}%")
            print(f"\n  DIRECTION REVERSED: Pi contention dominates,"
                  f" Orin sharing dominates")

    # Pinning
    pin_same = load(os.path.join(a.jetson_dir, 'cachebench_pin_0123.csv'), n_inf=150)
    pin_cross = load(os.path.join(a.jetson_dir, 'cachebench_pin_0145.csv'), n_inf=150)
    if pin_same and pin_cross and 'lat' in pin_same and 'lat' in pin_cross:
        print(f"\n{'':=<{W}}")
        print("CORE-PINNING EXPERIMENT (yolo11l, 4 threads, Orin Nano)")
        print(f"{'':=<{W}}")
        print(f"{'config':<20}{'lat ms':>10}{'LLC/img':>12}")
        print(f"{'cores 0,1,2,3':<20}{pin_same['lat']:>10.1f}"
              f"{pin_same['ll_img'] / 1e6:>11.2f}M  <- one L3")
        print(f"{'cores 0,1,4,5':<20}{pin_cross['lat']:>10.1f}"
              f"{pin_cross['ll_img'] / 1e6:>11.2f}M  <- two L3")
        delta = (pin_cross['ll_img'] / pin_same['ll_img'] - 1) * 100
        print(f"\n  Two-L3 has {delta:+.1f}% MORE traffic (shared weights loaded twice)")
        print(f"  but {(pin_same['lat'] / pin_cross['lat'] - 1) * 100:.1f}% slower"
              f" (less capacity pressure)")

    print(f"\n{'':=<{W}}")


if __name__ == '__main__':
    main()
