#!/usr/bin/env python3
"""
probe.py — Lightweight micro-benchmark to determine whether a new chip is
           contention-dominated or sharing-dominated.

Run a single model at 1 thread and 4 threads, compare LLC miss counts.
If traffic rises   → contention dominates (like Cortex-A76)
If traffic falls   → data sharing dominates (like Cortex-A78AE)

Total runtime: ~5 minutes on typical edge hardware.

Usage:
    sudo python3 probe.py --model yolo11m.onnx --imgs imgs/ --size 640

Output:
    One-line verdict:  CONTENTION-DOMINATED (+X.X%)  or  SHARING-DOMINATED (-X.X%)

Requires: onnxruntime, opencv-python, numpy, perf (linux-tools)
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time


def read_temp():
    for path in ['/sys/class/thermal/thermal_zone0/temp',
                 '/sys/devices/virtual/thermal/thermal_zone0/temp']:
        try:
            return int(open(path).read()) / 1000.0
        except Exception:
            continue
    return -1


def cool_to(target=55.0, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        temp = read_temp()
        if temp <= target or temp < 0:
            return temp
        print(f"  cooling... {temp:.1f}°C (target < {target}°C)", flush=True)
        time.sleep(5)
    return read_temp()


def find_perf():
    """Find a working perf binary."""
    import shutil
    candidates = [os.environ.get('PERF_BIN', '')]
    wp = shutil.which('perf')
    if wp:
        candidates.append(wp)
    candidates += sorted(glob.glob('/usr/lib/linux-tools-*/perf'), reverse=True)
    candidates += sorted(glob.glob('/usr/lib/linux-tools/*/perf'), reverse=True)

    for c in candidates:
        if not c:
            continue
        try:
            r = subprocess.run([c, '--version'], capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                return c
        except Exception:
            continue
    return 'perf'


def run_probe(model, imgs_dir, size, threads, n_imgs=30, perf_bin='perf'):
    """Run inference under perf and return (latency_ms, llc_miss_per_img)."""
    worker_code = f'''
import json, glob, os, time, sys
import numpy as np
import cv2
import onnxruntime as ort

so = ort.SessionOptions()
so.intra_op_num_threads = {threads}
so.inter_op_num_threads = 1
so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

sess = ort.InferenceSession("{model}", sess_options=so,
                            providers=["CPUExecutionProvider"])
inp_name = sess.get_inputs()[0].name

files = sorted(glob.glob(os.path.join("{imgs_dir}", "*.jpg"))
             + glob.glob(os.path.join("{imgs_dir}", "*.jpeg"))
             + glob.glob(os.path.join("{imgs_dir}", "*.png")))
files = files[:{n_imgs}]

imgs = []
for f in files:
    im = cv2.imread(f)
    if im is None: continue
    im = cv2.resize(im, ({size}, {size}))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    im = np.transpose(im, (2, 0, 1))[None, ...]
    imgs.append(np.ascontiguousarray(im))

# warm-up
for _ in range(3):
    sess.run(None, {{inp_name: imgs[0]}})

lats = []
for im in imgs:
    t0 = time.perf_counter()
    sess.run(None, {{inp_name: im}})
    lats.append((time.perf_counter() - t0) * 1000)

print(json.dumps({{"lat_mean": float(np.mean(lats)), "n": len(lats)}}))
'''
    # Write temp worker
    worker_path = '/tmp/_probe_worker.py'
    with open(worker_path, 'w') as f:
        f.write(worker_code)

    cmd = [perf_bin, 'stat', '-e', 'll_cache_miss_rd',
           sys.executable, worker_path]

    p = subprocess.run(cmd, capture_output=True, text=True)

    # Parse worker output
    lat_mean = None
    n = None
    for line in p.stdout.splitlines():
        if line.strip().startswith('{'):
            d = json.loads(line)
            lat_mean = d['lat_mean']
            n = d['n']

    # Parse perf output
    llc = None
    m = re.search(r'([\d,]+)\s+ll_cache_miss_rd', p.stderr)
    if m:
        llc = int(m.group(1).replace(',', ''))

    if lat_mean is None or llc is None:
        print(f"  ERROR: worker failed (threads={threads})")
        print(f"  stdout: {p.stdout[-200:]}")
        print(f"  stderr: {p.stderr[-200:]}")
        return None, None

    return lat_mean, llc / n


def main():
    ap = argparse.ArgumentParser(
        description='Probe: is this chip contention-dominated or sharing-dominated?')
    ap.add_argument('--model', required=True, help='ONNX model file')
    ap.add_argument('--imgs', required=True, help='Directory of test images')
    ap.add_argument('--size', type=int, default=640, help='Inference resolution')
    ap.add_argument('--n', type=int, default=30, help='Images per probe (default 30)')
    ap.add_argument('--cooldown', type=float, default=55.0, help='Max start temp (°C)')
    a = ap.parse_args()

    perf_bin = find_perf()
    print(f"perf: {perf_bin}")
    print(f"model: {a.model}")
    print(f"images: {a.imgs} (n={a.n})")
    print()

    # Probe 1: single thread
    print("--- Probe 1: 1 thread ---")
    cool_to(a.cooldown)
    lat1, llc1 = run_probe(a.model, a.imgs, a.size, 1, a.n, perf_bin)
    if lat1 is None:
        sys.exit(1)
    print(f"  latency = {lat1:.1f} ms/img, LLC misses = {llc1/1e6:.2f} M/img")

    # Probe 2: 4 threads
    print("\n--- Probe 2: 4 threads ---")
    cool_to(a.cooldown)
    lat4, llc4 = run_probe(a.model, a.imgs, a.size, 4, a.n, perf_bin)
    if lat4 is None:
        sys.exit(1)
    print(f"  latency = {lat4:.1f} ms/img, LLC misses = {llc4/1e6:.2f} M/img")

    # Verdict
    delta_pct = (llc4 - llc1) / llc1 * 100
    speedup = lat1 / lat4

    print("\n" + "=" * 60)
    if delta_pct > 2.0:
        print(f"  CONTENTION-DOMINATED: traffic +{delta_pct:.1f}%")
        print(f"  (like Cortex-A76: parallelism creates extra DRAM traffic)")
        print(f"  → tune thread count on ENERGY, not latency")
        print(f"  → expect diminishing returns beyond 2-3 threads")
    elif delta_pct < -2.0:
        print(f"  SHARING-DOMINATED: traffic {delta_pct:.1f}%")
        print(f"  (like Cortex-A78AE: threads reuse cached data)")
        print(f"  → use all available cores")
        print(f"  → parallelism is nearly free on this chip")
    else:
        print(f"  NEUTRAL: traffic {delta_pct:+.1f}% (within ±2%)")
        print(f"  (contention and sharing roughly cancel)")
        print(f"  → benchmark at each thread count to find optimum")

    print(f"\n  Speedup 1→4: {speedup:.2f}×")
    print(f"  Traffic change: {delta_pct:+.1f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
