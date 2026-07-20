import csv, json, os, statistics as st

def agg(f, n=50):
    if not os.path.exists(f): return None
    r = list(csv.DictReader(open(f, newline='')))
    if not r: return None
    d = {}
    for k, nm in [('lat_mean','lat'),('ll_cache_miss_rd','ll'),('l2d_cache_refill','l2'),
                  ('l3d_cache','l3'),('peak_temp','temp'),('p_active_w','pw'),
                  ('p_idle_w','pi'),('energy_per_img_j','ej')]:
        v = [float(x[k]) for x in r if x.get(k) not in (None,'','None')]
        if v:
            d[nm] = st.mean(v)
            d[nm+'_sd'] = st.stdev(v) if len(v)>1 else 0.0
    for k in ('governor','on_battery','ort_version'):
        if r[0].get(k): d[k] = r[0][k]
    if 'll' in d: d['ll_img'] = d['ll']/n
    return d

print("="*114); print("PAPER 7 - RASPBERRY PI 5 SUMMARY"); print("="*114)

print("\n[1] MODEL SWEEP (640, 4 threads)")
print(f"{'model':<10}{'GFLOP':>7}{'lat ms':>11}{'sd':>7}{'LLC/img':>11}{'MB/img':>9}{'GB/s':>8}{'GF/s':>8}{'F/B':>7}{'degC':>7}")
bw=[]
for nm,fl,f in [('yolo11n',6.5,'yolo11n_fp32.csv'),('yolo11s',21.5,'yolo11s_fp32.csv'),
                ('yolo11m',68.0,'yolo11m_fp32.csv'),('yolo11l',86.9,'yolo11l_fp32.csv'),
                ('yolo11x',194.9,'yolo11x_n50.csv')]:
    d=agg(f)
    if not d: print(f"{nm:<10}  [missing {f}]"); continue
    mb=d['ll_img']*64/1e6; b=mb/1e3/(d['lat']/1e3); bw.append(b)
    print(f"{nm:<10}{fl:>7.1f}{d['lat']:>11.1f}{d.get('lat_sd',0):>7.1f}{d['ll_img']/1e6:>10.2f}M"
          f"{mb:>9.0f}{b:>8.3f}{fl/(d['lat']/1e3):>8.1f}{fl*1e9/(d['ll_img']*64):>7.1f}{d.get('temp',0):>7.1f}")
if bw: print(f"  bandwidth {min(bw):.2f}-{max(bw):.2f} GB/s = {(max(bw)/min(bw)-1)*100:.0f}% spread")

print("\n[2] THREAD SWEEP (yolo11m 640)")
print(f"{'thr':>4}{'lat ms':>11}{'speedup':>9}{'eff':>7}{'LLC/img':>11}{'L3miss':>9}{'GB/s':>8}{'per-thr':>9}{'degC':>7}")
base=None
for T in (1,2,3,4):
    d=agg(f'yolo11m_t{T}.csv')
    if not d: print(f"{T:>4}  [missing]"); continue
    base=base or d['lat']; b=d['ll_img']*64/1e9/(d['lat']/1e3)
    print(f"{T:>4}{d['lat']:>11.1f}{base/d['lat']:>9.3f}{base/d['lat']/T*100:>6.1f}%{d['ll_img']/1e6:>10.2f}M"
          f"{d['ll']/d['l3']*100:>8.1f}%{b:>8.2f}{b/T:>9.3f}{d.get('temp',0):>7.1f}")

print("\n[3] THREAD SWEEP + POWER (battery)")
print(f"{'thr':>4}{'lat ms':>11}{'activeW':>10}{'idleW':>9}{'J/img':>9}{'margW':>9}{'margJ':>9}{'degC':>7}{'batt':>7}")
best=None
for T in (1,2,3,4):
    d=agg(f'yolo11m_pwr_t{T}.csv')
    if not d: print(f"{T:>4}  [missing]"); continue
    mw=d.get('pw',0)-d.get('pi',0)
    print(f"{T:>4}{d['lat']:>11.1f}{d.get('pw',0):>10.3f}{d.get('pi',0):>9.3f}{d.get('ej',0):>9.3f}"
          f"{mw:>9.3f}{mw*d['lat']/1000:>9.3f}{d.get('temp',0):>7.1f}{str(d.get('on_battery','?'))[:5]:>7}")
    if d.get('ej') and (best is None or d['ej']<best[1]): best=(T,d['ej'])
if best: print(f"  -> minimum energy at {best[0]} threads ({best[1]:.3f} J/img)")

print("\n[4] ISO-FLOP PAIRS")
print(f"{'pair':<5}{'config':<16}{'GFLOP':>8}{'wtMB':>8}{'lat ms':>11}{'LLC/img':>11}{'GB/s':>8}")
got={}
for p,nm,fl,wt,f in [('A','yolo11x @640',194.9,227.8,'yolo11x_n50.csv'),
                     ('A','yolo11m @1088',196.5,80.5,'yolo11m_1088.csv'),
                     ('B','yolo11s @640',21.5,37.9,'yolo11s_fp32.csv'),
                     ('B','yolo11n @1152',21.1,10.6,'yolo11n_1152.csv')]:
    d=agg(f)
    if not d: print(f"{p:<5}{nm:<16}  [missing {f}]"); continue
    got[nm]=(d['lat'],d['ll_img'],fl)
    print(f"{p:<5}{nm:<16}{fl:>8.1f}{wt:>8.1f}{d['lat']:>11.1f}{d['ll_img']/1e6:>10.2f}M"
          f"{d['ll_img']*64/1e9/(d['lat']/1e3):>8.3f}")
for a,b in (('yolo11x @640','yolo11m @1088'),('yolo11s @640','yolo11n @1152')):
    if a in got and b in got:
        lr=got[b][0]/got[a][0]; mr=got[b][1]/got[a][1]
        print(f"  {a} vs {b}: FLOP {abs(got[b][2]/got[a][2]-1)*100:.2f}% apart, "
              f"lat x{lr:.3f}, traffic x{mr:.3f}, agree {abs(lr-mr)/mr*100:.1f}%")

print("\n[5] INT8")
f32,i8=agg('fp32_smoke.csv'),agg('int8_smoke.csv')
for tag,d in (('FP32',f32),('INT8',i8)):
    print(f"  {tag:<6}{d['lat']:>9.1f} +/- {d.get('lat_sd',0):.1f} ms   LLC/img {d['ll_img']/1e6:.2f}M" if d else f"  {tag:<6}[missing]")
if f32 and i8:
    print(f"  -> INT8 {i8['lat']/f32['lat']:.2f}x {'SLOWER' if i8['lat']>f32['lat'] else 'faster'}"
          f"   LLC {(i8['ll_img']/f32['ll_img']-1)*100:+.0f}%   L2 {(i8['l2']/f32['l2']-1)*100:+.0f}%")

print("\n[6] WORKING SETS")
if os.path.exists('worksets.json'):
    w=json.load(open('worksets.json'))
    print(f"{'model':<10}{'wtMB':>9}{'worksetMB':>12}{'layers':>8}{'>L3':>6}{'>L3%':>8}{'measured':>10}{'re-stream':>11}")
    for k,v in w.items():
        print(f"{k:<10}{v['weight_MB']:>9.1f}{v['total_workset_MB']:>12.1f}{v['n_compute_layers']:>8}"
              f"{v['layers_over_L3']:>6}{v['frac_layers_over_L3']*100:>7.1f}%{v['measured_MB_per_img']:>10}"
              f"{v['measured_MB_per_img']/v['total_workset_MB']:>10.2f}x")
else: print("  [worksets.json missing]")
print("\n"+"="*114)
