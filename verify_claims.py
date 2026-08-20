import os
import json, glob, itertools
import numpy as np, pandas as pd
import os
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "") 
N_IMG = 50  # inferences per trial, Platform A

# Independent recomputation of every claim in the claim index of README.md.
# Deliberately written from scratch rather than reusing verify_tables.py, so
# that agreement between the two is evidence rather than a shared assumption.

def load(p):
    if "platform_b" in p:
        rows=[]
        with open(R+p) as f:
            header=f.readline().strip().split(",")
            for line in f:
                fl=line.strip().split(",")
                row={}
                for i,c in enumerate(header[:14]):
                    try: row[c]=float(fl[i]) if fl[i] else np.nan
                    except (ValueError,IndexError): row[c]=fl[i]
                rows.append(row)
        return pd.DataFrame(rows)
    return pd.read_csv(R + p)

def stat(p, n_img=N_IMG):
    d = load(p)
    lat = d.lat_mean
    llc = d.ll_cache_miss_rd / n_img
    return dict(lat=lat.mean(), sd=lat.std(ddof=1), llc=llc.mean(),
                l2=d.l2d_cache.mean()/n_img, l2r=d.l2d_cache_refill.mean()/n_img,
                l3=d.l3d_cache.mean()/n_img, d=d)

OUT = []
def chk(label, computed, paper, tol=0.06, unit=""):
    ok = abs(computed-paper) <= tol*max(abs(paper),1e-9)
    OUT.append((("OK " if ok else "XX "), label, computed, paper, unit))

# ---------- Table 1a ----------
A = {m: stat(f"data/platform_a/{f}") for m, f in
     [("yolo11n","yolo11n_fp32.csv"),("yolo11s","yolo11s_fp32.csv"),
      ("yolo11m","yolo11m_fp32.csv"),("yolo11l","yolo11l_fp32.csv"),
      ("yolo11x","yolo11x_n50.csv")]}
B = {m: stat(f"data/platform_b/cachebench_jetson_{k}.csv", 150) for m, k in
     [("yolo11n","11n"),("yolo11s","11s"),("yolo11m","11m"),
      ("yolo11l","11l"),("yolo11x","11x")]}
paperA = {"yolo11n":(139.0,0.3,6.20),"yolo11s":(353.4,1.2,15.66),"yolo11m":(963.6,2.7,46.25),
          "yolo11l":(1229.4,3.8,56.27),"yolo11x":(2475.1,7.4,124.48)}
paperB = {"yolo11n":(140.4,0.7,5.75),"yolo11s":(367.7,1.1,13.88),"yolo11m":(1034.4,2.9,36.47),
          "yolo11l":(1326.3,3.9,44.60),"yolo11x":(2774.6,10.7,93.99)}
GF = {"yolo11n":6.5,"yolo11s":21.5,"yolo11m":68.0,"yolo11l":86.9,"yolo11x":194.9}
for m in A:
    chk(f"T1a A {m} lat", A[m]['lat'], paperA[m][0], 0.01, "ms")
    chk(f"T1a A {m} sd", A[m]['sd'], paperA[m][1], 0.25, "ms")
    chk(f"T1a A {m} LLC", A[m]['llc']/1e6, paperA[m][2], 0.01, "M")
    chk(f"T1a B {m} lat", B[m]['lat'], paperB[m][0], 0.01, "ms")
    chk(f"T1a B {m} sd", B[m]['sd'], paperB[m][1], 0.30, "ms")
    chk(f"T1a B {m} LLC", B[m]['llc']/1e6, paperB[m][2], 0.01, "M")
    chk(f"T1a {m} B/A ratio", B[m]['lat']/A[m]['lat'],
        {"yolo11n":1.010,"yolo11s":1.040,"yolo11m":1.073,"yolo11l":1.079,"yolo11x":1.121}[m], 0.01)

# claim: B traffic advantage widens 7.3% -> 24.5%
for m, p in [("yolo11n",7.3),("yolo11x",24.5)]:
    chk(f"B less traffic {m}", (1-B[m]['llc']/A[m]['llc'])*100, p, 0.05, "%")

# ---------- effective bandwidth spread ----------
bwA = np.array([A[m]['llc']*64/ (A[m]['lat']/1000) /1e9 for m in A])
bwB = np.array([B[m]['llc']*64/ (B[m]['lat']/1000) /1e9 for m in B])
chk("BW spread A", (bwA.max()/bwA.min()-1)*100, 13.5, 0.06, "%")
chk("BW spread B", (bwB.max()/bwB.min()-1)*100, 21.8, 0.06, "%")
chk("BW mean A", bwA.mean(), 2.98, 0.02, "GB/s")

# ---------- Table 1b iso-FLOP ----------
iso = {"yolo11x@640":("yolo11x_n50.csv",194.9,2475.1,227.8,3.219),
       "yolo11m@1088":("yolo11m_1088.csv",196.52,2925.3,80.5,3.375),
       "yolo11s@640":("yolo11s_fp32.csv",21.5,353.4,37.9,2.836),
       "yolo11n@1152":("yolo11n_1152.csv",21.06,512.1,10.6,3.238)}
isod={}
for k,(f,gf,plat,pw,pbw) in iso.items():
    s = stat("data/platform_a/"+f); isod[k]=s
    chk(f"T1b {k} lat", s['lat'], plat, 0.01, "ms")
    chk(f"T1b {k} BW", s['llc']*64/(s['lat']/1000)/1e9, pbw, 0.02, "GB/s")
chk("iso pair1 FLOP match", (68.0*(1088/640)**2-194.9)/194.9*100, 0.83, 0.03, "%")
chk("iso pair2 FLOP match", (21.5-6.5*(1152/640)**2)/21.5*100, 2.05, 0.03, "%")
chk("iso pair1 lat diff", (isod["yolo11m@1088"]['lat']/isod["yolo11x@640"]['lat']-1)*100, 18.2, 0.03, "%")
chk("iso pair2 lat diff", (isod["yolo11n@1152"]['lat']/isod["yolo11s@640"]['lat']-1)*100, 44.9, 0.03, "%")
chk("iso pair1 traffic ratio", isod["yolo11m@1088"]['llc']/isod["yolo11x@640"]['llc'], 1.24, 0.03)
chk("iso pair2 traffic ratio", isod["yolo11n@1152"]['llc']/isod["yolo11s@640"]['llc'], 1.65, 0.03)
chk("iso pair1 weight ratio", 227.8/80.5, 2.8, 0.03)
chk("iso pair2 weight ratio", 37.9/10.6, 3.6, 0.03)

# ---------- predictors ----------
W = {m: json.load(open(R+"data/worksets.json"))[m]["weight_MB"] for m in A}
def maxerr(x, y, loo=False):
    x=np.asarray(x,float); y=np.asarray(y,float)
    if not loo:
        k=(x*y).sum()/(x*x).sum(); return np.abs(k*x-y).max()/y.max()*0 or np.abs((k*x-y)/y).max()*100
    e=[]
    for i in range(len(x)):
        m=np.ones(len(x),bool); m[i]=False
        k=(x[m]*y[m]).sum()/(x[m]*x[m]).sum()
        e.append(abs((k*x[i]-y[i])/y[i])*100)
    return max(e)
for tag,S,pf,pw,pl,pl_loo in [("A",A,38.8,19.4,10.0,10.1),("B",B,32.8,27.1,20.3,20.3)]:
    lat=[S[m]['lat'] for m in S]; llc=[S[m]['llc'] for m in S]
    gf=[GF[m] for m in S]; wt=[W[m] for m in S]
    chk(f"pred {tag} FLOP maxerr", maxerr(gf,lat), pf, 0.03,"%")
    chk(f"pred {tag} weight maxerr", maxerr(wt,lat), pw, 0.03,"%")
    chk(f"pred {tag} LLC maxerr", maxerr(llc,lat), pl, 0.03,"%")
    chk(f"pred {tag} LLC LOO maxerr", maxerr(llc,lat,True), pl_loo, 0.03,"%")
    chk(f"pred {tag} FLOP LOO maxerr", maxerr(gf,lat,True), pf, 0.03,"%")
    # R2 through-origin
    def r2(x,y):
        x=np.asarray(x,float);y=np.asarray(y,float)
        k=(x*y).sum()/(x*x).sum()
        return 1-((y-k*x)**2).sum()/((y-y.mean())**2).sum()
    chk(f"pred {tag} R2 LLC", r2(llc,lat), 0.9964 if tag=="A" else 0.9990, 0.001)
    chk(f"pred {tag} R2 FLOP", r2(gf,lat), 0.9918 if tag=="A" else 0.9966, 0.001)
# fixed-BW variant
latA=[A[m]['lat'] for m in A]; llcA=[A[m]['llc'] for m in A]
predfix=[l*64/(bwA.mean()*1e9)*1000 for l in llcA]
chk("fixed-BW predictor maxerr", max(abs((p-y)/y)*100 for p,y in zip(predfix,latA)), 7.9, 0.05,"%")
# yolo11m FLOP-model point prediction (890 ms)
kf=(np.array([GF[m] for m in A])*np.array(latA)).sum()/ (np.array([GF[m] for m in A])**2).sum()
chk("FLOP model yolo11m pred", kf*68.0, 890, 0.03, "ms")

# ---------- working sets ----------
ws=json.load(open(R+"data/worksets.json"))
for m,p in [("yolo11n",0.31),("yolo11x",0.78),("yolo11m",0.748),("yolo11l",0.730)]:
    chk(f"frac layers>L3 {m}", ws[m]["frac_layers_over_L3"], p, 0.02)

# ---------- thread sweeps ----------
TA={t:stat(f"data/platform_a/yolo11m_t{t}.csv") for t in (1,2,3,4)}
for t,(sp,eff,llcp,l3m) in {1:(1.000,100.0,40.85,24.5),2:(1.855,92.7,38.27,26.0),
                            3:(2.407,80.2,39.73,28.6),4:(2.681,67.0,46.38,34.4)}.items():
    chk(f"A t{t} speedup", TA[1]['lat']/TA[t]['lat'], sp, 0.01)
    chk(f"A t{t} efficiency", TA[1]['lat']/TA[t]['lat']/t*100, eff, 0.01,"%")
    chk(f"A t{t} LLC", TA[t]['llc']/1e6, llcp, 0.01,"M")
    chk(f"A t{t} L3 missrate", TA[t]['llc']/TA[t]['l3']*100, l3m, 0.03,"%")
chk("A 1->4 traffic", (TA[4]['llc']/TA[1]['llc']-1)*100, 13.5, 0.02,"%")
chk("A 1->4 L2 refill rise", (TA[4]['l2r']/TA[1]['l2r']-1)*100, 23.2, 0.05,"%")
chk("A L3 missrate rel rise", (34.4/24.5-1)*100, 40, 0.05,"%")
chk("A t4 sustained BW", TA[4]['llc']*64/(TA[4]['lat']/1000)/1e9, 3.08, 0.03,"GB/s")
for t,s in {2:0.078,3:0.123,4:0.164}.items():
    sp=TA[1]['lat']/TA[t]['lat']
    chk(f"A Amdahl s t{t}", (t/sp-1)/(t-1), s, 0.03)

TB={t:stat(f"data/platform_b/cachebench_orin_t{t}.csv",150) for t in (1,2,3,4)}
for t,(sp,eff,llcp) in {1:(1.000,100.0,46.95),2:(1.984,99.2,43.01),
                        3:(2.895,96.5,42.14),4:(3.789,94.7,41.88)}.items():
    chk(f"B t{t} speedup", TB[1]['lat']/TB[t]['lat'], sp, 0.01)
    chk(f"B t{t} efficiency", TB[1]['lat']/TB[t]['lat']/t*100, eff, 0.01,"%")
    chk(f"B t{t} LLC", TB[t]['llc']/1e6, llcp, 0.01,"M")
chk("B 1->4 traffic", (TB[4]['llc']/TB[1]['llc']-1)*100, -10.8, 0.02,"%")
chk("B L2 access 1 vs 4 diff", abs(TB[4]['l2']/TB[1]['l2']-1)*100, 0.04, 0.40,"%")

# ---------- core pinning ----------
P1=stat("data/platform_b/cachebench_pin_0123.csv",150)
P2=stat("data/platform_b/cachebench_pin_0145.csv",150)
UN=B["yolo11l"]
chk("pin one-block LLC", P1['llc']/1e6, 42.46, 0.01,"M")
chk("pin two-block LLC", P2['llc']/1e6, 45.32, 0.01,"M")
chk("pin unpinned LLC", UN['llc']/1e6, 44.60, 0.01,"M")
chk("pin one-block lat", P1['lat'], 1357.3, 0.01,"ms")
chk("pin two-block lat", P2['lat'], 1336.3, 0.01,"ms")
chk("pin traffic diff", (1-P1['llc']/P2['llc'])*100, 6.3, 0.03,"%")

# ---------- power ----------
PW={t:load(f"data/platform_a/yolo11m_pwr_t{t}.csv") for t in (1,2,3,4)}
for t,(lat,pa,pi,e,tp) in {1:(2587.9,6.881,4.672,17.729,57.3),2:(1399.2,9.119,4.788,12.684,64.5),
                           3:(1080.1,10.979,4.767,11.707,70.7),4:(969.3,12.339,4.782,11.846,74.3)}.items():
    d=PW[t]
    chk(f"pwr t{t} lat", d.lat_mean.mean(), lat, 0.01,"ms")
    chk(f"pwr t{t} P_active", d.p_active_w.mean(), pa, 0.005,"W")
    chk(f"pwr t{t} P_idle", d.p_idle_w.mean(), pi, 0.01,"W")
    chk(f"pwr t{t} E/img", d.energy_per_img_j.mean(), e, 0.01,"J")
    chk(f"pwr t{t} peak temp", d.peak_temp.mean(), tp, 0.01,"C")
p4=PW[4].p_active_w.mean(); p3=PW[3].p_active_w.mean()
chk("power ratio vs 6.8W", p4/6.8, 1.81, 0.01)
chk("power ratio vs 8.8W", p4/8.8, 1.40, 0.01)
chk("4th thread power rise", (p4/p3-1)*100, 12.4, 0.03,"%")
chk("4th thread lat gain", (1-PW[4].lat_mean.mean()/PW[3].lat_mean.mean())*100, 10.3, 0.03,"%")
chk("4th thread energy rise", (PW[4].energy_per_img_j.mean()/PW[3].energy_per_img_j.mean()-1)*100, 1.19, 0.06,"%")
chk("temp delta 3->4", PW[4].peak_temp.mean()-PW[3].peak_temp.mean(), 3.6, 0.06,"C")

# ---------- coverage model ----------
E, TRIP, TREES = 50.0, 5.7, 200
for t,p in [(1,255),(2,192),(3,160),(4,142)]:
    chk(f"coverage t{t}", E/PW[t].p_active_w.mean()/TRIP*TREES, p, 0.01,"trees")
chk("coverage @8.8W", E/8.8/TRIP*TREES, 199, 0.01,"trees")
chk("coverage @6.8W", E/6.8/TRIP*TREES, 258, 0.01,"trees")
chk("endurance @8.8W", E/8.8, 5.7, 0.01,"h")
chk("endurance t4", E/PW[4].p_active_w.mean(), 4.1, 0.02,"h")
chk("3 frames @t4", 3*PW[4].lat_mean.mean()/1000, 2.9, 0.03,"s")
chk("3 frames @t1", 3*PW[1].lat_mean.mean()/1000, 7.8, 0.03,"s")
chk("t_unit", TRIP*3600/TREES, 102.6, 0.01,"s")
chk("shortfall pct", (1-142/200)*100, 29, 0.03,"%")

# ---------- quantisation ----------
F=stat("data/platform_a/fp32_smoke.csv"); I=stat("data/platform_a/int8_smoke.csv")
chk("quant FP32 lat", F['lat'], 350.6, 0.01,"ms"); chk("quant FP32 sd", F['sd'], 3.6, 0.30,"ms")
chk("quant INT8 lat", I['lat'], 657.2, 0.01,"ms"); chk("quant INT8 sd", I['sd'], 1.0, 0.40,"ms")
chk("quant slowdown", I['lat']/F['lat'], 1.87, 0.01)
chk("quant LLC rise", (I['llc']/F['llc']-1)*100, 10, 0.30,"%")
chk("quant L2 refill rise", (I['l2r']/F['l2r']-1)*100, 26, 0.20,"%")
chk("quant compression", 37.9/9.9, 3.83, 0.01)

# ---------- replication ----------
chk("replication t1", abs(TA[1]['lat']/PW[1].lat_mean.mean()-1)*100, 0.008, 0.40,"%")
chk("replication t4", abs(TA[4]['lat']/PW[4].lat_mean.mean()-1)*100, 0.41, 0.30,"%")

bad=[r for r in OUT if r[0]=="XX "]
for r in OUT:
    if r[0]=="XX ":
        print(f"{r[0]}{r[1]:38s} computed={r[2]:12.4f}  manuscript={r[3]:10.4f} {r[4]}")
print(f"\n{len(OUT)-len(bad)}/{len(OUT)} checks passed")
