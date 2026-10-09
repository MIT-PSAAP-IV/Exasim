#!/usr/bin/env python3
"""Per-stage cost breakdown from the MK_COUNTERS rocprofv3 passes (run_e2e_mpi_tuo.sh).

usage: analyze_counters.py <counters dir> [rank] [trace-pass]     (trace-pass: "trace" = stage microbenchmark, "solve" = real solve)

Attribution is by GPU ORDER: every stage range launches mkprof_begin right after its roctx push and mkprof_end right
before its pop, so the kernels between the k-th begin and the matching end belong to the k-th range (by host push
order). Kernels outside every range are labelled from the neighbouring ranges: "pc.(unwrapped)" inside a preconditioner
build (the state stage, trace, the K copy), "res.(unwrapped)" between residual stages (halo exchange, post), else
"gm.(unwrapped)" (GMRES / FD-matvec vector ops, norms). The first build is excluded (one-time first-use costs).
Counter passes rerun the same program (deterministic kernel sequence) and are joined by (kernel name, occurrence).
"""
import csv, sys, os, glob, collections

D = sys.argv[1]; RANK = sys.argv[2] if len(sys.argv) > 2 else "0"; TP = sys.argv[3] if len(sys.argv) > 3 else "trace"
COPY_BW = 3.0e12

def rows(d, pat):
    out = []
    for f in sorted(glob.glob(f"{d}/{pat}")):
        out += list(csv.DictReader(open(f, newline="").read().replace("\x00", "").splitlines()))
    return out

def rankdir(tag):
    ds = [d for d in sorted(glob.glob(f"{D}/{tag}/*")) if os.path.isdir(d)]
    for d in ds:
        if os.path.basename(d) == RANK: return d
    return ds[0] if ds else None

td = rankdir(TP)
STAGES = ["pc.elem", "pc.elemface", "pc.schur", "pc.cross", "pc.inverse", "gm.apply", "gm.cgs",
          "res.uhat", "res.q", "res.w", "res.elem", "res.face", "res.tail"]          # = mkprof::code() order
K = []; G = []; bad = 0
for r in rows(td, "*kernel_trace.csv"):
    try: K.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"], int(r.get("Grid_Size_X") or r.get("Grid_Size") or 0)))
    except Exception: bad += 1
K.sort()
names = []
print(f"{TP} rank {RANK}: {len(K)} kernels ({bad} malformed rows skipped), "
      f"{sum(1 for k in K if k[2].startswith('mkprof_begin'))} begin markers")
# attribution by GPU order: the stage is the begin marker's grid size (stage code)
lab = [None] * len(K); cur = None
for i, (a, b, n, g) in enumerate(K):
    if n.startswith("mkprof_begin"):
        cur = len(names); names.append(STAGES[g - 1] if 1 <= g <= len(STAGES) else f"code{g}"); lab[i] = "#"; continue
    if n.startswith("mkprof_end"): cur = None; lab[i] = "#"; continue
    lab[i] = cur
K = [(a, b, n) for a, b, n, g in K]
nxt = [None] * len(K); last = None
for i in range(len(K) - 1, -1, -1):
    if isinstance(lab[i], int): last = lab[i]
    nxt[i] = last
ri = [None] * len(K); p = None
for i in range(len(K)):
    if isinstance(lab[i], int): p = lab[i]
    ri[i] = p
def category(i):
    l = lab[i]
    if l == "#": return None
    if isinstance(l, int): return names[l]
    pn = names[ri[i]] if ri[i] is not None else ""; nn = names[nxt[i]] if nxt[i] is not None else ""
    if pn.startswith("pc.") and nn.startswith("pc."): return "pc.(unwrapped)"
    if pn.startswith("res.") and nn.startswith("res."): return "res.(unwrapped)"
    return "gm.(unwrapped)"
kfirst = None
for k, n in enumerate(names):
    if n.startswith("pc."): kfirst = k
    elif kfirst is not None: break
steady_from = 0
if kfirst is not None:
    for i in range(len(K)):
        if lab[i] == kfirst: steady_from = i
print(f"first build = ranges 0..{kfirst}; steady from kernel {steady_from}")

occ = collections.Counter(); keys = []
for a, b, n in K: occ[n] += 1; keys.append((n, occ[n]))
ctr = collections.defaultdict(dict)
for tag in ("fetch", "write", "f64", "mops", "occ", "mbusy", "l2", "lds", "dram"):
    d = rankdir(tag)
    if not d: continue
    disp = collections.defaultdict(dict); dname = {}
    for r in rows(d, "*counter_collection.csv"):
        try: did = int(r["Dispatch_Id"]); disp[did][r["Counter_Name"]] = float(r["Counter_Value"]); dname[did] = r["Kernel_Name"]
        except Exception: pass
    o2 = collections.Counter()
    for did in sorted(disp):
        n = dname[did]; o2[n] += 1; ctr[(n, o2[n])].update(disp[did])
print(f"kernels with counters: {sum(1 for k in keys if k in ctr)}/{len(keys)}")

S = collections.defaultdict(lambda: collections.defaultdict(float)); TK = collections.defaultdict(lambda: collections.defaultdict(float))
def add(s, d, v):
    s["n"] += 1; s["t"] += d
    s["by"] += (v.get("FETCH_SIZE", 0) + v.get("WRITE_SIZE", 0)) * 1024
    s["f64"] += v.get("TOTAL_64_OPS", 0); s["mfma"] += v.get("SQ_INSTS_VALU_MFMA_MOPS_F64", 0) * 512
    s["hit"] += v.get("TCC_HIT_sum", 0); s["miss"] += v.get("TCC_MISS_sum", 0)
    s["lds"] += v.get("SQ_INSTS_LDS", 0); s["ldsbc"] += v.get("SQ_LDS_BANK_CONFLICT", 0)
    if "OccupancyPercent" in v: s["occw"] += v["OccupancyPercent"] * d; s["occt"] += d
for i, ((a, b, n), key) in enumerate(zip(K, keys)):
    if i < steady_from: continue
    c = category(i)
    if c is None: continue
    d = (b - a) * 1e-9; v = ctr.get(key, {})
    add(S[c], d, v); add(TK[(c, n.split("(")[0].replace("void ", "")[:48])], d, v)

def line(name, s, width=26):
    t = s["t"]; by = s["by"]; fl = s["f64"]
    bw = by / t if t else 0; ff = fl / t if t else 0
    mf = 100 * s["mfma"] / fl if fl else 0
    hit = 100 * s["hit"] / (s["hit"] + s["miss"]) if s.get("hit", 0) + s.get("miss", 0) else float("nan")
    oc = s["occw"] / s["occt"] if s.get("occt") else float("nan")
    bc = 100 * s["ldsbc"] / s["lds"] if s.get("lds") else float("nan")
    return (f"{name:{width}s} {int(s['n']):6d} {t*1e3:9.2f} {by/1e9:8.2f} {bw/1e12:6.2f} {100*bw/COPY_BW:5.0f}% "
            f"{fl/1e9:9.1f} {ff/1e12:6.2f} {mf:5.0f}% {fl/by if by else 0:7.2f} {hit:6.1f} {oc:5.1f} {bc:6.1f}")
tot = sum(s["t"] for s in S.values())
hdr = (f"{'stage':26s} {'kern':>6s} {'GPU ms':>9s} {'GB':>8s} {'TB/s':>6s} {'copy%':>6s} {'GFLOP':>9s} {'TF/s':>6s} "
       f"{'MFMA%':>6s} {'flop/B':>7s} {'L2hit':>6s} {'occ%':>5s} {'LDSbc':>6s}")
print(f"\nsteady GPU kernel time {tot*1e3:.1f} ms (rank {RANK}, {TP})\n"); print(hdr); print("-" * len(hdr))
for c, s in sorted(S.items(), key=lambda x: -x[1]["t"]): print(line(c, s))
print("\nGB = FETCH_SIZE+WRITE_SIZE (L2 <-> memory); copy% = vs the measured 3.0 TB/s copy ceiling (HBM peak 5.3 TB/s);"
      "\nTF/s = TOTAL_64_OPS/time (f64 peaks: vector 61.3, MFMA 122.6); MFMA% = share of f64 flops on the matrix cores;"
      "\nLDSbc = LDS bank-conflict cycles per LDS instruction (%).")
print("\ntop kernels per stage:")
for c, s in sorted(S.items(), key=lambda x: -x[1]["t"]):
    ks = sorted(((k, v) for (cc, k), v in TK.items() if cc == c), key=lambda x: -x[1]["t"])[:5]
    print(f"  [{c}]")
    for k, v in ks: print("    " + line(k, v, 44))

# ---------------------------------------------------------------------------------------------------------------------
# combined report (when TP == "solve"): times and call counts from the real solve, per-call bytes / flops / L2 / occupancy
# measured in the stage microbenchmark (same kernels, same sizes), joined per (stage, kernel). Lower bound per stage =
# max(bytes / 3.0 TB/s copy ceiling, vector flops / 61.3 TF + MFMA flops / 122.6 TF) and a launch floor of 4 us/kernel.
if TP == "solve":
    import subprocess, json
    mb = {}
    # re-run this script's attribution on the microbenchmark trace in-process: simplest is to recompute from its files
    def per_call(tp):
        global td
        out = collections.defaultdict(lambda: collections.defaultdict(float))
        tdd = rankdir(tp); KK = []
        for r in rows(tdd, "*kernel_trace.csv"):
            try: KK.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"], int(r.get("Grid_Size_X") or 0)))
            except Exception: pass
        KK.sort(); nm = []; lb = [None] * len(KK); cu = None
        for i, (a, b, n, g) in enumerate(KK):
            if n.startswith("mkprof_begin"): cu = len(nm); nm.append(STAGES[g - 1] if 1 <= g <= len(STAGES) else "?"); lb[i] = "#"; continue
            if n.startswith("mkprof_end"): cu = None; lb[i] = "#"; continue
            lb[i] = cu
        oc = collections.Counter()
        for i, (a, b, n, g) in enumerate(KK):
            oc[n] += 1
            if lb[i] == "#" or lb[i] is None: continue
            v = ctr.get((n, oc[n]))
            if not v: continue
            o = out[(nm[lb[i]], n.split("(")[0].replace("void ", "")[:48])]
            o["n"] += 1; o["t"] += (b - a) * 1e-9
            o["by"] += (v.get("FETCH_SIZE", 0) + v.get("WRITE_SIZE", 0)) * 1024
            o["f64"] += v.get("TOTAL_64_OPS", 0); o["mfma"] += v.get("SQ_INSTS_VALU_MFMA_MOPS_F64", 0) * 512
            o["hit"] += v.get("TCC_HIT_sum", 0); o["miss"] += v.get("TCC_MISS_sum", 0)
            if "OccupancyPercent" in v: o["occw"] += v["OccupancyPercent"] * (b - a) * 1e-9; o["occt"] += (b - a) * 1e-9
        return out
    # counters were collected on the microbenchmark: rebuild ctr from its passes (ctr above was keyed on the solve trace)
    ctr.clear()
    for tag in ("fetch", "write", "f64", "mops", "occ", "mbusy", "l2", "lds", "dram"):
        d = rankdir(tag)
        if not d: continue
        disp = collections.defaultdict(dict); dname = {}
        for r in rows(d, "*counter_collection.csv"):
            try: did = int(r["Dispatch_Id"]); disp[did][r["Counter_Name"]] = float(r["Counter_Value"]); dname[did] = r["Kernel_Name"]
            except Exception: pass
        o2 = collections.Counter()
        for did in sorted(disp):
            n = dname[did]; o2[n] += 1; ctr[(n, o2[n])].update(disp[did])
    MB = per_call("trace")
    print("\n================ COMBINED: solve times x microbenchmark per-call counters ================")
    hdr = (f"{'stage':18s} {'GPU ms':>8s} {'%':>5s} {'launches':>8s} {'GB':>7s} {'TB/s':>6s} {'GFLOP':>7s} {'MFMA%':>6s} {'L2hit':>6s} "
           f"{'occ%':>5s} {'mem floor':>9s} {'flop floor':>10s} {'launch floor':>12s} {'bound ms':>8s} {'headroom':>8s}")
    print(hdr); print("-" * len(hdr))
    tot = sum(s["t"] for s in S.values()); rowsout = []
    for c in S:
        t = 0; n = 0; by = 0; fv = 0; fm = 0; hit = 0; miss = 0; ow = 0; ot = 0; covered = 0
        for (cc, k), v in TK.items():
            if cc != c: continue
            t += v["t"]; n += v["n"]
            m = MB.get((c, k)) or next((MB[x] for x in MB if x[1] == k), None)
            if m and m["n"]:
                f = v["n"] / m["n"]; covered += v["t"]
                by += m["by"] * f; fm += m["mfma"] * f; fv += (m["f64"] - m["mfma"]) * f
                hit += m["hit"] * f; miss += m["miss"] * f; ow += m["occw"] * f; ot += m["occt"] * f
        memf = by / 3.0e12; flopf = fv / 61.3e12 + fm / 122.6e12; lf = n * 4e-6
        bound = max(memf, flopf, lf)
        rowsout.append((t, c, n, by, fv + fm, fm, hit, miss, ow, ot, memf, flopf, lf, bound, covered))
    for t, c, n, by, fl, fm, hit, miss, ow, ot, memf, flopf, lf, bound, cov in sorted(rowsout, reverse=True):
        print(f"{c:18s} {t*1e3:8.1f} {100*t/tot:5.1f} {int(n):8d} {by/1e9:7.1f} {by/t/1e12 if t else 0:6.2f} {fl/1e9:7.1f} "
              f"{100*fm/fl if fl else 0:5.0f}% {100*hit/(hit+miss) if hit+miss else float('nan'):6.1f} {ow/ot if ot else float('nan'):5.1f} "
              f"{memf*1e3:9.1f} {flopf*1e3:10.1f} {lf*1e3:12.1f} {bound*1e3:8.1f} {(t-bound)*1e3:8.1f}"
              + ("" if cov > 0.9 * t else f"   (counters cover {100*cov/t:.0f}% of time)"))
    print(f"{'TOTAL':18s} {tot*1e3:8.1f}")
    print("mem floor = bytes at 3.0 TB/s; flop floor = vector/61.3 + MFMA/122.6 TF; launch floor = 4 us per kernel;"
          "\nheadroom = GPU time - max(floors): what a perfect kernel at the measured ceilings would save (GPU time only;"
          "\nhost gaps between kernels are NOT included -- see the solve wall vs GPU time).")
