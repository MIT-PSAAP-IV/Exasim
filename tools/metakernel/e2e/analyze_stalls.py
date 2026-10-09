#!/usr/bin/env python3
"""Per-kernel stall breakdown from the MK_STALLS rocprofv3 passes (run_e2e_mpi_tuo.sh).

usage: analyze_stalls.py <stalls dir> [rank] [top]

Every pass reruns the same stage microbenchmark, so counters are aggregated per KERNEL NAME (sums over its calls).
Wave-cycle shares follow the usual rocprof-compute reading: SQ_WAVE_CYCLES = cycles waves were resident;
SQ_ACTIVE_INST_<X> = cycles issuing instructions of type X; SQ_WAIT_INST_ANY = cycles waiting for a dependency
(s_waitcnt on memory/LDS/export, i.e. latency the wave could not hide); SQ_WAIT_ANY = waiting for anything.
VMEM latency = SQ_INST_LEVEL_VMEM / SQ_INSTS_VMEM (average cycles a vector memory instruction is in flight).
"""
import csv, sys, os, glob, collections

D = sys.argv[1]; RANK = sys.argv[2] if len(sys.argv) > 2 else "0"; TOP = int(sys.argv[3]) if len(sys.argv) > 3 else 14

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
def short(n): return n.split("(")[0].replace("void ", "").replace("exasim_mfma::", "")[:40]

T = collections.defaultdict(lambda: {"n": 0, "t": 0.0, "vgpr": 0, "agpr": 0, "sgpr": 0, "lds": 0, "scr": 0, "wg": 0, "grid": 0})
for r in rows(rankdir("trace"), "*kernel_trace.csv"):
    try:
        k = short(r["Kernel_Name"]); t = T[k]; t["n"] += 1; t["t"] += (int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) * 1e-9
        t["vgpr"] = max(t["vgpr"], int(r.get("VGPR_Count") or 0)); t["agpr"] = max(t["agpr"], int(r.get("Accum_VGPR_Count") or 0))
        t["sgpr"] = max(t["sgpr"], int(r.get("SGPR_Count") or 0)); t["lds"] = max(t["lds"], int(r.get("LDS_Block_Size") or 0))
        t["scr"] = max(t["scr"], int(r.get("Scratch_Size") or 0)); t["wg"] = int(r.get("Workgroup_Size_X") or 0)
        t["grid"] = max(t["grid"], int(r.get("Grid_Size_X") or 0))
    except Exception: pass
C = collections.defaultdict(lambda: collections.defaultdict(float)); CN = collections.defaultdict(lambda: collections.defaultdict(int))
for tag in sorted(glob.glob(f"{D}/p*")):
    d = rankdir(os.path.basename(tag))
    if not d: continue
    for r in rows(d, "*counter_collection.csv"):
        try: k = short(r["Kernel_Name"]); C[k][r["Counter_Name"]] += float(r["Counter_Value"]); CN[k][r["Counter_Name"]] += 1
        except Exception: pass
def avg(k, c): return C[k][c] / CN[k][c] if CN[k][c] else float("nan")
def ratio(k, a, b): return 100 * C[k][a] / C[k][b] if C[k][b] else float("nan")

ks = sorted([k for k in T if k in C and not k.startswith("mkprof")], key=lambda k: -T[k]["t"])[:TOP]
tot = sum(T[k]["t"] for k in T)
print(f"stage microbenchmark, rank {RANK}: {sum(T[k]['n'] for k in T)} kernel calls, {tot*1e3:.1f} ms GPU; top {len(ks)} kernels by time\n")
h1 = f"{'kernel':40s} {'calls':>5s} {'us/call':>8s} {'VGPR':>4s} {'AGPR':>4s} {'LDS':>6s} {'scr':>4s} {'WG':>4s} {'occ%':>5s}"
print(h1); print("-" * len(h1))
for k in ks:
    t = T[k]
    print(f"{k:40s} {t['n']:5d} {t['t']/t['n']*1e6:8.1f} {t['vgpr']:4d} {t['agpr']:4d} {t['lds']:6d} {t['scr']:4d} {t['wg']:4d} {avg(k,'OccupancyPercent'):5.1f}")
print("\nshare of wave-resident cycles (%):")
h2 = (f"{'kernel':40s} {'VALU':>5s} {'VMEM':>5s} {'LDS':>5s} {'SALU':>5s} {'MISC':>5s} {'depwait':>7s} {'anywait':>7s} | {'VMEMlat':>7s} {'VmemLat':>7s} "
      f"{'IfetchLat':>9s} {'lanes%':>6s} {'issue%':>6s} {'MFMAbusy%':>9s} {'RESstall':>9s} {'WVLIMstall':>10s}")
print(h2); print("-" * len(h2))
for k in ks:
    W = "SQ_WAVE_CYCLES"
    vml = C[k]["SQ_INST_LEVEL_VMEM"] / C[k]["SQ_INSTS_VMEM"] if C[k]["SQ_INSTS_VMEM"] else float("nan")
    mf = ratio(k, "SQ_VALU_MFMA_BUSY_CYCLES", "SQ_BUSY_CU_CYCLES")
    print(f"{k:40s} {ratio(k,'SQ_ACTIVE_INST_VALU',W):5.1f} {ratio(k,'SQ_ACTIVE_INST_VMEM',W):5.1f} {ratio(k,'SQ_ACTIVE_INST_LDS',W):5.1f} "
          f"{ratio(k,'SQ_ACTIVE_INST_SCA',W):5.1f} {ratio(k,'SQ_ACTIVE_INST_MISC',W):5.1f} {ratio(k,'SQ_WAIT_INST_ANY',W):7.1f} {ratio(k,'SQ_WAIT_ANY',W):7.1f} | "
          f"{vml:7.0f} {avg(k,'VmemLatency'):7.0f} {avg(k,'InstrFetchLatency'):9.0f} {avg(k,'VALUUtilization'):6.1f} {avg(k,'ValuPipeIssueUtil'):6.1f} "
          f"{mf:9.1f} {C[k]['SPI_RA_RES_STALL_CSN']:9.0f} {C[k]['SPI_RA_WVLIM_STALL_CSN']:10.0f}")
print("\ninstruction mix per call:")
h3 = f"{'kernel':40s} {'VALU':>9s} {'VMEM rd':>8s} {'VMEM wr':>8s} {'LDS':>8s} {'SALU':>8s} {'SMEM':>7s} {'branch':>7s} {'ifetch':>8s}"
print(h3); print("-" * len(h3))
for k in ks:
    n = T[k]["n"]
    g = lambda c: C[k][c] / max(1, CN[k][c] // max(1, 1)) if CN[k][c] else float("nan")
    per = lambda c: C[k][c] / CN[k][c] if CN[k][c] else float("nan")
    print(f"{k:40s} {per('SQ_INSTS_VALU'):9.0f} {per('SQ_INSTS_VMEM_RD'):8.0f} {per('SQ_INSTS_VMEM_WR'):8.0f} {per('SQ_INSTS_LDS'):8.0f} "
          f"{per('SQ_INSTS_SALU'):8.0f} {per('SQ_INSTS_SMEM'):7.0f} {per('SQ_INSTS_BRANCH'):7.0f} {per('SQ_IFETCH'):8.0f}")
print("\n(instruction counts are per call, summed over all waves; depwait = SQ_WAIT_INST_ANY = unhidden latency;"
      "\n VMEMlat = SQ_INST_LEVEL_VMEM / SQ_INSTS_VMEM in cycles; RES/WVLIM stall = wave launches held back by registers/LDS vs a wave limit)")
