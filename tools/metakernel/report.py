#!/usr/bin/env python3
"""Join meta-kernel run results with compiler resource remarks.
usage: report.py results.txt <logs dir> schedules.txt [--csv out.csv]
"""
import re, sys
res, blog, slist = sys.argv[1:4]
csv = sys.argv[sys.argv.index("--csv") + 1] if "--csv" in sys.argv else None
order = [l.split()[0] for l in open(slist) if l.strip()]
tot, ker, fail, refk, iso = {}, {}, {}, {}, {}
for l in open(res, errors="replace"):
    p = l.split()
    if not p: continue
    if p[0] == "MKRES": tot[p[1]] = dict(ms=float(p[2]), ref=float(p[3]), sp=float(p[4]), rel=float(p[5]), bit=p[6] == "1", extra=" ".join(p[7:]))
    elif p[0] == "MKKER": ker.setdefault(p[1], []).append((p[2], float(p[3]), " ".join(p[4:])))
    elif p[0] == "MKFAIL": fail[p[1]] = " ".join(p[2:])
    elif p[0] == "MKISO": iso.setdefault(p[1], []).append((p[2], float(p[3])))
    elif p[0] == "MKREF":
        if p[2].isdigit(): refk.setdefault(p[1], []).append((int(p[2]), float(p[3]), p[4]))      # v1: MKREF s j ms name
        else: refk.setdefault(p[1], []).append((len(refk.get(p[1], [])), float(p[3]), p[2]))    # e2e: MKREF s name ms
# resources: "Function Name: <mangled containing MK_<sched>_K<j>>" then VGPRs/AGPRs/ScratchSize/Occupancy/LDS lines
import os
rsc, cur = {}, None
lines = []
for s_ in order:   # per-schedule logs (blog is the logs directory)
    f_ = os.path.join(blog, s_ + ".log")
    if os.path.exists(f_): lines += open(f_, errors="replace").read().replace("\0", "").splitlines()
for l in lines:
    m = re.search(r"Function Name: (\S+)", l)
    if m:
        k = re.search(r"MK_([A-Za-z0-9_]*?)_K(\d+)(?=[^0-9])", m.group(1) + "E")
        cur = (k.group(1), int(k.group(2))) if k else None
        if cur: rsc[cur] = {}
        continue
    if cur:
        for key, pat in (("v", r"VGPRs: (\d+)"), ("a", r"AGPRs: (\d+)"), ("sc", r"ScratchSize \[bytes/lane\]: (\d+)"),
                         ("occ", r"Occupancy \[waves/SIMD\]: (\d+)"), ("lds", r"LDS Size \[bytes/block\]: (\d+)")):
            mm = re.search(pat, l)
            if mm: rsc[cur][key] = int(mm.group(1))
rows = []
if refk:   # production path per step, from the first schedule that reported it (the same code in every executable)
    r0 = [x for x in refk[next(iter(refk))] if x[2] not in ("sum", "total")]
    tot_ = [x[1] for x in refk[next(iter(refk))] if x[2] == "total"]
    if iso: print("  (same stages timed in isolation: " + " | ".join("%s %.3f" % kv for kv in iso[next(iter(iso))]) + ")")
    print("reference (production) per step: " + " | ".join("%s %.3f" % (n, ms) for _, ms, n in sorted(r0)) + "  (sum %.3f%s)" % (sum(ms for _, ms, _ in r0), ", whole %.3f" % tot_[0] if tot_ else ""))
print("%-16s %9s %8s %9s %s" % ("schedule", "total ms", "vs ref", "rel diff", "kernels: ms [VGPR+AGPR/spill B/waves/LDS B]"))
for s in order:
    if s in fail or s not in tot: print("%-16s  FAILED %s" % (s, fail.get(s, "(no result)"))); continue
    t = tot[s]
    ks = []
    for j, ms, lab in ker.get(s, []):
        jj = int(j[1:]) if j[:1] == "e" and j[1:].isdigit() else (int(j) if j.isdigit() and not any(x[0][:1] in "ex" for x in ker.get(s, [])) else None)
        r = rsc.get((s, jj)) if jj is not None else None
        rs = " [%d+%d/%d/%d/%d]" % (r.get("v", 0), r.get("a", 0), r.get("sc", 0), r.get("occ", 0), r.get("lds", 0)) if r else ""
        ks.append("%s %.3f%s" % (lab, ms, rs))
        rows.append((s, j, lab, ms, r or {}))
    print("%-16s %9.4f %7.2fx %9.1e%s %s  %s" % (s, t["ms"], t["sp"], t["rel"], "" if not t["bit"] else " (bitwise)", t["extra"].split("backend=")[-1] if "backend=" in t["extra"] else "", " | ".join(ks)))
if csv:
    with open(csv, "w") as f:
        f.write("schedule,kernel,label,ms,vgpr,agpr,scratch,waves,lds,total_ms,ref_ms,speedup,rel\n")
        for s, j, lab, ms, r in rows:
            t = tot[s]; f.write("%s,%s,%s,%.5f,%s,%s,%s,%s,%s,%.5f,%.5f,%.4f,%.3e\n" % (s, j, lab, ms, r.get("v", ""), r.get("a", ""), r.get("sc", ""), r.get("occ", ""), r.get("lds", ""), t["ms"], t["ref"], t["sp"], t["rel"]))
