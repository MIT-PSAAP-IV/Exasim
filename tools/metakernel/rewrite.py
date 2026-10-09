#!/usr/bin/env python3
"""Turn a generated Exasim Kokkos kernel into a binding-agnostic body include for the meta-kernel.

Every point-indexed access is abstracted, so the same body compiles against registers, LDS or global buffers:
    udg[k*ng+i] -> MK_RD_u(k)    wdg -> MK_RD_w(k)    xdg -> MK_RD_x(k)    uhg -> MK_RD_uh(k)    nlg -> MK_RD_n(k)
    odg[k*ng+i] -> MK_RD_o(k), or MK_RD_pr(k-NCO) when k >= NCO (the properties FluxP reads)
    f[k*ng+i] = e;  -> MK_WR(k) = e;
param[] / uinf[] / time are left as is (the wrapper provides them). Any leftover use of ng or i is an error.
usage: rewrite.py kernel.cpp out.inc [--nco NCO] [--func KokkosFbou3]   (--func picks one function of a multi-function file)
"""
import re, sys
src, dst = sys.argv[1], sys.argv[2]
if "--nco" not in sys.argv: sys.exit("rewrite.py: pass --nco <the case's nco> (needed to split odg from the property inputs)")
NCO = int(sys.argv[sys.argv.index("--nco") + 1])
s = open(src).read()
if "--func" in sys.argv:
    fn = sys.argv[sys.argv.index("--func") + 1]
    st = s.find("void %s(" % fn)
    if st < 0: sys.exit("no function %s in %s" % (fn, src))
    s = s[st:]; nx = s.find("\nvoid ", 1); s = s if nx < 0 else s[:nx]
l = re.search(r"Kokkos::parallel_for\(\"\w+\",.*?KOKKOS_LAMBDA\(const size_t i\) \{\n", s, re.S)
if not l: sys.exit("no launch in " + src)
b = s[l.end():]; b = b[:b.rfind("\t});")]
IDX = [re.compile(r"\b(\w+)\[\s*(\d+)\s*\*\s*ng\s*\+\s*i\s*\]"), re.compile(r"\b(\w+)\[\s*i\s*\+\s*ng\s*\*\s*(\d+)\s*\]"),
       re.compile(r"\b(\w+)\[\s*ng\s*\*\s*(\d+)\s*\+\s*i\s*\]"), re.compile(r"\b(\w+)\[\s*i\s*\+\s*(\d+)\s*\*\s*ng\s*\]")]
IDX0 = re.compile(r"\b(\w+)\[\s*i\s*\]")
# element-major kernels (EoS / EoSdw): a[j+npe*c+npe*X*k] with j = i % npe, k = i / npe
IDXE = re.compile(r"\b(\w+)\[\s*j\s*\+\s*npe\s*\*\s*(\d+)\s*\+\s*npe\s*\*\s*\w+\s*\*\s*k\s*\]")
b = re.sub(r"^\s*int\s+j\s*=\s*i\s*%\s*npe\s*;\s*$", "", b, flags=re.M); b = re.sub(r"^\s*int\s+k\s*=\s*i\s*/\s*npe\s*;\s*$", "", b, flags=re.M)
cnt = {}
def sub(a, k):
    if a == "f": key, out = "WR", "MK_WR(%d)" % k
    elif a == "udg": key, out = "u", "MK_RD_u(%d)" % k
    elif a == "wdg": key, out = "w", "MK_RD_w(%d)" % k
    elif a == "xdg": key, out = "x", "MK_RD_x(%d)" % k
    elif a == "uhg": key, out = "uh", "MK_RD_uh(%d)" % k
    elif a == "nlg": key, out = "n", "MK_RD_n(%d)" % k
    elif a == "odg": key, out = ("pr", "MK_RD_pr(%d)" % (k - NCO)) if k >= NCO else ("o", "MK_RD_o(%d)" % k)
    else: return None
    cnt[key] = cnt.get(key, 0) + 1; return out
b = IDX0.sub(lambda m: sub(m.group(1), 0) or m.group(0), b)
b = IDXE.sub(lambda m: sub(m.group(1), int(m.group(2))) or m.group(0), b)
for r in IDX: b = r.sub(lambda m: sub(m.group(1), int(m.group(2))) or m.group(0), b)
left = [t for t in re.findall(r"\b(ng|i)\b", b)]
if left: sys.exit("%s: %d accesses not abstracted (first: %s)" % (src, len(left), re.search(r".*\b(ng|i)\b.*", b).group(0).strip()[:100]))
open(dst, "w").write(b)
print("%-28s -> %-22s %s" % (src.split("/")[-1], dst.split("/")[-1], " ".join("%s:%d" % kv for kv in sorted(cnt.items()))))
