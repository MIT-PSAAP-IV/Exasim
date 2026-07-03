#!/usr/bin/env python3
"""Compare an Exasim app's outputs against a stored golden baseline.

For EVERY output family present in the baseline dir (files matching out<family>_np<rank>.bin --
e.g. outudg = volume solution, outuhat = HDG trace, outbouudg = boundary output) the tool
concatenates all ranks and reports the relative L2
    ||new - base||_2 / ||base||_2 .
If the baseline has outqoi.txt, its numbers are compared too (max relative diff). ParMETIS
partitioning is deterministic for a fixed (mesh, nprocs), so baseline and new share per-rank
ordering/shapes. A byte-identical solver gives ~0 for every family; any change in the native
(non-PETSc) solve or the output writers shows up here.

Baselines historically stored only outudg, so a udg-only baseline behaves exactly as before;
adding outuhat/outqoi to a baseline automatically starts comparing them -- no CLI change.

Usage:  compare_app_l2.py <baseline_dir> <new_dataout_dir> [tol]
"""
import glob
import os
import re
import sys

import numpy as np


def families(d):
    """family -> {rank_filename: vector} for every out<family>_np<rank>.bin in dir d."""
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "out*_np*.bin"))):
        base = os.path.basename(f)
        m = re.match(r"(out.+)_np\d+\.bin$", base)
        if not m:
            continue
        out.setdefault(m.group(1), {})[base] = np.frombuffer(open(f, "rb").read(), dtype=np.float64)
    return out


def rel_l2(base, new):
    keys = sorted(set(base) | set(new))
    diff2 = ref2 = 0.0
    for k in keys:
        if k not in base or k not in new:
            return None, f"missing rank file {k}"
        b, n = base[k], new[k]
        if b.shape != n.shape:
            return None, f"shape mismatch {k}: {b.shape} vs {n.shape} (mesh/partition changed)"
        diff2 += float(np.sum((n - b) ** 2))
        ref2 += float(np.sum(b ** 2))
    return diff2 ** 0.5 / (ref2 ** 0.5 + 1e-300), None


def qoi_reldiff(base_dir, new_dir):
    """Max relative diff of the floats in outqoi.txt, or (None, None) if baseline has none."""
    bf = os.path.join(base_dir, "outqoi.txt")
    if not os.path.exists(bf):
        return None, None
    nf = os.path.join(new_dir, "outqoi.txt")
    if not os.path.exists(nf):
        return None, "new run has no outqoi.txt"
    fre = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
    b = [float(x) for x in fre.findall(open(bf).read())]
    n = [float(x) for x in fre.findall(open(nf).read())]
    if len(b) != len(n):
        return None, f"outqoi.txt count differs ({len(b)} vs {len(n)})"
    # QoI values include integrals like ∫(u-u_exact)^2 that are legitimately ~1e-13, where a pure
    # relative diff is ill-conditioned. Floor the denominator at a physical scale so a near-zero QoI
    # is compared essentially in absolute terms.
    worst = 0.0
    for x, y in zip(b, n):
        worst = max(worst, abs(y - x) / (abs(x) + 1e-8))
    return worst, None


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    base_dir, new_dir = sys.argv[1], sys.argv[2]
    tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-10
    bfam, nfam = families(base_dir), families(new_dir)
    if not bfam:
        print("[l2] ERROR: no out*_np*.bin in baseline")
        sys.exit(2)

    worst = 0.0
    parts = []
    for fam in sorted(bfam):
        l2, err = rel_l2(bfam[fam], nfam.get(fam, {}))
        if err:
            print(f"[l2] ERROR ({fam}): {err}")
            sys.exit(2)
        worst = max(worst, l2)
        parts.append(f"{fam}={l2:.2e}")
    q, qerr = qoi_reldiff(base_dir, new_dir)
    if qerr:
        print(f"[l2] ERROR (qoi): {qerr}")
        sys.exit(2)
    if q is not None:
        worst = max(worst, q)
        parts.append(f"qoi={q:.2e}")

    verdict = "PASS" if worst < tol else "FAIL"
    print(f"[l2] rel_L2 = {worst:.3e}  (tol {tol:.1e})  {verdict}   [{'  '.join(parts)}]")
    sys.exit(0 if worst < tol else 1)
