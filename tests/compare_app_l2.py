#!/usr/bin/env python3
"""Compare an Exasim app's solution output against a stored golden baseline via relative L2.

Each app run writes dataout/outudg_np<rank>.bin -- the volume solution (udg) DOF vector per MPI
rank. ParMETIS partitioning is deterministic for a fixed (mesh, nprocs), so the baseline and the
new run produce the same per-rank ordering and shapes; we concatenate all ranks and report
    ||udg_new - udg_base||_2 / ||udg_base||_2.
A byte-identical solver gives ~0; any change in the native (non-PETSc) solve shows up here.

Usage:  compare_app_l2.py <baseline_dir> <new_dataout_dir> [tol]
"""
import glob
import os
import sys

import numpy as np


def load_udg(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "outudg_np*.bin"))):
        out[os.path.basename(f)] = np.frombuffer(open(f, "rb").read(), dtype=np.float64)
    return out


def rel_l2(base, new):
    keys = sorted(set(base) | set(new))
    if not keys:
        return None, "no outudg_np*.bin found"
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


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    base_dir, new_dir = sys.argv[1], sys.argv[2]
    tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-10
    l2, err = rel_l2(load_udg(base_dir), load_udg(new_dir))
    if err:
        print(f"[l2] ERROR: {err}")
        sys.exit(2)
    verdict = "PASS" if l2 < tol else "FAIL"
    print(f"[l2] rel_L2 = {l2:.3e}  (tol {tol:.1e})  {verdict}")
    sys.exit(0 if l2 < tol else 1)
