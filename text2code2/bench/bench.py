#!/usr/bin/env python3
"""Benchmark pyt2c (one-stage) vs the C++ text2code pipeline.

The C++ text2code produces `my_model.hpp` by emitting a SymEngine C++ program
(`Code2Cpp.cpp`), compiling it with g++/clang against libsymengine, and running
it. pyt2c does the symbolic work directly in Python via the symengine pip
package — no compile-a-program step. This measures the difference.

Reported phases:
  * pyt2c end-to-end          : `python -m pyt2c pdemodel.txt`
  * C++ text2code end-to-end  : the whole `text2code pdeapp.txt` invocation
  * C++ Code2Cpp compile only : just the g++ step (the irreducible cost pyt2c drops)

Usage:
  python bench/bench.py --venv-python /path/to/venv/python \
      --text2code /path/to/text2code --pdeapp CASE/pdeapp.txt --pdemodel CASE/pdemodel.txt \
      [--compile-cmd-file compilecmd.txt] [-n 5]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time


def timeit(cmd, cwd=None, env=None, n=5):
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        subprocess.run(cmd, cwd=cwd, env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ts.append(time.perf_counter() - t)
    ts.sort()
    return {"min": ts[0], "median": ts[len(ts) // 2], "max": ts[-1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--venv-python", default="python3")
    ap.add_argument("--pyt2c-dir", required=True, help="dir containing the pyt2c package")
    ap.add_argument("--text2code", required=True)
    ap.add_argument("--pdeapp", required=True)
    ap.add_argument("--pdemodel", required=True)
    ap.add_argument("--compile-cmd-file", help="file with the g++ Code2Cpp compile line")
    ap.add_argument("-n", type=int, default=5)
    args = ap.parse_args()

    case_dir = os.path.dirname(os.path.abspath(args.pdeapp))
    out = os.path.join(case_dir, "out")
    os.makedirs(out, exist_ok=True)

    env = dict(os.environ, PYTHONPATH=args.pyt2c_dir)
    py = timeit([args.venv_python, "-m", "pyt2c", args.pdemodel, "-o",
                 os.path.join(case_dir, "outpy")], cwd=args.pyt2c_dir, env=env, n=args.n)

    cxx = timeit([args.text2code, "pdeapp.txt", "--out-dir", out], cwd=case_dir, n=args.n)

    comp = None
    if args.compile_cmd_file and os.path.exists(args.compile_cmd_file):
        with open(args.compile_cmd_file) as f:
            cmd = f.read().strip()
        comp = timeit(["/bin/sh", "-c", cmd], cwd=case_dir, n=max(3, args.n // 2))

    def fmt(d):
        return f"min={d['min']*1000:8.1f} ms   median={d['median']*1000:8.1f} ms"

    print("=" * 64)
    print(f"pyt2c (one-stage python)     : {fmt(py)}")
    print(f"C++ text2code (end-to-end)   : {fmt(cxx)}")
    if comp:
        print(f"  of which Code2Cpp g++ compile: {fmt(comp)}")
    print("-" * 64)
    print(f"speedup pyt2c vs C++ end-to-end (median): {cxx['median']/py['median']:.1f}x")
    if comp:
        print(f"speedup pyt2c vs just the g++ compile   : {comp['median']/py['median']:.1f}x")
    print("=" * 64)


if __name__ == "__main__":
    main()
