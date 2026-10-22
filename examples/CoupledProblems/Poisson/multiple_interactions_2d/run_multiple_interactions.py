#!/usr/bin/env python3
"""Callable and command-line runner for the four-domain 2D example."""

import argparse
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from example_driver import (DEFAULT_N, DEFAULT_PORDER, DEFAULT_RANKS,
                            parse_active_edges, parse_ranks, run_example)


def run_multiple_interactions(porder, N, domain_ranks, outdir,
                              active_edges=(1, 1, 1)):
    return run_example(porder, N, domain_ranks, outdir, active_edges)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--porder", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--ranks", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--active-edges", default="1,1,1")
    args = parser.parse_args(argv)
    result = run_multiple_interactions(
        args.porder, args.N, parse_ranks(args.ranks), args.outdir,
        parse_active_edges(args.active_edges))
    print(f"Global final residual: {result['final_residual']:.17e}")
    print(f"Maximum nodal error: {result['max_error']:.17e}")
    print(f"EXASIM_OUTDIR={result['case']['outdir']}")
    return result


if __name__ == "__main__":
    main()
