"""Interactive and command-line entry point for the four-domain 2D example."""

import argparse

from example_driver import (DEFAULT_N, DEFAULT_PORDER, DEFAULT_RANKS,
                            build_case, parse_active_edges, parse_ranks,
                            run_example)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--porder", type=int, default=DEFAULT_PORDER)
    parser.add_argument("--N", type=int, default=DEFAULT_N)
    parser.add_argument("--ranks", default=",".join(map(str, DEFAULT_RANKS)))
    parser.add_argument("--outdir")
    parser.add_argument("--active-edges", default="1,1,1")
    args = parser.parse_args(argv)
    result = run_example(args.porder, args.N, parse_ranks(args.ranks), args.outdir,
                         parse_active_edges(args.active_edges))
    print(f"workdir={result['case']['outdir']}")
    print(f"global final residual: {result['final_residual']:.17e}")
    print(f"maximum nodal error: {result['max_error']:.17e}")
    return result


if __name__ == "__main__":
    main()
