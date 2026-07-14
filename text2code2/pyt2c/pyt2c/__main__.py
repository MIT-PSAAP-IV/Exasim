"""CLI: ``python -m pyt2c <pdemodel.txt> [-o OUTDIR|-] [--stdout]``.

Writes ``<OUTDIR>/my_model.hpp`` (default OUTDIR=``generated``).
"""
from __future__ import annotations

import argparse
import os
import sys

from .parser import parse_file
from .codegen import generate_header


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pyt2c", description=__doc__)
    ap.add_argument("model", help="path to pdemodel.txt")
    ap.add_argument("-o", "--out-dir", default="generated",
                    help="output directory for my_model.hpp (default: generated)")
    ap.add_argument("--stdout", action="store_true", help="write the header to stdout")
    ap.add_argument("--emit-app", metavar="DIR",
                    help="emit a full standalone header-only C++ app scaffold into DIR "
                         "(driver + CMakeLists + build.sh + README + generated/my_model.hpp)")
    ap.add_argument("--app-name", help="app/target name (default: basename of --emit-app DIR)")
    ap.add_argument("--model-id", type=int, default=100,
                    help="builtinmodelID passed to CSolution in the generated driver (default: 100)")
    args = ap.parse_args(argv)

    spec = parse_file(args.model)

    if args.emit_app:
        from .appgen import emit_app
        dest = emit_app(spec, args.emit_app, app_name=args.app_name, model_id=args.model_id)
        print(f"wrote standalone app scaffold to {dest}")
        return 0

    header = generate_header(spec)
    if args.stdout or args.out_dir == "-":
        sys.stdout.write(header)
        return 0
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "my_model.hpp")
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
