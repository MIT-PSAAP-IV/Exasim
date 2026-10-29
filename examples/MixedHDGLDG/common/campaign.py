#!/usr/bin/env python3
"""Launch the Phase 6 matched runs with run.py and summarize them with compare.py.

parity ROOT : every case, scheme and route on one small matched case
study  ROOT : one verified route, p=1,2,3 and N=4,8,16, all three schemes
naca   ROOT : pure HDG, pure LDG and mixed NACA at equal total ranks

Pure accuracy references use one rank and mixed runs one H plus one L rank.
NACA uses two ranks for every scheme. Each command's output root is new and
under /tmp; failed launches are kept and listed, never retried silently.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ACCURACY = ("Poisson", "Poisson3D", "Couette", "ManufacturedNavierStokes")
FRONTENDS = ("python", "matlab", "julia", "text2code")
SCHEMES = ("hdg", "ldg", "mixed")


def runner(args, case, frontend, scheme, N, p, root, model=None):
    ranks = 2 if scheme == "mixed" or case == "NACA2D" else 1
    command = [args.python, HERE / "run.py", "--case", case, "--frontend", frontend, "--scheme", scheme,
               "--N", str(N), "--p", str(p), "--ranks", str(ranks), "--root", root,
               "--exasim-dir", args.exasim_dir, "--preprocess", args.preprocess]
    if frontend == "text2code":
        command += ["--text2code", args.text2code]
    if frontend == "matlab":
        command += ["--matlab", args.matlab]
    if model:
        command += ["--model-build", model]
    return [str(c) for c in command]


def execute(command, log):
    with open(log, "w") as stream:
        stream.write("COMMAND " + json.dumps(command) + "\n")
        stream.flush()
        status = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT).returncode
    return status


def measure(args, root):
    log = Path(str(root) + ".measure.log")
    status = execute([args.python, str(HERE / "compare.py"), "measure", str(root)], log)
    return status


def chain(args, case, frontend, N, p, base, schemes=SCHEMES):
    """Build one route's model with the mixed run, then reuse it for pure runs."""
    rows = []
    model = None
    for scheme in ("mixed",) + tuple(s for s in schemes if s != "mixed"):
        root = base / ("%s-%s-%s-N%d-p%d" % (case, frontend, scheme, N, p))
        status = execute(runner(args, case, frontend, scheme, N, p, root, model), str(root) + ".runner.log")
        measured = measure(args, root) if status == 0 else None
        rows.append({"case": case, "frontend": frontend, "scheme": scheme, "N": N, "p": p,
                     "root": str(root), "run_status": status, "measure_status": measured})
        if scheme == "mixed":
            if status != 0:
                # Without this route's model no pure run of the route can start.
                for other in schemes:
                    if other != "mixed":
                        rows.append({"case": case, "frontend": frontend, "scheme": other, "N": N, "p": p,
                                     "root": None, "run_status": "not run: model build failed"})
                break
            model = root / "model"
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("parity", "study", "naca"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--exasim-dir", required=True, type=Path)
    parser.add_argument("--preprocess", required=True, type=Path)
    parser.add_argument("--text2code", type=Path)
    parser.add_argument("--matlab", default="/usr/local/bin/matlab")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--frontend", default="python", help="route used by study/naca")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--cases", nargs="*", default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=False, exist_ok=False)
    tasks = []
    if args.command == "parity":
        for case in (args.cases or ACCURACY + ("NACA2D",)):
            for frontend in FRONTENDS:
                tasks.append((case, frontend, 4, 4 if case == "NACA2D" else 2))
    elif args.command == "study":
        for case in (args.cases or ACCURACY):
            for p in (1, 2, 3):
                for N in (4, 8, 16):
                    tasks.append((case, args.frontend, N, p))
    else:
        tasks.append(("NACA2D", args.frontend, 4, 4))
    with ThreadPoolExecutor(args.jobs) as pool:
        results = list(pool.map(lambda t: chain(args, t[0], t[1], t[2], t[3], root), tasks))
    rows = [row for chunk in results for row in chunk]
    (root / "campaign.json").write_text(json.dumps(rows, indent=1) + "\n")
    failed = [r for r in rows if r["run_status"] != 0 or r.get("measure_status") not in (0, None)]
    print(json.dumps({"runs": len(rows), "failed_or_not_run": failed}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
