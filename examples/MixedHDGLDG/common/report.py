#!/usr/bin/env python3
"""Tabulate a refinement campaign written by campaign.py study.

report.py STUDY_ROOT -> STUDY_ROOT/study.json and STUDY_ROOT/study.md

Rates are log(E1/E2)/log(h1/h2) for every consecutive interval, using the
actual element size h = 1/N of the structured unit square/cube. A rate is
withheld when either error is at or below the component's floor
1e-10*max(1, max|exact component|), the fixed nonlinear tolerance scale,
or when either run did not converge or its error quadrature was unstable.
"""
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cases  # noqa: E402

REGIONS = ("global", "H", "L")


def names(case):
    model = cases.Case(case)
    u = ["u%d" % c for c in range(model.ncu)]
    q = ["q%s%d" % ("xyz"[d], c) for d in range(model.nd) for c in range(model.ncu)]
    return u + q


def scales(case):
    model = cases.Case(case)
    grid = np.linspace(0, 1, 9)
    points = np.array(np.meshgrid(*([grid]*model.nd), indexing="ij")).reshape(model.nd, -1).T
    return np.maximum(1., abs(model.exact(points)).max(axis=0))


def converged(result, tolerance=1e-10):
    return result["history"]["final_residual"] < tolerance


def main(root):
    rows = json.loads((root / "campaign.json").read_text())
    data, failures = {}, []
    for row in rows:
        if row["run_status"] != 0 or row.get("measure_status") != 0:
            failures.append(row)
            continue
        result = json.loads((Path(row["root"]) / "results.json").read_text())
        data[(row["case"], row["scheme"], row["p"], row["N"])] = result
    table = []
    lines = ["# Refinement study", "", "Rates use actual h=1/N; '-' marks a withheld rate (floor,",
             "non-converged run or unstable quadrature). q = -grad(u).", ""]
    for case in sorted({k[0] for k in data}):
        labels, floor = names(case), 1e-10*scales(case)
        lines += ["## " + case, ""]
        for scheme in ("hdg", "ldg", "mixed"):
            for p in (1, 2, 3):
                levels = [N for N in (4, 8, 16) if (case, scheme, p, N) in data]
                if not levels:
                    continue
                for region in REGIONS:
                    header = "| %s p=%d %s | N | h | elements | volume | " % (scheme, p, region) + " | ".join(labels) + " |"
                    lines += [header, "|" + "---|"*(5 + len(labels))]
                    previous = None
                    for N in levels:
                        result = data[(case, scheme, p, N)]
                        entry = result["errors"]["errors"][region]
                        errors = entry["rms"]
                        ok = converged(result) and result["errors"]["quadrature_stable"]
                        rate = [None]*len(labels)
                        if previous is not None:
                            prevN, prevE, prevOK = previous
                            for c in range(len(labels)):
                                if ok and prevOK and errors[c] > floor[c] and prevE[c] > floor[c]:
                                    rate[c] = math.log(prevE[c]/errors[c])/math.log(N/prevN)
                        table.append({"case": case, "scheme": scheme, "p": p, "N": N, "region": region,
                                      "h": 1/N, "elements": entry["elements"], "volume": entry["volume"],
                                      "errors": errors, "rates_from_previous": rate,
                                      "converged": converged(result),
                                      "quadrature_points": result["errors"]["quadrature"][-1]["points_per_direction"],
                                      "quadrature_stable": result["errors"]["quadrature_stable"],
                                      "newton_updates": result["history"]["newton_updates"],
                                      "gmres": result["history"]["gmres"],
                                      "linear_limit_exits": result["history"]["linear_limit_exits"],
                                      "final_residual": result["history"]["final_residual"]})
                        cells = ["%.3e%s" % (e, "" if r is None else " (%.2f)" % r) for e, r in zip(errors, rate)]
                        lines.append("| %s | %d | %.5f | %d | %.6f | %s |" % (
                            "" if ok else "NOT ACCEPTED", N, 1/N, entry["elements"], entry["volume"], " | ".join(cells)))
                        previous = (N, errors, ok)
                    lines.append("")
    lines += ["## Failed or not-run rows", ""] + ["- %s" % json.dumps(f) for f in failures] + [""]
    (root / "study.json").write_text(json.dumps({"rows": table, "failures": failures}, indent=1) + "\n")
    (root / "study.md").write_text("\n".join(lines) + "\n")
    print("rows", len(table), "failures", len(failures))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
