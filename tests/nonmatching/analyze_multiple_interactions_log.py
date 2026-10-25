#!/usr/bin/env python3
"""Extract exact per-edge donor/inverse-map evidence from a Phase 5 log."""
import argparse
import collections
import json
import math
import pathlib
import re

DONOR = re.compile(
    r"Non-matching donor search, Gamma\^(?P<gamma>[12]) \(owners in model (?P<model>[12])\): "
    r"(?P<faces>[0-9]+) faces, (?P<points>[0-9]+) points;.*?"
    r"max distance = (?P<distance>[-+0-9.eE]+), points off donor face = (?P<off>[0-9]+)")
INVERSE = re.compile(
    r"inverse maps: (?P<points>[0-9]+) points, (?P<exact>[0-9]+) projected exact, "
    r"(?P<inexact>[0-9]+) projected inexact; max residual = (?P<residual>[-+0-9.eE]+), "
    r"max relative residual = (?P<relative>[-+0-9.eE]+), "
    r"max distance from E_hat = (?P<ehat>[-+0-9.eE]+)")
NESTING = re.compile(r"mixed nesting violations, Gamma\^(?P<gamma>[12]) = (?P<count>[0-9]+)")


def finite(record, names):
    return all(math.isfinite(float(record[name])) for name in names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--mixed", action="store_true")
    parser.add_argument("--expected-edge", type=int, choices=range(3))
    args = parser.parse_args()
    lines = args.log.read_text(errors="replace").splitlines()
    records = []
    inverse_records = []
    nesting_records = []
    counts = {1: 0, 2: 0}
    for line in lines:
        donor = DONOR.search(line)
        if donor:
            gamma = int(donor.group("gamma"))
            if int(donor.group("model")) != gamma:
                raise SystemExit("FAIL: donor owner-model label disagrees with Gamma")
            counts[gamma] += 1
            record = {
                "gamma": gamma,
                "faces": int(donor.group("faces")),
                "points": int(donor.group("points")),
                "max_distance": float(donor.group("distance")),
                "off_face_points": int(donor.group("off")),
            }
            records.append(record)
            continue
        inverse = INVERSE.search(line)
        if inverse:
            inverse_records.append({
                "inverse_points": int(inverse.group("points")),
                "projected_exact": int(inverse.group("exact")),
                "projected_inexact": int(inverse.group("inexact")),
                "max_inverse_residual": float(inverse.group("residual")),
                "max_relative_inverse_residual": float(inverse.group("relative")),
                "max_distance_from_E_hat": float(inverse.group("ehat")),
            })
            continue
        nesting = NESTING.search(line)
        if nesting:
            nesting_records.append({
                "gamma": int(nesting.group("gamma")),
                "nesting_violations": int(nesting.group("count")),
            })

    expected_counts = {1: 1, 2: 1} if args.expected_edge is not None else {1: 3, 2: 3}
    if args.expected_edge is not None and counts == {1: 3, 2: 3}:
        raise SystemExit(
            "FAIL: physical edge attribution was requested for an intact graph log; "
            "outer edges 0 and 2 are indistinguishable by the available diagnostics")
    if counts != expected_counts or len(records) != sum(expected_counts.values()):
        qualifier = (f"isolated edge {args.expected_edge}" if args.expected_edge is not None
                     else "intact three-edge graph")
        raise SystemExit(f"FAIL: {qualifier} donor-record counts are {counts}, expected {expected_counts}")
    if len(inverse_records) != len(records):
        raise SystemExit("FAIL: donor and inverse-map record counts differ")
    for record in records:
        if record["faces"] <= 0 or record["points"] <= 0:
            raise SystemExit("FAIL: an edge direction has no interface work")
        if not finite(record, ("max_distance",)):
            raise SystemExit("FAIL: nonfinite donor diagnostic")
    for record in inverse_records:
        if not finite(record, ("max_inverse_residual", "max_relative_inverse_residual",
                               "max_distance_from_E_hat")):
            raise SystemExit("FAIL: nonfinite inverse-map diagnostic")
        if record["projected_exact"] + record["projected_inexact"] != record["inverse_points"]:
            raise SystemExit("FAIL: inverse-map projection counts do not sum to the point count")
    donor_points = collections.Counter(record["points"] for record in records)
    inverse_points = collections.Counter(record["inverse_points"] for record in inverse_records)
    if donor_points != inverse_points:
        raise SystemExit("FAIL: donor and inverse-map point-count multisets differ")

    # A combined graph log is intentionally treated as a multiset. Edges 0
    # and 2 have identical prescribed coarse/fine face and point counts, and
    # rank stdout may interleave, so the existing diagnostics cannot identify
    # those two physical edges. Physically attributed checks therefore use an
    # isolated-edge run selected by --expected-edge.
    if args.expected_edge is not None:
        by_gamma = {record["gamma"]: record for record in records}
        for inverse in inverse_records:
            candidates = [record for record in records
                          if record["points"] == inverse["inverse_points"]
                          and "inverse_points" not in record]
            if len(candidates) != 1:
                raise SystemExit("FAIL: isolated-edge inverse diagnostics are ambiguous")
            candidates[0].update(inverse)
        if args.mixed:
            nesting_counts = {gamma: 0 for gamma in (1, 2)}
            for nesting in nesting_records:
                gamma = nesting["gamma"]
                nesting_counts[gamma] += 1
                by_gamma[gamma].update(nesting)
            if nesting_counts != {1: 1, 2: 1}:
                raise SystemExit(
                    f"FAIL: isolated mixed-edge nesting-record counts are {nesting_counts}")
        elif nesting_records:
            raise SystemExit("FAIL: unexpected mixed-nesting diagnostics in a uniform-family run")

        normative_gamma = 1 if args.expected_edge == 1 else 2
        for gamma, record in by_gamma.items():
            record["edge_ordinal"] = args.expected_edge
            record["direction_role"] = (
                "even_domain_flux_owner" if gamma == normative_gamma
                else "coarse_owner_crossing_diagnostic")
        normative = by_gamma[normative_gamma]
        if normative["off_face_points"] != 0:
            raise SystemExit(
                f"FAIL: edge {args.expected_edge} Gamma^{normative_gamma} "
                "even-domain direction has off-face points")
        if args.mixed and normative["nesting_violations"] != 0:
            raise SystemExit(
                f"FAIL: edge {args.expected_edge} Gamma^{normative_gamma} "
                "even-domain direction has nesting violations")
    else:
        if args.mixed:
            nesting_counts = {gamma: 0 for gamma in (1, 2)}
            for nesting in nesting_records:
                nesting_counts[nesting["gamma"]] += 1
            if nesting_counts != {1: 3, 2: 3}:
                raise SystemExit(
                    f"FAIL: intact mixed-graph nesting-record counts are {nesting_counts}")
        elif nesting_records:
            raise SystemExit("FAIL: unexpected mixed-nesting diagnostics in a uniform-family run")
    payload = {
        "records": records,
        "inverse_records": inverse_records if args.expected_edge is None else [],
        "nesting_records": nesting_records if args.expected_edge is None else [],
        "attribution": ("isolated-edge content" if args.expected_edge is not None
                        else "intact multiset; outer edges 0 and 2 are indistinguishable"),
        "message_tags": "not exposed by existing production diagnostics",
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.expected_edge is None:
        print(f"STRUCTURE PASS: intact six-direction donor multiset in {args.log}; "
              f"evidence={args.output}")
    else:
        print(f"STRUCTURE PASS: isolated edge {args.expected_edge} has both donor directions; "
              f"evidence={args.output}")


if __name__ == "__main__":
    main()
