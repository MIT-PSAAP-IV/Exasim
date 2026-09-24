"""Compare Python/Julia equilibrium-air outputs with the MATLAB reference."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


PREFIXES = (
    "outxdg_np",
    "outudg_np",
    "outwdg_np",
    "out_meshadapt_aviter",
)


def _category(name):
    for suffix in (
        "wall_distance",
        "av_raw",
        "av_smoothed",
        "flow_solution",
        "sensor_scalar",
        "sensor_raw",
        "field1",
        "field2",
        "eta",
        "displacement",
        "xdg",
    ):
        if suffix in name:
            return suffix
    if name.startswith("outxdg"):
        return "final_xdg"
    if name.startswith("outudg"):
        return "final_udg"
    if name.startswith("outwdg"):
        return "final_wdg"
    return "other"


def compare(reference_directory, candidate_directory, tolerance):
    reference_directory = Path(reference_directory)
    candidate_directory = Path(candidate_directory)
    files = sorted(
        path
        for path in reference_directory.glob("*.bin")
        if path.name.startswith(PREFIXES)
    )
    if not files:
        raise FileNotFoundError(f"No reference output files in {reference_directory}")

    results = []
    missing = []
    for reference in files:
        candidate = candidate_directory / reference.name
        if not candidate.is_file():
            missing.append(reference.name)
            continue
        ref = np.fromfile(reference, dtype=np.float64)
        got = np.fromfile(candidate, dtype=np.float64)
        if ref.shape != got.shape:
            results.append((reference.name, np.inf, np.inf, ref.size, got.size))
            continue
        difference = got - ref
        absolute = float(np.max(np.abs(difference))) if difference.size else 0.0
        relative = float(
            np.linalg.norm(difference) / max(np.linalg.norm(ref), np.finfo(float).eps)
        )
        results.append((reference.name, relative, absolute, ref.size, got.size))

    print(f"Reference: {reference_directory}")
    print(f"Candidate: {candidate_directory}")
    categories = sorted({_category(item[0]) for item in results})
    for category in categories:
        entries = [item for item in results if _category(item[0]) == category]
        worst_relative = max(entries, key=lambda item: item[1])
        worst_absolute = max(entries, key=lambda item: item[2])
        print(
            f"{category:18s} rel={worst_relative[1]:.6e} "
            f"abs={worst_absolute[2]:.6e} file={worst_relative[0]}"
        )
    if missing:
        print(f"Missing {len(missing)} files; first missing file: {missing[0]}")

    failures = [item for item in results if item[1] > tolerance]
    if failures:
        failures.sort(key=lambda item: item[1], reverse=True)
        print("Largest discrepancies:")
        for name, relative, absolute, _, _ in failures[:10]:
            print(f"  {relative:.6e}  {absolute:.6e}  {name}")
    return not missing and not failures


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", choices=("python", "julia"))
    parser.add_argument("--reference", default=str(here / "dataout"))
    parser.add_argument("--tolerance", type=float, default=1.0e-6)
    args = parser.parse_args()
    candidate = here / f"{args.candidate}_run" / "dataout"
    success = compare(args.reference, candidate, args.tolerance)
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
