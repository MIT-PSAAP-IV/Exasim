"""Compare Python and Julia Sharp-B equilibrium outputs with MATLAB."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


STEMS = ("outxdg", "outudg", "outvdg", "outwdg")


def _load(directory, stem):
    fields = []
    rank = 0
    while (directory / f"{stem}_np{rank}.bin").is_file():
        fields.append(np.fromfile(directory / f"{stem}_np{rank}.bin", dtype=np.float64))
        rank += 1
    if not fields:
        raise FileNotFoundError(f"No {stem}_np*.bin files in {directory}")
    return fields


def _error(candidate, reference):
    if len(candidate) != len(reference):
        return None
    if any(current.shape != baseline.shape for current, baseline in zip(candidate, reference)):
        return None
    current = np.concatenate(candidate)
    baseline = np.concatenate(reference)
    difference = current - baseline
    relative = np.linalg.norm(difference) / max(
        np.linalg.norm(baseline), np.finfo(np.float64).eps
    )
    return float(relative), float(np.max(np.abs(difference)))


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", choices=("python", "julia"))
    parser.add_argument("--tolerance", type=float, default=1.0e-6)
    args = parser.parse_args()

    reference = here / "dataout"
    candidate = here / f"{args.candidate}_run" / "dataout"
    passed = True
    print(f"{args.candidate.capitalize()} versus MATLAB")
    for stem in STEMS:
        result = _error(_load(candidate, stem), _load(reference, stem))
        if result is None:
            passed = False
            print(f"  {stem}: partition layouts differ")
            continue
        relative, maximum = result
        passed = passed and relative <= args.tolerance
        print(f"  {stem}: relative={relative:.6e}, maxabs={maximum:.6e}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
