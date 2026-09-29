"""Compare MATLAB, Python, and Julia outputs for the Sharp-B case."""

from __future__ import annotations

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = {
    "MATLAB": os.path.join(HERE, "dataout"),
    "Python": os.path.join(HERE, "python_run", "dataout"),
    "Julia": os.path.join(HERE, "julia_run", "dataout"),
}


def _load(directory, stem):
    arrays = []
    rank = 0
    while True:
        filename = os.path.join(directory, f"{stem}_np{rank}.bin")
        if not os.path.isfile(filename):
            break
        arrays.append(np.fromfile(filename, dtype=np.float64))
        rank += 1
    if not arrays:
        raise FileNotFoundError(f"No {stem}_np*.bin files in {directory}")
    return arrays


def _error(candidate, reference):
    if len(candidate) != len(reference):
        return None
    differences, reference_values = [], []
    for current, baseline in zip(candidate, reference):
        if current.shape != baseline.shape:
            return None
        differences.append(current - baseline)
        reference_values.append(baseline)
    difference = np.concatenate(differences)
    baseline = np.concatenate(reference_values)
    relative = np.linalg.norm(difference) / max(
        np.linalg.norm(baseline), np.finfo(np.float64).eps
    )
    return relative, np.max(np.abs(difference))


def main():
    for frontend in ("Python", "Julia"):
        print(f"{frontend} versus MATLAB")
        for stem in ("outxdg", "outvdg", "outudg"):
            result = _error(_load(OUTPUTS[frontend], stem), _load(OUTPUTS["MATLAB"], stem))
            if result is None:
                print(f"  {stem}: partition layouts differ; direct rankwise comparison unavailable")
            else:
                relative, maximum = result
                print(f"  {stem}: relative={relative:.6e}, maxabs={maximum:.6e}")


if __name__ == "__main__":
    main()
