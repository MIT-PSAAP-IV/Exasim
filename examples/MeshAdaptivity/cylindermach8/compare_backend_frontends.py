"""Compare MATLAB, Python, and Julia backend mesh-adaptivity runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = {
    "MATLAB": HERE / "backend_run",
    "Python": HERE / "python_backend_run",
    "Julia": HERE / "julia_backend_run",
}
NPE = 9
ND = 2
NC = 12
NCO = 2
NE = 51 * 32
TOLERANCE = 1.0e-6


def read_payload(path: Path, size: int) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float64)
    if values.size == size + 3:
        values = values[3:]
    if values.size != size:
        raise ValueError(f"{path} has {values.size} values; expected {size}.")
    return values


def read_structured(path: Path) -> list[np.ndarray]:
    values = np.fromfile(path, dtype=np.float64)
    field_count = int(round(values[0]))
    sizes = np.rint(values[1 : 1 + field_count]).astype(np.int64)
    offset = 1 + field_count
    fields = []
    for size in sizes:
        fields.append(values[offset : offset + size])
        offset += size
    if offset != values.size:
        raise ValueError(f"Malformed structured binary file: {path}")
    return fields


def errors(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float, float]:
    difference = candidate - reference
    l2 = float(np.linalg.norm(difference))
    linf = float(np.max(np.abs(difference)))
    relative = l2 / max(float(np.linalg.norm(reference)), np.finfo(float).tiny)
    return l2, linf, relative


def main() -> None:
    final_fields = {
        "adapted_xdg": ("out_meshadapt_xdg_np0.bin", NPE * ND * NE),
        "outudg": ("outudg_np0.bin", NPE * NC * NE),
        "outvdg": ("outvdg_np0.bin", NPE * NCO * NE),
    }
    values = {
        frontend: {
            field: read_payload(root / "dataout" / filename, size)
            for field, (filename, size) in final_fields.items()
        }
        for frontend, root in RUNS.items()
    }

    failed = False
    print("field         comparison       L2 error       Linf error     relative L2")
    for frontend in ("Python", "Julia"):
        for field in final_fields:
            l2, linf, relative = errors(
                values["MATLAB"][field], values[frontend][field]
            )
            print(
                f"{field:13s} MATLAB/{frontend:6s} "
                f"{l2:14.6e} {linf:14.6e} {relative:14.6e}"
            )
            failed |= relative > TOLERANCE

    matlab_sol = read_structured(RUNS["MATLAB"] / "datain" / "sol.bin")
    for frontend in ("Python", "Julia"):
        candidate_sol = read_structured(RUNS[frontend] / "datain" / "sol.bin")
        for index, field in ((1, "initial_xdg"), (2, "initial_udg"), (3, "initial_vdg")):
            l2, linf, relative = errors(matlab_sol[index], candidate_sol[index])
            print(
                f"{field:13s} MATLAB/{frontend:6s} "
                f"{l2:14.6e} {linf:14.6e} {relative:14.6e}"
            )
            failed |= relative > TOLERANCE

    if failed:
        raise SystemExit(f"Frontend mismatch exceeds relative tolerance {TOLERANCE:g}.")


if __name__ == "__main__":
    main()
