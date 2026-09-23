"""Compare per-iteration MATLAB and backend mesh-adaptivity diagnostics."""

from array import array
from math import sqrt
from pathlib import Path


HERE = Path(__file__).resolve().parent
FRONTEND = HERE / "frontend_run" / "verification"
BACKEND = HERE / "backend_run" / "dataout"
FIELDS = (
    "wall_distance", "av_raw", "av_smoothed", "flow_solution",
    "sensor_scalar", "sensor_raw", "field1", "field2", "eta",
    "displacement", "xdg",
)


def read(path: Path) -> array:
    values = array("d")
    with path.open("rb") as stream:
        values.fromfile(stream, path.stat().st_size // values.itemsize)
    return values


def errors(reference: array, candidate: array) -> tuple[float, float, float]:
    if len(reference) != len(candidate):
        raise ValueError(f"size mismatch: {len(reference)} != {len(candidate)}")
    squared_error = 0.0
    squared_reference = 0.0
    maximum = 0.0
    for expected, actual in zip(reference, candidate):
        difference = actual - expected
        squared_error += difference * difference
        squared_reference += expected * expected
        maximum = max(maximum, abs(difference))
    absolute = sqrt(squared_error)
    denominator = sqrt(squared_reference)
    return absolute, maximum, absolute / max(denominator, 2.2250738585072014e-308)


def main() -> None:
    print("iteration field              L2 error       Linf error     relative L2")
    for iteration in range(1, 11):
        for field in FIELDS:
            reference = FRONTEND / f"aviter{iteration}_{field}.bin"
            backend_field = f"iter1_{field}" if field in ("displacement", "xdg") else field
            candidate = BACKEND / f"out_meshadapt_aviter{iteration}_{backend_field}_np0.bin"
            if not reference.is_file() or not candidate.is_file():
                continue
            l2, linf, relative = errors(read(reference), read(candidate))
            print(f"{iteration:9d} {field:18s} {l2:14.6e} {linf:14.6e} {relative:14.6e}")


if __name__ == "__main__":
    main()
