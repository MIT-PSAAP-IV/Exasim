"""Regression checks for zero-based Python partition solution readback."""

import os
import sys
import tempfile

try:
    import numpy as np
except ImportError:
    print("SKIP: numpy unavailable")
    sys.exit(77)


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "frontends", "Python"))

from exasim.Postprocessing.getsolution import getsolution  # noqa: E402
from exasim.Postprocessing.getsolutions import getsolutions  # noqa: E402


def main():
    partitions = [
        {"elempartpts": np.array([1, 0]), "elempart": np.array([0])},
        {"elempartpts": np.array([1, 0]), "elempart": np.array([1])},
        {"elempartpts": np.array([1, 0]), "elempart": np.array([2])},
    ]
    with tempfile.TemporaryDirectory() as directory:
        header_base = os.path.join(directory, "outudg")
        raw_base = os.path.join(directory, "outvdg")
        for rank, value in enumerate((11.0, 22.0, 33.0)):
            np.array([1.0, 1.0, 1.0, value], dtype=np.float64).tofile(
                f"{header_base}_np{rank}.bin"
            )
            np.array([value], dtype=np.float64).tofile(f"{raw_base}_np{rank}.bin")

        assembled = getsolutions(header_base, partitions)
        raw = getsolution(raw_base, partitions, 1)
        expected = np.array([11.0, 22.0, 33.0])
        assert np.array_equal(assembled[0, 0, :, 0], expected)
        assert np.array_equal(raw[0, 0, :], expected)

    print("PASS: test_solution_readback.py")


if __name__ == "__main__":
    main()
