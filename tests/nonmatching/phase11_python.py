"""Python end-to-end coverage for 3D conformal and non-matching interfaces.

The C++ unit matrix covers distributed geometry, orientation, and failure
protocols. This test exercises the installed solver through the Python frontend
on both tetrahedral and hexahedral meshes without requiring MATLAB.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path


SKIP = 77
ROOT = Path(__file__).resolve().parents[2]
PREFIX = Path(os.environ.get("EXASIM_INSTALL", os.environ.get("EXASIM_PREFIX", "")))


def skip(message: str) -> None:
    print(f"SKIP: {message}")
    raise SystemExit(SKIP)


if not PREFIX or not (PREFIX / "lib" / "cmake" / "Exasim" / "ExasimConfig.cmake").is_file():
    skip("installed Exasim prefix is unavailable")
if shutil.which("mpirun") is None and shutil.which("mpiexec") is None:
    skip("MPI launcher is unavailable")

try:
    import numpy
    import scipy  # noqa: F401
    import sympy  # noqa: F401
except ImportError as error:
    skip(f"Python dependency is unavailable: {error.name}")

os.environ["EXASIM_PREFIX"] = str(PREFIX)
sys.path.insert(0, str(ROOT / "frontends" / "Python"))

try:
    import exasim
except ImportError as error:
    skip(f"Python frontend is unavailable: {error}")


def run_case(case_name: str, interface_type: int, work: Path):
    case = ROOT / "examples" / "CoupledProblems" / "Poisson" / case_name
    if not case.is_dir():
        raise RuntimeError(f"missing Python example {case}")

    pde = [None, None]
    mesh = [None, None]
    namespace = {"exasim": exasim, "mesh": mesh, "numpy": numpy, "pde": pde}
    sys.path.insert(0, str(case))
    try:
        for model in (1, 2):
            source = case / f"pdeapp{model}.py"
            exec(compile(source.read_text(encoding="utf-8"), str(source), "exec"), namespace)
    finally:
        sys.path.pop(0)

    run = work / f"{case_name}_type{interface_type}"
    run.mkdir(parents=True)
    for model in range(2):
        pde[model]["interfacetype"] = interface_type
        pde[model]["datapath"] = str(run / "data")
        # Type 0 and type 1 share generated model kernels; reuse the family build.
        pde[model]["builddir"] = str(work / f"{case_name}_build")
        pde[model]["buildpath"] = pde[model]["builddir"]

    previous = Path.cwd()
    sys.path.insert(0, str(case))
    try:
        os.chdir(run)
        solution = exasim.exasim(pde, mesh)[0]
    finally:
        os.chdir(previous)
        sys.path.pop(0)

    arrays = [numpy.asarray(value) for value in solution]
    if len(arrays) != 2 or any(value.size == 0 for value in arrays):
        raise AssertionError(f"{case_name} type {interface_type}: incomplete solution")
    if any(not numpy.all(numpy.isfinite(value)) for value in arrays):
        raise AssertionError(f"{case_name} type {interface_type}: non-finite solution")
    return arrays


def relative_difference(left, right) -> float:
    values = []
    for a, b in zip(left, right):
        if a.shape != b.shape:
            raise AssertionError(f"solution shapes differ: {a.shape} != {b.shape}")
        values.append(numpy.linalg.norm((a - b).ravel()) /
                      max(numpy.linalg.norm(a.ravel()), 1.0))
    return max(values)


with tempfile.TemporaryDirectory(prefix="exasim_phase11_python_") as directory:
    work = Path(directory)
    for family in ("matching3d_tet", "matching3d_hex"):
        conformal = run_case(family, 0, work)
        nonmatching = run_case(family, 1, work)
        relative = relative_difference(conformal, nonmatching)
        if relative >= 5.0e-12:
            raise AssertionError(
                f"{family}: type-0/type-1 relative difference {relative:.17e} exceeds 5e-12"
            )
        print(f"PASS {family}: type-0/type-1 relative difference {relative:.3e}")

    # Exercise an actually non-matching interface, not only the type-1 path on
    # coincident meshes. The finite/nonempty gates above catch failed solves.
    run_case("nonmatching3d_tet", 1, work)
    print("PASS nonmatching3d_tet: unequal-interface Python solve")

print("PASS phase11_python")
