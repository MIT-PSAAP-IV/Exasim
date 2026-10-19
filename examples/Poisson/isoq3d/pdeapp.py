"""Python 3D ISOQ-shell Poisson example."""

from pathlib import Path
import shutil
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(REPOSITORY / "frontends" / "Python"))
import exasim  # noqa: E402


def build_case(run_directory=None):
    pde, mesh = exasim.initializeexasim()
    pde["model"], pde["modelfile"] = "ModelD", "pdemodel"
    pde["platform"], pde["mpiprocs"], pde["hybrid"] = "cpu", 4, 1
    pde["porder"], pde["pgauss"] = 2, 4
    pde["physicsparam"], pde["tau"] = np.array([1.0, 0.0, 1.0]), np.array([2.0])
    pde["linearsolveriter"], pde["GMRESrestart"] = 151, 150
    pde["NLtol"], pde["linearsolvertol"] = 1.0e-4, 1.0e-6
    pde["GMRESortho"], pde["RBdim"], pde["NLiter"], pde["ppdegree"] = 1, 0, 2, 20
    run_directory = Path(run_directory or HERE / "python_run")
    run_directory.mkdir(parents=True, exist_ok=True)
    pde["datapath"], pde["builddir"] = str(run_directory), str(run_directory / ".exasim")
    geometry = run_directory / "isoq3d.geo"
    shutil.copyfile(HERE / "isoq3d.geo", geometry)
    mesh["p"], mesh["t"] = exasim.Mesh.gmshcall(pde, str(geometry.with_suffix("")), 3, 0)[0:2]
    radius = np.hypot(mesh["p"][1], mesh["p"][2])
    mesh["boundaryexpr"] = [
        lambda p: np.abs(p[1]) < 1.0e-6,
        lambda p: np.abs(p[2]) < 1.0e-6,
        lambda p: np.abs(p[0] - 1.0) < 1.0e-6,
        lambda p: np.abs(np.hypot(p[1], p[2]) - 0.05) < 1.0e-5,
        lambda p: np.abs(np.hypot(p[1], p[2]) - radius.max()) < 1.0e-5,
    ]
    mesh["boundarycondition"] = np.array([3, 3, 3, 1, 2], dtype=np.int64)
    return pde, mesh


def main():
    pde, mesh = build_case()
    solution, pde, mesh = exasim.exasim(pde, mesh)[0:3]
    print("Python ISOQ3D solution:", solution.shape)
    return solution, pde, mesh


if __name__ == "__main__":
    main()
