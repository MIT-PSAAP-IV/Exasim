"""Python axisymmetric Poisson solve on the tracked Orion geometry."""

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
    pde["porder"], pde["pgauss"] = 3, 6
    pde["physicsparam"], pde["tau"] = np.array([1.0, 0.0, 1.0, 1.0]), np.array([1.0])
    pde["linearsolvertol"], pde["ppdegree"], pde["RBdim"] = 1.0e-8, 1, 0
    run_directory = Path(run_directory or HERE / "python_run")
    run_directory.mkdir(parents=True, exist_ok=True)
    pde["datapath"] = str(run_directory)
    pde["builddir"] = str(run_directory / ".exasim")

    geometry = run_directory / "domain6.geo"
    shutil.copyfile(HERE / "domain6.geo", geometry)
    mesh["p"], mesh["t"] = exasim.Mesh.gmshcall(pde, str(geometry.with_suffix("")), 2, 1)[0:2]
    xmin, xmax = mesh["p"][0].min(), mesh["p"][0].max()
    ymin = mesh["p"][1].min()
    mesh["boundaryexpr"] = [
        lambda p: np.abs(p[1] - ymin) < 1.0e-6,
        lambda p: np.abs(p[0] - xmax) < 1.0e-6,
        lambda p: p[0] < xmin + 1.0e-6,
        lambda p: np.abs(p[0]) < xmax + 1.0,
    ]
    mesh["boundarycondition"] = np.array([1, 4, 3, 2], dtype=np.int64)
    return pde, mesh


def main():
    pde, mesh = build_case()
    solution, pde, mesh = exasim.exasim(pde, mesh)[0:3]
    print("Python Orion solution:", solution.shape)
    return solution, pde, mesh


if __name__ == "__main__":
    main()
