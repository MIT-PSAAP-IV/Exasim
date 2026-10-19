"""Python Poisson solve on a periodic HCA flared-plate extrusion."""

from pathlib import Path
import shutil
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
sys.path.insert(0, str(HERE)); sys.path.insert(1, str(REPOSITORY / "frontends" / "Python"))
import exasim  # noqa: E402


def build_case(run_directory=None):
    pde, mesh = exasim.initializeexasim()
    pde["model"], pde["modelfile"] = "ModelD", "pdemodel"
    pde["platform"], pde["mpiprocs"], pde["hybrid"] = "cpu", 4, 1
    pde["porder"], pde["pgauss"] = 2, 4
    pde["physicsparam"], pde["tau"] = np.array([1.0, 0.0, 1.0, 1.0]), np.array([1.0])
    pde["linearsolvertol"], pde["ppdegree"] = 1.0e-8, 0
    pde["linearsolveriter"], pde["GMRESrestart"] = 400, 200
    pde["preconditioner"], pde["RBdim"] = 1, 0
    run_directory = Path(run_directory or HERE / "python_run"); run_directory.mkdir(parents=True, exist_ok=True)
    pde["datapath"], pde["builddir"] = str(run_directory), str(run_directory / ".exasim")
    geometry = run_directory / "hca_flare.geo"
    shutil.copyfile(HERE / "hca_flare.geo", geometry)
    mesh["p"], mesh["t"] = exasim.Mesh.gmshcall(pde, str(geometry.with_suffix("")), 3, 1)[0:2]
    xmin, xmax = mesh["p"][0].min(), mesh["p"][0].max()
    ymax = mesh["p"][1].max()
    corner_x, lower_y = 0.04988, 0.10645
    lower_slope = (0.17000 - lower_y) / (xmax - corner_x)
    mesh["boundaryexpr"] = [
        lambda p: np.abs(p[2]) < 1.0e-7,
        lambda p: np.abs(p[2] - 0.05) < 1.0e-7,
        lambda p: (np.abs(p[1] - lower_y) < 1.0e-7) & (p[0] <= corner_x + 1.0e-7),
        lambda p: np.abs(p[1] - (lower_y + lower_slope * (p[0] - corner_x))) < 1.0e-7,
        lambda p: p[0] > xmax - 1.0e-6,
        lambda p: (p[1] > ymax - 1.0e-6) & (p[0] >= corner_x - 1.0e-7),
        lambda p: (p[1] > ymax - 1.0e-6) & (p[0] <= corner_x + 1.0e-7),
        lambda p: p[0] < xmin + 1.0e-6,
    ]
    mesh["boundarycondition"] = np.array([3, 3, 1, 1, 3, 2, 2, 2], dtype=np.int64)
    return pde, mesh


def main():
    pde, mesh = build_case(); solution, pde, mesh = exasim.exasim(pde, mesh)[0:3]
    print("Python HCA flared-plate Poisson solution:", solution.shape)
    return solution, pde, mesh


if __name__ == "__main__": main()
