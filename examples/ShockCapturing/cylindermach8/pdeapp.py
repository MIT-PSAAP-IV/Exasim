"""Python frontend for the Mach-8 shock-capturing cylinder example."""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPOSITORY = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPOSITORY, "frontends", "Python"))

import exasim  # noqa: E402
from cylinder_mesh import initialize_solution, make_cylinder_mesh, wall_distance  # noqa: E402


def build_case():
    pde, _ = exasim.initializeexasim()
    pde["model"] = "ModelD"
    pde["modelfile"] = "pdemodel"
    pde["platform"] = "cpu"
    pde["mpiprocs"] = 1
    pde["hybrid"] = 1
    pde["porder"] = 4

    run_directory = os.path.join(HERE, "python_run")
    pde["datapath"] = run_directory
    pde["builddir"] = os.path.join(run_directory, ".exasim")
    pde["buildpath"] = pde["builddir"]
    # Match the MATLAB/Julia empty optional tails in app.bin.
    pde["flag"] = np.empty(0, dtype=np.int64)
    pde["problem"] = np.empty(0, dtype=np.int64)
    pde["factor"] = np.empty(0)
    pde["solversparam"] = np.empty(0)
    pde["stgdata"] = np.empty(0)
    pde["stgparam"] = np.empty(0)
    pde["stgib"] = np.empty(0, dtype=np.int64)
    pde["dae_dt"] = np.empty(0)

    gam = 1.4
    reynolds = 1.835e5
    prandtl = 0.71
    mach_inf = 8.03
    tref = 265.0
    twall = 300.0
    pinf = 1.0 / (gam * mach_inf**2)
    tinf = pinf / (gam - 1.0)
    rinf = 1.0
    ruinf = 1.0
    rvinf = 0.0
    rEinf = 0.5 + pinf / (gam - 1.0)

    pde["tau"] = np.array([1.0])
    pde["GMRESrestart"] = 200
    pde["linearsolvertol"] = 1.0e-6
    pde["linearsolveriter"] = 200
    pde["RBdim"] = 0
    pde["ppdegree"] = 0
    pde["NLtol"] = 1.0e-6
    pde["NLiter"] = 10
    pde["matvectol"] = 1.0e-6

    pde["AV"] = 1
    pde["AVcontinuationIter"] = 10
    pde["AVcontinuationLogScale"] = 1.0
    pde["AVcoeffStart"] = 0.060
    pde["AVcoeffEnd"] = 0.015
    pde["AVdistfunction"] = 1
    pde["AVsmoothingMethod"] = 1
    pde["AVHelmholtzCoeff"] = 0.025

    av_max_divergence = 2.0
    av_distance_coefficient = 10.0
    pde["physicsparam"] = np.array(
        [
            gam,
            reynolds,
            prandtl,
            mach_inf,
            rinf,
            ruinf,
            rvinf,
            rEinf,
            tinf,
            tref,
            twall,
            av_max_divergence,
            av_distance_coefficient,
            pde["AVcoeffStart"],
            pde["AVcoeffEnd"],
        ]
    )

    mesh = make_cylinder_mesh(pde["porder"])
    distance = wall_distance(mesh, pde["porder"])
    mesh["vdg"] = np.zeros((distance.shape[0], 2, distance.shape[2]), order="F")
    mesh["vdg"][:, 0:1, :] = distance
    mesh["udg"] = initialize_solution(mesh, distance, pde["physicsparam"])

    return pde, mesh


def main():
    pde, mesh = build_case()
    solution, pde, mesh, master, dmd = exasim.exasim(pde, mesh)[0:5]
    print(
        "Python solution:",
        solution.shape,
        "rho range =",
        (float(solution[:, 0, :, -1].min()), float(solution[:, 0, :, -1].max())),
    )
    return solution, pde, mesh, master, dmd


if __name__ == "__main__":
    main()
