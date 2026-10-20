"""Python frontend for the Mach-5.38 rounded-nose flat-plate example."""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPOSITORY = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPOSITORY, "frontends", "Python"))

import exasim  # noqa: E402
from auplate_mesh import make_auplate_mesh, wall_distance  # noqa: E402


def build_case():
    pde, _ = exasim.initializeexasim()
    pde["model"] = "ModelD"
    pde["modelfile"] = "pdemodel"
    pde["platform"] = "cpu"
    pde["mpiprocs"] = 8
    pde["hybrid"] = 1
    pde["porder"] = 2

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
    reynolds = 8.9e6
    prandtl = 0.71
    mach_inf = 5.38
    tref = 1020.0
    twall = 296.0
    pinf = 1.0 / (gam * mach_inf**2)
    tinf = pinf / (gam - 1.0)
    rinf = 1.0
    ruinf = 1.0
    rvinf = 0.0
    rEinf = 0.5 + pinf / (gam - 1.0)

    pde["tau"] = np.array([10.0])
    pde["GMRESrestart"] = 250
    pde["linearsolvertol"] = 1.0e-6
    pde["linearsolveriter"] = 500
    pde["RBdim"] = 0
    pde["ppdegree"] = 0
    pde["NLtol"] = 1.0e-6
    pde["NLiter"] = 5
    pde["matvectol"] = 1.0e-6

    pde["AV"] = 1
    pde["AVcontinuationIter"] = 10
    pde["AVcontinuationLogScale"] = 2.0
    pde["AVcoeffStart"] = 0.02
    pde["AVcoeffEnd"] = 1.0e-5
    pde["AVdistfunction"] = 1
    pde["distanceboundaryconditions"] = np.array([3], dtype=int)
    pde["AVsmoothingMethod"] = 1
    pde["AVHelmholtzCoeff"] = 1.0e-4

    av_max_divergence = 4.0
    av_distance_coefficient = 1.5e3
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

    mesh = make_auplate_mesh(pde["porder"])
    distance = wall_distance(mesh)
    mesh["vdg"] = np.zeros((distance.shape[0], 2, distance.shape[2]), order="F")
    mesh["vdg"][:, 0:1, :] = distance
    npe, _, ne = mesh["dgnodes"].shape
    initial = np.empty((npe, 4, ne), order="F")
    initial[:, 0, :] = rinf
    initial[:, 1, :] = ruinf * np.tanh(100.0 * distance[:, 0, :])
    initial[:, 2, :] = rvinf * np.tanh(100.0 * distance[:, 0, :])
    near_wall_temperature = tinf * (twall / tref - 1.0) * np.exp(-100.0 * distance[:, 0, :]) + tinf
    initial[:, 3, :] = near_wall_temperature + 0.5 * (initial[:, 1, :]**2 + initial[:, 2, :]**2)
    mesh["udg"] = initial

    return pde, mesh


def main():
    pde, mesh = build_case()
    solution, pde, mesh, master, dmd = exasim.exasim(pde, mesh)[0:5]
    print(
        "Python rounded-nose flat-plate solution:",
        solution.shape,
        "rho range =",
        (float(solution[:, 0, :, -1].min()), float(solution[:, 0, :, -1].max())),
    )
    return solution, pde, mesh, master, dmd


if __name__ == "__main__":
    main()
