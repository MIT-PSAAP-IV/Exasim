"""Python backend mesh-adaptivity counterpart of pdeapp_backend.m."""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPOSITORY = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SOURCE_INSTALL = os.path.join(os.path.dirname(REPOSITORY), "exasim_install")
if "EXASIM_PREFIX" not in os.environ and os.path.isfile(
    os.path.join(SOURCE_INSTALL, "lib", "cmake", "Exasim", "ExasimConfig.cmake")
):
    os.environ["EXASIM_PREFIX"] = SOURCE_INSTALL
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
    pde["porder"] = 2

    run_directory = os.path.join(HERE, "python_backend_run")
    os.makedirs(run_directory, exist_ok=True)
    pde["datapath"] = run_directory
    pde["builddir"] = os.path.join(run_directory, ".exasim")
    pde["buildpath"] = pde["builddir"]
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
    pde["AVcontinuationLogScale"] = 1.5
    pde["AVcoeffStart"] = 0.060
    pde["AVcoeffEnd"] = 0.01
    pde["AVdistfunction"] = 1
    pde["distanceboundaryconditions"] = np.array([3], dtype=np.int64)
    pde["AVsmoothingMethod"] = 1
    pde["AVHelmholtzCoeff"] = 0.025

    av_max_divergence = 2.0
    av_distance_coefficient = 100.0
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

    pde["meshadaptenabled"] = 1
    pde["meshadaptfield"] = 3
    pde["meshadaptalpha"] = 0.5
    pde["meshadaptHelmholtzCoeff"] = 5.0e-2
    pde["meshadaptforcescale"] = 0.2
    pde["meshadaptsmoothingpasses"] = 30
    pde["meshadaptboundaryconditions"] = np.array([2, 3, 3], dtype=np.int64)

    mesh = make_cylinder_mesh(
        pde["porder"], nx=51, ny=32, radial_decay=3.0, outer_offset=4.0
    )
    distance = wall_distance(mesh, pde["porder"])
    mesh["vdg"] = np.zeros((distance.shape[0], 2, distance.shape[2]), order="F")
    mesh["vdg"][:, 0:1, :] = distance
    mesh["udg"] = initialize_solution(mesh, distance, pde["physicsparam"])
    return pde, mesh


def main():
    pde, mesh = build_case()
    old_verification = os.environ.get("EXASIM_MESHADAPT_VERIFY")
    os.environ["EXASIM_MESHADAPT_VERIFY"] = "0"
    try:
        solution, pde, mesh, master, dmd = exasim.exasim(pde, mesh)[0:5]
    finally:
        if old_verification is None:
            os.environ.pop("EXASIM_MESHADAPT_VERIFY", None)
        else:
            os.environ["EXASIM_MESHADAPT_VERIFY"] = old_verification

    adapted_file = os.path.join(
        pde["datapath"], "dataout", "out_meshadapt_xdg_np0.bin"
    )
    adapted_nodes = np.fromfile(adapted_file, dtype=np.float64).reshape(
        mesh["dgnodes"].shape, order="F"
    )
    mesh["dgnodes"] = adapted_nodes
    print(
        "Python backend mesh adaptivity:",
        solution.shape,
        "adapted x range =",
        (float(adapted_nodes[:, 0, :].min()), float(adapted_nodes[:, 0, :].max())),
    )
    return solution, pde, mesh, master, dmd


if __name__ == "__main__":
    main()
