"""Python version of the axisymmetric ideal-gas ISOQ mesh-adaptivity case."""

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
sys.path.insert(1, os.path.join(REPOSITORY, "frontends", "Python"))

import exasim  # noqa: E402
from isoq_mesh import make_isoq_mesh, wall_distance  # noqa: E402


def build_case(run_directory=None):
    pde, _ = exasim.initializeexasim()
    pde["model"] = "ModelD"
    pde["modelfile"] = "pdemodel"
    pde["platform"] = "cpu"
    pde["mpiprocs"] = 4
    pde["porder"] = 2
    pde["pgauss"] = 2 * pde["porder"]
    pde["hybrid"] = 1
    pde["debugmode"] = 0
    pde["saveParaview"] = 1

    run_directory = run_directory or os.path.join(HERE, "python_run")
    os.makedirs(run_directory, exist_ok=True)
    pde["datapath"] = run_directory
    pde["builddir"] = os.path.join(run_directory, ".exasim")
    pde["buildpath"] = pde["builddir"]

    gam, reynolds, prandtl, mach_inf = 1.4, 1.56e5, 0.71, 7.6
    reference_temperature, wall_temperature = 266.5, 300.0
    pressure_inf = 1.0 / (gam * mach_inf**2)
    temperature_inf = pressure_inf / (gam - 1.0)
    density_inf = 1.0
    momentum_x_inf, momentum_y_inf = 1.0, 0.0
    energy_inf = 0.5 + pressure_inf / (gam - 1.0)

    pde["AV"] = 1
    pde["AVcontinuationIter"] = 6
    pde["AVcontinuationLogScale"] = 1.5
    pde["AVcoeffStart"] = 0.002
    pde["AVcoeffEnd"] = 0.000005
    pde["AVdistfunction"] = 1
    pde["distanceboundaryconditions"] = np.array([3], dtype=np.int64)
    pde["AVsmoothingMethod"] = 1
    pde["AVHelmholtzCoeff"] = 0.001
    av_max_divergence, av_distance_coefficient = 50.0, 100.0

    pde["meshadaptenabled"] = 1
    pde["meshadaptfield"] = 2
    pde["meshadaptavcomponent"] = 1
    pde["meshadaptalpha"] = 0.5
    pde["meshadaptqmin"] = 0.2
    pde["meshadaptqmax"] = 0.8
    pde["meshadaptHelmholtzCoeff"] = 0.001
    pde["meshadaptforcescale"] = 0.25
    pde["meshadaptsmoothingpasses"] = 30
    pde["meshadaptboundaryconditions"] = np.array([3, 3, 3, 2], dtype=np.int64)

    pde["physicsparam"] = np.array(
        [
            gam,
            reynolds,
            prandtl,
            mach_inf,
            density_inf,
            momentum_x_inf,
            momentum_y_inf,
            energy_inf,
            temperature_inf,
            reference_temperature,
            wall_temperature,
            av_max_divergence,
            av_distance_coefficient,
            pde["AVcoeffStart"],
            pde["AVcoeffEnd"],
        ]
    )
    pde["tau"] = np.array([4.0])
    pde["GMRESrestart"] = 250
    pde["GMRESortho"] = 1
    pde["linearsolvertol"] = 1.0e-6
    pde["linearsolveriter"] = 500
    pde["preconditioner"] = 1
    pde["RBdim"] = 0
    pde["ppdegree"] = 0
    pde["NLtol"] = 1.0e-6
    pde["NLiter"] = 10
    pde["matvectol"] = 1.0e-6

    mesh = make_isoq_mesh(pde["porder"], 5.0e-4)
    distance = wall_distance(mesh, pde["porder"])
    mesh["vdg"] = np.zeros((distance.shape[0], 2, distance.shape[2]), order="F")
    mesh["vdg"][:, 0:1, :] = distance
    npe, _, ne = mesh["dgnodes"].shape
    mesh["udg"] = np.empty((npe, 4, ne), dtype=np.float64, order="F")
    mesh["udg"][:, 0, :] = density_inf
    mesh["udg"][:, 1, :] = momentum_x_inf * np.tanh(
        av_distance_coefficient * distance[:, 0, :]
    )
    mesh["udg"][:, 2, :] = momentum_y_inf
    near_wall_temperature = temperature_inf * (
        wall_temperature / reference_temperature - 1.0
    ) * np.exp(-av_distance_coefficient * distance[:, 0, :]) + temperature_inf
    mesh["udg"][:, 3, :] = near_wall_temperature + 0.5 * (
        mesh["udg"][:, 1, :] ** 2 + mesh["udg"][:, 2, :] ** 2
    )
    return pde, mesh


def main():
    pde, mesh = build_case()
    solution, pde, mesh, master, dmd = exasim.exasim(pde, mesh)[0:5]
    print(
        "Python ISOQ mesh adaptivity:", solution.shape,
        "output =", os.path.join(pde["datapath"], "dataout"),
    )
    return solution, pde, mesh, master, dmd


if __name__ == "__main__":
    main()
