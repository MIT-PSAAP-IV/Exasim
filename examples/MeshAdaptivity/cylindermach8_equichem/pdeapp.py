"""Python version of the equilibrium-air Mach-8 mesh-adaptivity example."""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPOSITORY = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
IDEAL_CASE = os.path.join(os.path.dirname(HERE), "cylindermach8")
SOURCE_INSTALL = os.path.join(os.path.dirname(REPOSITORY), "exasim_install")
if "EXASIM_PREFIX" not in os.environ and os.path.isfile(
    os.path.join(SOURCE_INSTALL, "lib", "cmake", "Exasim", "ExasimConfig.cmake")
):
    os.environ["EXASIM_PREFIX"] = SOURCE_INSTALL
sys.path.insert(0, HERE)
sys.path.insert(1, IDEAL_CASE)
sys.path.insert(2, os.path.join(REPOSITORY, "frontends", "Python"))

import exasim  # noqa: E402
from cylinder_mesh import make_cylinder_mesh, wall_distance  # noqa: E402


def _read_material_database(filename):
    with open(filename, "r", encoding="utf-8") as stream:
        header = np.fromstring(stream.readline(), sep=" ", dtype=np.int64)
        rows = np.loadtxt(stream)
    if header.size != 5 or header[0] != 2 or header[1] < 15:
        raise ValueError(f"Unexpected equilibrium-air database header in {filename}")
    if rows.shape[0] != int(np.prod(header[2:5])):
        raise ValueError("Material database row count does not match its header")
    xi = np.unique(rows[:, 0])
    energy = np.unique(rows[:, 1])
    properties = np.zeros((xi.size, energy.size, header[1]))
    xi_index = {value: index for index, value in enumerate(xi)}
    energy_index = {value: index for index, value in enumerate(energy)}
    for row in rows:
        properties[xi_index[row[0]], energy_index[row[1]], :] = row[2:]
    return xi, energy, properties


def _interpolate_xi(xi_grid, properties, xi):
    upper = int(np.searchsorted(xi_grid, xi, side="right"))
    lower = min(max(upper - 1, 0), xi_grid.size - 2)
    weight = (xi - xi_grid[lower]) / (xi_grid[lower + 1] - xi_grid[lower])
    return (1.0 - weight) * properties[lower, :, :] + weight * properties[lower + 1, :, :]


def _initial_solution(mesh, distance, database, xi_inf, temperature_inf,
                      wall_temperature, energy_ref, velocity_x, velocity_y):
    xi_grid, energy_grid, properties = database
    temperature_grid = _interpolate_xi(xi_grid, properties, xi_inf)[:, 1]
    target_temperature = temperature_inf + (
        wall_temperature - temperature_inf
    ) * np.exp(-10.0 * distance[:, 0, :])
    if target_temperature.min() < temperature_grid.min() or target_temperature.max() > temperature_grid.max():
        raise ValueError("Initial temperature is outside the material database")
    internal_energy = np.interp(
        target_temperature.reshape(-1), temperature_grid, energy_grid
    ).reshape(target_temperature.shape) / energy_ref
    density = np.ones_like(target_temperature)
    ux = velocity_x * np.tanh(10.0 * distance[:, 0, :])
    uy = velocity_y * np.tanh(10.0 * distance[:, 0, :])
    udg = np.empty((distance.shape[0], 4, distance.shape[2]), order="F")
    udg[:, 0, :] = density
    udg[:, 1, :] = density * ux
    udg[:, 2, :] = density * uy
    udg[:, 3, :] = density * (internal_energy + 0.5 * (ux * ux + uy * uy))
    return udg


def build_case(run_directory=None):
    pde, _ = exasim.initializeexasim()
    pde["model"] = "ModelD"
    pde["modelfile"] = "pdemodel"
    pde["platform"] = "cpu"
    pde["mpiprocs"] = 4
    pde["hybrid"] = 1
    pde["ncw"] = 15
    pde["porder"] = 2
    pde["pgauss"] = 2 * pde["porder"]
    pde["tau"] = np.array([1.0])
    pde["GMRESrestart"] = 200
    pde["linearsolvertol"] = 1.0e-7
    pde["linearsolveriter"] = 200
    pde["RBdim"] = 0
    pde["ppdegree"] = 20
    pde["NLtol"] = 1.0e-6
    pde["NLiter"] = 30
    pde["matvectol"] = 1.0e-6

    run_directory = run_directory or os.path.join(HERE, "python_run")
    os.makedirs(run_directory, exist_ok=True)
    pde["datapath"] = run_directory
    pde["builddir"] = os.path.join(run_directory, ".exasim")
    pde["buildpath"] = pde["builddir"]

    database_file = os.path.join(
        REPOSITORY,
        "apps",
        "materialdatabases",
        "equilibriumAir5logdensityexasim.dat",
    )
    pde["materialdatabase"] = database_file
    database = _read_material_database(database_file)

    mach_inf = 8.03
    temperature_inf = 265.0
    wall_temperature = 300.0
    length_ref = 1.0
    chemistry_conductivity = 0.0
    outlet_relaxation = 0.0
    energy_inf_dimensional = -1.097328937016845e5
    pressure_inf_dimensional = 97.467348012325772
    density_inf_dimensional = 0.001276095150944
    sound_speed_inf = 3.232136939449924e2
    velocity_inf_dimensional = mach_inf * sound_speed_inf
    xi_inf = np.log(density_inf_dimensional)

    density_ref = density_inf_dimensional
    velocity_ref = velocity_inf_dimensional
    pressure_ref = density_ref * velocity_ref**2
    energy_ref = velocity_ref**2
    transport_factor = 1.0

    pde["AV"] = 1
    pde["AVcontinuationIter"] = 10
    pde["AVcontinuationLogScale"] = 1.5
    pde["AVcoeffStart"] = 0.060
    pde["AVcoeffEnd"] = 0.008
    pde["AVdistfunction"] = 1
    pde["distanceboundaryconditions"] = np.array([3], dtype=np.int64)
    pde["AVsmoothingMethod"] = 1
    pde["AVHelmholtzCoeff"] = 0.025
    av_max_divergence = 2.0
    av_distance_coefficient = 30.0

    pde["meshadaptenabled"] = 1
    pde["meshadaptfield"] = 1
    pde["meshadaptalpha"] = 0.5
    pde["meshadaptHelmholtzCoeff"] = 5.0e-2
    pde["meshadaptforcescale"] = 0.15
    pde["meshadaptsmoothingpasses"] = 30
    pde["meshadaptboundaryconditions"] = np.array([2, 3, 3], dtype=np.int64)

    energy_inf = energy_inf_dimensional / energy_ref
    freestream = np.array([1.0, 1.0, 0.0, energy_inf + 0.5])
    pde["externalparam"] = freestream
    pde["physicsparam"] = np.array(
        [
            density_ref,
            velocity_ref,
            pressure_ref,
            energy_ref,
            length_ref,
            transport_factor,
            chemistry_conductivity,
            wall_temperature,
            pressure_inf_dimensional,
            outlet_relaxation,
            av_max_divergence,
            av_distance_coefficient,
            pde["AVcoeffStart"],
            pde["AVcoeffEnd"],
        ]
    )

    mesh = make_cylinder_mesh(
        pde["porder"], nx=51, ny=32, radial_decay=3.0, outer_offset=4.0
    )
    mesh["boundarycondition"] = np.array([3, 2, 1], dtype=np.int64)
    distance = wall_distance(mesh, pde["porder"])
    mesh["dist"] = distance
    mesh["vdg"] = np.zeros((distance.shape[0], 2, distance.shape[2]), order="F")
    mesh["vdg"][:, 0:1, :] = distance
    mesh["udg"] = _initial_solution(
        mesh,
        distance,
        database,
        xi_inf,
        temperature_inf,
        wall_temperature,
        energy_ref,
        1.0,
        0.0,
    )
    return pde, mesh


def main():
    pde, mesh = build_case()
    solution, pde, mesh, master, dmd = exasim.exasim(pde, mesh)[0:5]
    print(
        "Python equilibrium-air mesh adaptivity:",
        solution.shape,
        "output =",
        os.path.join(pde["datapath"], "dataout"),
    )
    return solution, pde, mesh, master, dmd


if __name__ == "__main__":
    main()
