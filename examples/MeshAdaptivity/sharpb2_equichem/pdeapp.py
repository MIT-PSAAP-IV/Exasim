"""Python version of the Sharp-B equilibrium-air mesh-adaptivity case."""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPOSITORY = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SHARPB_IDEAL_CASE = os.path.join(os.path.dirname(HERE), "sharpb2_idealgas")
SOURCE_INSTALL = os.path.join(os.path.dirname(REPOSITORY), "exasim_install")
if "EXASIM_PREFIX" not in os.environ and os.path.isfile(
    os.path.join(SOURCE_INSTALL, "lib", "cmake", "Exasim", "ExasimConfig.cmake")
):
    os.environ["EXASIM_PREFIX"] = SOURCE_INSTALL
sys.path.insert(0, HERE)
sys.path.insert(1, SHARPB_IDEAL_CASE)
sys.path.insert(2, os.path.join(REPOSITORY, "frontends", "Python"))

import exasim  # noqa: E402
from postprocess_surfacequantities import postprocess_surfacequantities  # noqa: E402
from sharpb2_mesh import make_sharpb2_mesh, wall_distance  # noqa: E402


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


def _interpolate(database, xi_value, energy_value):
    xi, energy, properties = database
    i = min(max(np.searchsorted(xi, xi_value, side="right") - 1, 0), xi.size - 2)
    j = min(
        max(np.searchsorted(energy, energy_value, side="right") - 1, 0),
        energy.size - 2,
    )
    tx = (xi_value - xi[i]) / (xi[i + 1] - xi[i])
    te = (energy_value - energy[j]) / (energy[j + 1] - energy[j])
    return (
        (1.0 - tx) * (1.0 - te) * properties[i, j, :]
        + tx * (1.0 - te) * properties[i + 1, j, :]
        + (1.0 - tx) * te * properties[i, j + 1, :]
        + tx * te * properties[i + 1, j + 1, :]
    )


def _energy_for_temperature(database, xi_value, temperature):
    energy = database[1]
    table_temperature = np.array(
        [_interpolate(database, xi_value, value)[1] for value in energy]
    )
    target = np.asarray(temperature)
    if target.min() < table_temperature.min() or target.max() > table_temperature.max():
        raise ValueError("Initial temperature is outside the material database")
    result = np.interp(target.reshape(-1), table_temperature, energy)
    return result.reshape(target.shape)


def _initial_solution(distance, database, xi_inf, temperature_inf,
                      wall_temperature, energy_ref, slope):
    target_temperature = temperature_inf + (
        wall_temperature - temperature_inf
    ) * np.exp(-slope * distance[:, 0, :])
    internal_energy = (
        _energy_for_temperature(database, xi_inf, target_temperature) / energy_ref
    )
    velocity_z = np.tanh(slope * distance[:, 0, :])
    velocity_r = np.zeros_like(velocity_z)
    udg = np.empty((distance.shape[0], 4, distance.shape[2]), order="F")
    udg[:, 0, :] = 1.0
    udg[:, 1, :] = velocity_z
    udg[:, 2, :] = velocity_r
    udg[:, 3, :] = internal_energy + 0.5 * (
        velocity_z * velocity_z + velocity_r * velocity_r
    )
    return udg


def build_case(run_directory=None):
    pde, _ = exasim.initializeexasim()
    pde["model"] = "ModelD"
    pde["modelfile"] = "pdemodel"
    pde["platform"] = "cpu"
    pde["mpiprocs"] = 8
    pde["porder"] = 2
    pde["pgauss"] = 2 * pde["porder"]
    pde["hybrid"] = 1
    pde["debugmode"] = 0
    pde["nd"] = 2
    pde["extendedW"] = 1
    pde["ncw"] = 15
    pde["saveParaview"] = 1
    pde["saveSolBouFreq"] = 1
    pde["ibs"] = 3
    pde["saveSolBouLoc"] = 1

    run_directory = run_directory or os.path.join(HERE, "python_run")
    os.makedirs(run_directory, exist_ok=True)
    pde["datapath"] = run_directory
    pde["builddir"] = os.path.join(run_directory, ".exasim")
    pde["buildpath"] = pde["builddir"]

    database_file = os.path.join(
        REPOSITORY, "apps", "materialdatabases", "equilibriumAir5logdensityexasim.dat"
    )
    pde["materialdatabase"] = database_file
    database = _read_material_database(database_file)

    mach_inf = 21.38
    reynolds = 9.84e5
    temperature_inf = 260.6
    wall_temperature = 1400.0
    length_ref = 1.0
    chemistry_conductivity = 0.0
    outlet_relaxation = 0.0

    density_inf = 1.0e-3
    for _ in range(12):
        xi_inf = np.log(density_inf)
        energy_inf = float(_energy_for_temperature(database, xi_inf, temperature_inf))
        properties_inf = _interpolate(database, xi_inf, energy_inf)
        velocity_inf = mach_inf * properties_inf[5]
        density_new = reynolds * properties_inf[2] / (velocity_inf * length_ref)
        if abs(density_new - density_inf) <= 1.0e-12 * max(1.0, density_inf):
            density_inf = density_new
            break
        density_inf = density_new

    xi_inf = np.log(density_inf)
    energy_inf = float(_energy_for_temperature(database, xi_inf, temperature_inf))
    properties_inf = _interpolate(database, xi_inf, energy_inf)
    pressure_inf = properties_inf[0]
    sound_speed_inf = properties_inf[5]
    velocity_inf = mach_inf * sound_speed_inf

    density_ref = density_inf
    velocity_ref = velocity_inf
    pressure_ref = density_ref * velocity_ref**2
    energy_ref = velocity_ref**2

    pde["AV"] = 2
    pde["AVcontinuationIter"] = 9
    pde["AVcontinuationLogScale"] = 2.0
    pde["AVcoeffStart"] = 0.005
    pde["AVcoeffEnd"] = 0.00005
    pde["AVdistfunction"] = 1
    pde["distanceboundaryconditions"] = np.array([3], dtype=np.int64)
    pde["AVsmoothingMethod"] = 1
    pde["AVHelmholtzCoeff"] = 0.001
    av_max_divergence = 20.0
    av_distance_coefficient = 100.0

    pde["meshadaptenabled"] = 1
    pde["meshadaptfield"] = 2
    pde["meshadaptavcomponent"] = 1
    pde["meshadaptalpha"] = 0.1
    pde["meshadaptqmin"] = 0.2
    pde["meshadaptqmax"] = 0.8
    pde["meshadaptHelmholtzCoeff"] = 0.001
    pde["meshadaptforcescale"] = 0.7
    pde["meshadaptsmoothingpasses"] = 30
    pde["meshadaptboundaryconditions"] = np.array([3, 3, 3, 2, 3], dtype=np.int64)

    specific_energy_inf = energy_inf / energy_ref
    freestream = np.array([1.0, 1.0, 0.0, specific_energy_inf + 0.5])
    pde["externalparam"] = freestream
    pde["physicsparam"] = np.array([
        density_ref, velocity_ref, pressure_ref, energy_ref, length_ref, 1.0,
        chemistry_conductivity, wall_temperature, pressure_inf, outlet_relaxation,
        av_max_divergence, av_distance_coefficient,
        pde["AVcoeffStart"], pde["AVcoeffEnd"],
    ])

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

    mesh = make_sharpb2_mesh(pde["porder"])
    mesh["boundarycondition"] = np.array([5, 1, 1, 3, 2], dtype=np.int64)
    distance = wall_distance(mesh, pde["porder"])
    mesh["vdg"] = np.zeros((distance.shape[0], 3, distance.shape[2]), order="F")
    mesh["vdg"][:, 0:1, :] = distance
    mesh["udg"] = _initial_solution(
        distance, database, xi_inf, temperature_inf, wall_temperature,
        energy_ref, av_distance_coefficient,
    )

    print("\nEquilibrium-air Sharp-B setup")
    print(
        f"  rho_inf = {density_inf:.10g} kg/m^3, p_inf = {pressure_inf:.10g} Pa, "
        f"T_inf = {temperature_inf:.10g} K"
    )
    print(
        f"  a_inf = {sound_speed_inf:.10g} m/s, U_inf = {velocity_inf:.10g} m/s, "
        f"Mach = {mach_inf:.8g}, Re = {reynolds:.8g}"
    )
    return pde, mesh


def main():
    pde, mesh = build_case()
    solution, pde, mesh, master, dmd = exasim.exasim(pde, mesh)[0:5]
    print(
        "Python equilibrium-air Sharp-B mesh adaptivity:", solution.shape,
        "output =", os.path.join(pde["datapath"], "dataout"),
    )
    postprocess_surfacequantities(pde)
    return solution, pde, mesh, master, dmd


if __name__ == "__main__":
    main()
