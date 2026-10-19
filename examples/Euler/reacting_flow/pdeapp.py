"""Python frontend for the five-species reacting-flow cylinder example."""

import numpy as np
import exasim

from reacting_mesh import element_scales, make_mesh, wall_distance


def build_case(av_coefficient=0.2):
    pde, _ = exasim.initializeexasim()
    pde["model"], pde["modelfile"] = "ModelD", "pdemodel"
    pde["platform"], pde["mpiprocs"], pde["hybrid"] = "gpu", 1, 1
    pde["porder"], pde["pgauss"], pde["tau"] = 3, 6, np.array([1.0])
    species = np.array([1.00555e-9, 3.5318161606e-4, 1.58723747e-5, 1.16691299088e-3, 1.103201281e-5])
    momentum_x, momentum_y, total_energy = 9.213932, 0.0, 33563.20282790763
    rho_ref, velocity_ref, temperature_ref = 0.001547, 5976.0, 901.0
    energy_ref = rho_ref * velocity_ref**2
    mu_ref, kappa_ref, cp_ref, length_ref = 3.8196e-5, 0.076685370133649, 1182.1920097928833, 0.045
    reynolds = rho_ref * length_ref * velocity_ref / mu_ref
    prandtl = mu_ref * cp_ref / kappa_ref
    physics = np.zeros(23)
    physics[:5], physics[5:8] = species, [momentum_x, momentum_y, total_energy]
    physics[14], physics[18:23] = pde["porder"], [1.0e13, 1.0e4, prandtl, reynolds, 1.0]
    pde["physicsparam"] = physics
    pde["externalparam"] = np.array([rho_ref, velocity_ref, energy_ref, temperature_ref, mu_ref, kappa_ref, cp_ref, length_ref])
    pde["GMRESrestart"], pde["linearsolveriter"] = 200, 200
    pde["linearsolvertol"], pde["NLtol"], pde["NLiter"] = 1.0e-8, 1.0e-4, 30
    pde["RBdim"], pde["ppdegree"], pde["precMatrixType"] = 0, 20, 2
    pde["ptcMatrixType"], pde["matvectol"], pde["AV"] = 0, 1.0e-6, 1

    mesh = make_mesh(pde["porder"])
    distance = wall_distance(mesh, pde["porder"])
    npe, _, ne = mesh["dgnodes"].shape
    initial = np.empty((npe, 8, ne), order="F")
    initial[:, :5, :] = (species / rho_ref)[None, :, None]
    initial[:, 5, :] = momentum_x / (rho_ref * velocity_ref) * np.tanh(4.0 * distance[:, 0, :])
    initial[:, 6, :] = 0.0
    internal = (total_energy - 0.5 * np.sum(species) * velocity_ref**2) / energy_ref
    initial[:, 7, :] = internal + 0.5 * (initial[:, 5, :]**2 + initial[:, 6, :]**2) / np.sum(initial[:, :5, :], axis=1)
    mesh["udg"] = initial
    mesh["wdg"] = np.ones((npe, 1, ne), order="F")
    mesh["vdg"] = np.zeros((npe, 2, ne), order="F")
    mesh["vdg"][:, 0, :] = av_coefficient * np.tanh(25.0 * distance[:, 0, :])
    mesh["vdg"][:, 1, :] = element_scales(mesh) * np.tanh(1000.0 * distance[:, 0, :])
    return pde, mesh


def main():
    solution = None
    for coefficient in (0.2, 0.1, 0.05, 0.03, 0.015, 0.01):
        pde, mesh = build_case(coefficient)
        if solution is not None:
            mesh["udg"] = solution
            pde["NLtol"] = 1.0e-6
        solution, pde, mesh = exasim.exasim(pde, mesh)[0:3]
    print("Python reacting-flow solution:", solution.shape)
    return solution, pde, mesh


if __name__ == "__main__":
    main()
