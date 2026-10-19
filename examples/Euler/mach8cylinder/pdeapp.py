"""Python frontend for the legacy Mach-8.98 cylinder setup."""

import numpy as np
import exasim

from cylinder_mesh import curved_map_scale, initialize_solution, make_cylinder_mesh, wall_distance


pde, _ = exasim.initializeexasim()
pde["model"], pde["modelfile"] = "ModelD", "pdemodel"
pde["platform"], pde["mpiprocs"] = "gpu", 1
pde["porder"], pde["pgauss"] = 2, 4
pde["torder"], pde["nstage"] = 1, 1
pde["dt"] = 1.0e-5 * np.ones(10000)
pde["visdt"], pde["saveSolFreq"] = pde["dt"][0], 100
pde["soltime"] = np.arange(100, pde["dt"].size + 1, 100, dtype=int)

gamma, mach, reynolds, prandtl = 1.4, 8.98, 0.71, 23319.0
tref, twall = 200.0, 500.0
pinf = 1.0 / (gamma * mach**2)
tinf = pinf / (gamma - 1.0)
pde["physicsparam"] = np.array([
    gamma, reynolds, prandtl, mach, 1.0, 1.0, 0.0,
    0.5 + pinf / (gamma - 1.0), tinf, tref, twall,
    1.5, 0.5, 0.0, pde["porder"], 0.001, 2.5,
])
pde["tau"] = np.array([4.0])
pde["GMRESrestart"], pde["linearsolveriter"] = 29, 30
pde["linearsolvertol"], pde["NLtol"], pde["NLiter"] = 1.0e-4, 1.0e-10, 3
pde["precMatrixType"], pde["ptcMatrixType"] = 2, 0
pde["matvectol"], pde["AV"] = 1.0e-6, 1

mesh = make_cylinder_mesh(pde["porder"])
distance = wall_distance(mesh, pde["porder"])
npe, _, ne = mesh["dgnodes"].shape
mesh["vdg"] = np.zeros((npe, 2, ne), order="F")
mesh["vdg"][:, 1, :] = curved_map_scale(mesh)
mesh["vdg"][:, 1, :] *= np.tanh(100.0 * distance[:, 0, :])
mesh["udg"] = initialize_solution(mesh, distance, pde["physicsparam"])

# Match the MATLAB script: generate and compile, but do not launch the 10,000-step run.
pde, mesh = exasim.preprocessing(pde, mesh)[0:2]
exasim.Gencode.gencode(pde)
exasim.Gencode.compilecode(pde)
print("Generated and compiled the Python Mach-8.98 cylinder application.")
