import numpy as np

import exasim


pde, mesh = exasim.initializeexasim()
pde["model"] = "ModelD"
pde["modelfile"] = "pdemodel"
pde["platform"] = "cpu"
pde["mpiprocs"] = 1
pde["hybrid"] = 1
pde["debugmode"] = 0
pde["nd"] = 1
pde["Cxxpreprocessing"] = 0
pde["porder"] = 3
pde["pgauss"] = 2 * pde["porder"]
pde["torder"] = 1
pde["nstage"] = 1
pde["tau"] = np.array([1.0])

temperature = 1400.0
boltzmann = 8.617e-5
length = 5.0e-2
diffusivity = 9.28e-6 * np.exp(-1.46 / (boltzmann * temperature))
time_scale = 0.01 * length * length / diffusivity
pde["physicsparam"] = np.array([
    time_scale * diffusivity / (length * length),
    1.0e-5 * time_scale,
    1.0,
])
pde["ppdegree"] = 1
pde["RBdim"] = 0
pde["linearsolvertol"] = 1.0e-12

dt = 5.0e-3
pde["dt"] = dt * np.ones(200)
pde["soltime"] = np.arange(1, 201)
pde["visdt"] = dt
mesh["p"], mesh["t"] = exasim.Mesh.linemesh(301)
mesh["boundaryexpr"] = [
    lambda p: np.abs(p[0, :]) < 1.0e-8,
    lambda p: np.abs(p[0, :] - 1.0) < 1.0e-8,
]
mesh["boundarycondition"] = np.array([1, 2])
pde["gencode"] = 0

sol, pde, mesh = exasim.exasim(pde, mesh)[0:3]
print("Concentration range:", float(np.min(sol[:, 0, ...])), float(np.max(sol[:, 0, ...])))
