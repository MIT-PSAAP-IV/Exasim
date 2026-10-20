import numpy as np

import exasim


pde, mesh = exasim.initializeexasim()
pde["model"] = "ModelD"
pde["modelfile"] = "pdemodel"
pde["platform"] = "cpu"
pde["mpiprocs"] = 1
pde["hybrid"] = 1
pde["Cxxpreprocessing"] = 0
pde["porder"] = 3
pde["pgauss"] = 2 * pde["porder"]
pde["torder"] = 1
pde["nstage"] = 1
pde["tau"] = np.array([1.0])

length = 5.0e-2
wall_temperature = 500.0
temperature_scale = 1000.0
heat_capacity = 170.0 * 13000.0
thermal_diffusivity = 15.0 / heat_capacity
time_scale = length**2 / thermal_diffusivity
pde["physicsparam"] = np.array([
    length, wall_temperature, temperature_scale, 8.314, 8.617e-5, 1.0e-5,
    1.415e10, -104.9e3, 7.45e4, 9.28e-6, 1.46, time_scale,
    time_scale * 9.28e-6 / length**2,
    time_scale * thermal_diffusivity / length**2, heat_capacity, 1.5,
])
pde["ppdegree"] = 10
pde["RBdim"] = 0
pde["linearsolvertol"] = 1.0e-8
pde["dt"] = 0.001 * np.arange(1, 201, dtype=float) ** 2
pde["soltime"] = np.arange(1, 201)

mesh["p"], mesh["t"] = exasim.Mesh.linemesh(301)
mesh["boundaryexpr"] = [lambda p: np.abs(p[0, :]) < 1.0e-8, lambda p: np.abs(p[0, :] - 1.0) < 1.0e-8]
mesh["boundarycondition"] = np.array([1, 2])

sol, pde, mesh = exasim.exasim(pde, mesh)[0:3]
print("Temperature range:", float(np.min(sol[:, 0, ...])), float(np.max(sol[:, 0, ...])))
print("Concentration range:", float(np.min(sol[:, 1, ...])), float(np.max(sol[:, 1, ...])))
