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

length = 5.0e-2
wall_temperature = 500.0
temperature_scale = 1000.0
gas_constant = 8.314
boltzmann = 8.617e-5
reaction_rate = 1.0e-5
chemistry_modulus = 1.415e10
reference_potential = -104.9e3
reference_concentration = 7.45e4
diffusivity = 9.28e-6
barrier = 1.46
heat_capacity = 170.0 * 13000.0
thermal_diffusivity = 15.0 / heat_capacity
time_scale = length * length / thermal_diffusivity
mu = np.array([
    length, wall_temperature, temperature_scale, gas_constant, boltzmann,
    reaction_rate, chemistry_modulus, reference_potential, reference_concentration,
    diffusivity, barrier, time_scale, time_scale * diffusivity / length**2,
    time_scale * thermal_diffusivity / length**2, heat_capacity, 1.75, np.sqrt(200.0),
])
pde["physicsparam"] = mu
pde["ppdegree"] = 1
pde["RBdim"] = 0
pde["linearsolvertol"] = 1.0e-12
pde["dt"] = 0.01 * np.ones(100)
pde["soltime"] = np.arange(1, 101)
pde["visdt"] = 0.01

mesh["p"], mesh["t"] = exasim.Mesh.linemesh(301)
mesh["boundaryexpr"] = [lambda p: np.abs(p[0, :]) < 1.0e-8, lambda p: np.abs(p[0, :] - 1.0) < 1.0e-8]
mesh["boundarycondition"] = np.array([1, 2])

sol, pde, mesh = exasim.exasim(pde, mesh)[0:3]
print("Temperature range:", float(np.min(sol[:, 0, ...])), float(np.max(sol[:, 0, ...])))
