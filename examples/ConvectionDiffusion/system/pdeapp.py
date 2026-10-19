import numpy as np

import exasim


pde, mesh = exasim.initializeexasim()
pde["model"] = "ModelD"
pde["modelfile"] = "pdemodel"
pde["mpiprocs"] = 1
pde["hybrid"] = 1
pde["Cxxpreprocessing"] = 0
pde["porder"] = 3
pde["physicsparam"] = np.array([0.1])
pde["tau"] = np.array([1.0])
pde["GMRESrestart"] = 50
pde["ppdegree"] = 20
pde["linearsolvertol"] = 1.0e-6

mesh["p"], mesh["t"] = exasim.Mesh.SquareMesh(32, 32, 1)[0:2]
mesh["boundaryexpr"] = [
    lambda p: np.abs(p[1, :]) < 1.0e-8,
    lambda p: np.abs(p[0, :] - 1.0) < 1.0e-8,
    lambda p: np.abs(p[1, :] - 1.0) < 1.0e-8,
    lambda p: np.abs(p[0, :]) < 1.0e-8,
]
mesh["boundarycondition"] = np.ones(4, dtype=int)

sol, pde, mesh = exasim.exasim(pde, mesh)[0:3]
pde["visscalars"] = ["temperature1", 0, "temperature2", 1]
pde["visvectors"] = [
    "temperature gradient 1", np.array([2, 4], dtype=int),
    "temperature gradient 2", np.array([3, 5], dtype=int),
]
exasim.vis(sol, pde, mesh)
print("Done!")
