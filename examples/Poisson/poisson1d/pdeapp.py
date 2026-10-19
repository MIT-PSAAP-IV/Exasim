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
pde["porder"] = 2
pde["pgauss"] = 2 * pde["porder"]
pde["physicsparam"] = np.array([1.0])
pde["tau"] = np.array([1.0])
pde["linearsolvertol"] = 1.0e-8
pde["ppdegree"] = 1
pde["RBdim"] = 0
pde["Cxxpreprocessing"] = 0

# Python's linemesh argument is the number of points; six points give five elements.
mesh["p"], mesh["t"] = exasim.Mesh.linemesh(6)
mesh["boundaryexpr"] = [
    lambda p: np.abs(p[0, :]) < 1.0e-8,
    lambda p: np.abs(p[0, :] - 1.0) < 1.0e-8,
]
mesh["boundarycondition"] = np.array([1, 1])

sol, pde, mesh = exasim.exasim(pde, mesh)[0:3]
print("Poisson 1D solution range:", float(sol[:, 0, :].min()), float(sol[:, 0, :].max()))
