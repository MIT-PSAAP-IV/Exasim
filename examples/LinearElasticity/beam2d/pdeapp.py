import numpy as np

import exasim


pde, mesh = exasim.initializeexasim()
pde["model"] = "ModelD"
pde["modelfile"] = "pdemodel"
pde["platform"] = "cpu"
pde["mpiprocs"] = 1
pde["hybrid"] = 1
pde["debugmode"] = 0
pde["porder"] = 3
pde["pgauss"] = 2 * pde["porder"]
pde["physicsparam"] = np.array([1.0, 1.0, 0.0, 0.002])
pde["tau"] = np.array([2.0])
pde["linearsolvertol"] = 1.0e-6
pde["ppdegree"] = 10
pde["RBdim"] = 0
pde["GMRESrestart"] = 100
pde["linearsolveriter"] = 100
pde["preconditioner"] = 1

mesh["p"], mesh["t"] = exasim.Mesh.SquareMesh(64, 8, 1)[0:2]
mesh["p"][0, :] *= 8.0
mesh["boundaryexpr"] = [
    lambda p: np.abs(p[1, :]) < 1.0e-6,
    lambda p: np.abs(p[0, :] - 8.0) < 1.0e-6,
    lambda p: np.abs(p[1, :] - 1.0) < 1.0e-6,
    lambda p: np.abs(p[0, :]) < 1.0e-6,
]
mesh["boundarycondition"] = np.array([1, 1, 3, 2])

sol, pde, mesh = exasim.exasim(pde, mesh)[0:3]
pde["visscalars"] = []
pde["visvectors"] = ["displacement", np.array([0, 1], dtype=int)]
exasim.vis(sol, pde, mesh)
print("Done!")
