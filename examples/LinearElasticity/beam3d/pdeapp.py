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
pde["physicsparam"] = np.array([1.0, 1.0, 0.0, 0.0, 0.01])
pde["tau"] = np.array([2.0])
pde["RBdim"] = 0
pde["linearsolvertol"] = 1.0e-6
pde["NLtol"] = 1.0e-8
pde["GMRESortho"] = 1
pde["GMRESrestart"] = 100
pde["linearsolveriter"] = 1000
pde["preconditioner"] = 1
pde["ppdegree"] = 0

mesh["p"], mesh["t"] = exasim.Mesh.cubemesh(20, 4, 4, 1)[0:2]
mesh["p"][0, :] *= 5.0
mesh["boundaryexpr"] = [
    lambda p: np.abs(p[1, :]) < 1.0e-6,
    lambda p: np.abs(p[0, :] - 5.0) < 1.0e-6,
    lambda p: np.abs(p[1, :] - 1.0) < 1.0e-6,
    lambda p: np.abs(p[0, :]) < 1.0e-6,
    lambda p: np.abs(p[2, :]) < 1.0e-6,
    lambda p: np.abs(p[2, :] - 1.0) < 1.0e-6,
]
mesh["boundarycondition"] = np.array([1, 1, 1, 2, 1, 3])

sol, pde, mesh = exasim.exasim(pde, mesh)[0:3]
pde["visscalars"] = []
pde["visvectors"] = ["displacement", np.array([0, 1, 2], dtype=int)]
exasim.vis(sol, pde, mesh)
print("Done!")
