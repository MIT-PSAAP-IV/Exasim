from pathlib import Path

import numpy as np

import exasim


pde, mesh = exasim.initializeexasim()
pde["model"] = "ModelD"
pde["modelfile"] = "pdemodel"
pde["platform"] = "cpu"
pde["mpiprocs"] = 1
pde["hybrid"] = 1
pde["debugmode"] = 0
pde["porder"] = 1
pde["pgauss"] = 2
pde["physicsparam"] = np.array([1.25, 0.833, 0.0, 0.1])
pde["tau"] = np.array([2.0])
pde["linearsolvertol"] = 1.0e-6
pde["ppdegree"] = 1
pde["RBdim"] = 0
pde["GMRESrestart"] = 1000
pde["linearsolveriter"] = 1000
pde["preconditioner"] = 1
pde["NLiter"] = 2

geometry = str(Path(__file__).with_name("seawall"))
mesh["p"], mesh["t"] = exasim.Mesh.gmshcall(pde, geometry, 2, 1)[0:2]
mesh["t"] = mesh["t"][::-1, :]
ymin, ymax = np.min(mesh["p"][1, :]), np.max(mesh["p"][1, :])
mesh["boundaryexpr"] = [
    lambda p: np.abs(p[1, :] - ymin) < 1.0e-3,
    lambda p: np.abs(p[1, :] - ymax) < 1.0e-3,
    lambda p: p[0, :] < 1.0e-3,
    lambda p: np.abs(p[0, :] - 0.51816) < 1.0e-3,
    lambda p: np.abs(p[1, :]) < 100.0,
]
mesh["boundarycondition"] = np.array([2, 1, 3, 1, 2])

sol, pde, mesh = exasim.exasim(pde, mesh)[0:3]
print("Displacement range:", float(np.min(sol[:, 0:2, ...])), float(np.max(sol[:, 0:2, ...])))
