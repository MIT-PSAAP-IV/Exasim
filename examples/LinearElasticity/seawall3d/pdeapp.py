import numpy as np

import exasim
from seawall_mesh import make_seawall_mesh


pde, mesh = exasim.initializeexasim()
pde["model"] = "ModelD"
pde["modelfile"] = "pdemodel"
pde["platform"] = "cpu"
pde["mpiprocs"] = 8
pde["hybrid"] = 1
pde["debugmode"] = 0
pde["porder"] = 2
pde["pgauss"] = 4
pde["physicsparam"] = np.array([1.25, 0.833, 0.015, 0.0, -0.015])
pde["tau"] = np.array([5.0])
pde["RBdim"] = 0
pde["linearsolvertol"] = 1.0e-6
pde["NLtol"] = 1.0e-6
pde["GMRESortho"] = 1
pde["GMRESrestart"] = 100
pde["linearsolveriter"] = 200
pde["preconditioner"] = 1
pde["ppdegree"] = 0

mesh["p"], mesh["t"], mesh["dgnodes"] = make_seawall_mesh(pde)
ymin, ymax = np.min(mesh["p"][1, :]), np.max(mesh["p"][1, :])
zmin, zmax = np.min(mesh["p"][2, :]), np.max(mesh["p"][2, :])
mesh["boundaryexpr"] = [
    lambda p: np.abs(p[2, :] - zmin) < 1.0e-3,
    lambda p: np.abs(p[2, :] - zmax) < 1.0e-3,
    lambda p: np.abs(p[1, :] - ymin) < 1.0e-3,
    lambda p: np.abs(p[1, :] - ymax) < 1.0e-3,
    lambda p: p[0, :] < 1.0e-3,
    lambda p: np.abs(p[0, :] - 0.34544) < 1.0e-3,
    lambda p: np.abs(p[1, :] - 1.9812) < 1.0e-3,
    lambda p: np.abs(p[1, :]) < 100.0,
]
mesh["boundarycondition"] = np.array([2, 2, 2, 1, 3, 1, 1, 2])

sol, pde, mesh = exasim.exasim(pde, mesh)[0:3]
print("Displacement range:", float(np.min(sol[:, 0:3, ...])), float(np.max(sol[:, 0:3, ...])))
