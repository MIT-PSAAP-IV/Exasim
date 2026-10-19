import numpy as np

import exasim
from plate_mesh import make_plate_mesh


pde, mesh = exasim.initializeexasim()
pde["model"] = "ModelD"
pde["modelfile"] = "pdemodel"
pde["platform"] = "cpu"
pde["mpiprocs"] = 1
pde["hybrid"] = 1
pde["debugmode"] = 0

youngs_modulus, poisson_ratio = 1.0, 0.30
shear_modulus = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
lame_first = poisson_ratio * youngs_modulus / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
pde["porder"] = 2
pde["pgauss"] = 2 * pde["porder"]
pde["physicsparam"] = np.array([shear_modulus, lame_first, 0.08, 0.0, 0.15])
pde["tau"] = np.array([2.0 * lame_first])
pde["linearsolvertol"] = 1.0e-8
pde["ppdegree"] = 12
pde["RBdim"] = 0
pde["GMRESrestart"] = 100
pde["linearsolveriter"] = 200
pde["preconditioner"] = 1

mesh["p"], mesh["t"] = make_plate_mesh()
mesh["boundaryexpr"] = [
    lambda p: np.abs(p[1, :]) < 1.0e-8,
    lambda p: np.abs(p[0, :] - 0.5) < 1.0e-8,
    lambda p: np.abs(p[1, :] - 0.2) < 1.0e-8,
    lambda p: np.abs(p[0, :] + 0.5) < 1.0e-8,
]
mesh["boundarycondition"] = np.array([2, 1, 1, 1])

sol, pde, mesh = exasim.exasim(pde, mesh)[0:3]
print("Displacement range:", float(np.min(sol[:, 0:2, ...])), float(np.max(sol[:, 0:2, ...])))
