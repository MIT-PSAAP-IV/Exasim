import numpy as np

import exasim


pde, mesh = exasim.initializeexasim()
pde["model"] = "ModelD"
pde["modelfile"] = "pdemodel"
pde["platform"] = "cpu"
pde["mpiprocs"] = 1
pde["hybrid"] = 1
pde["debugmode"] = 0

youngs_modulus = 240.565
poisson_ratio = 0.495
shear_modulus = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
lame_first = poisson_ratio * youngs_modulus / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
pde["porder"] = 3
pde["pgauss"] = 2 * pde["porder"]
pde["physicsparam"] = np.array([shear_modulus, lame_first, 0.0, 16.0])
pde["tau"] = np.array([2.0 * shear_modulus])
pde["linearsolvertol"] = 1.0e-6
pde["ppdegree"] = 10
pde["RBdim"] = 0
pde["GMRESrestart"] = 100
pde["linearsolveriter"] = 100
pde["preconditioner"] = 1

mesh["p"], mesh["t"] = exasim.Mesh.SquareMesh(10, 10, 1)[0:2]
xi = mesh["p"][0, :].copy()
eta = mesh["p"][1, :].copy()
mesh["p"][0, :] = 48.0 * xi
mesh["p"][1, :] = eta * (44.0 + 16.0 * xi)
xmin, xmax = mesh["p"][0, :].min(), mesh["p"][0, :].max()
mesh["boundaryexpr"] = [
    lambda p: np.abs(p[0, :] - xmin) < 1.0e-6,
    lambda p: np.abs(p[0, :] - xmax) < 1.0e-6,
    lambda p: p[1, :] < 44.0 + 1.0e-6,
    lambda p: p[1, :] > 44.0 - 1.0e-6,
]
mesh["boundarycondition"] = np.array([2, 3, 1, 1])

sol, pde, mesh = exasim.exasim(pde, mesh)[0:3]
print("Displacement range:", float(np.min(sol[:, 0:2, ...])), float(np.max(sol[:, 0:2, ...])))
