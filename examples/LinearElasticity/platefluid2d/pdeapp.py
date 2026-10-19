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
youngs_modulus, poisson_ratio, amplitude = 1.0, 0.30, 0.10
shear_modulus = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
lame_first = poisson_ratio * youngs_modulus / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
pde["porder"] = 2
pde["pgauss"] = 2 * pde["porder"]
pde["physicsparam"] = np.array([shear_modulus, lame_first, amplitude])
pde["tau"] = np.array([2.0 * shear_modulus])
pde["linearsolvertol"] = 1.0e-8
pde["ppdegree"] = 4
pde["RBdim"] = 0
pde["GMRESrestart"] = 100
pde["linearsolveriter"] = 200
pde["preconditioner"] = 1

geometry = str(Path(__file__).with_name("platefluid"))
mesh["p"], mesh["t"] = exasim.Mesh.gmshcall(pde, geometry, 2, 0)[0:2]
mesh["boundaryexpr"] = [
    lambda p: np.abs(p[1, :] + 4.975) < 1.0e-6,
    lambda p: np.abs(p[0, :] - 5.5) < 1.0e-6,
    lambda p: np.abs(p[1, :] - 5.025) < 1.0e-6,
    lambda p: np.abs(p[0, :] + 4.5) < 1.0e-6,
    lambda p: (p[0, :] >= -1.0e-6) & (p[0, :] <= 1.0 + 1.0e-6) & (p[1, :] >= -1.0e-6) & (p[1, :] <= 0.05 + 1.0e-6),
]
mesh["boundarycondition"] = np.array([1, 1, 1, 1, 2])

sol, pde, mesh = exasim.exasim(pde, mesh)[0:3]
print("Displacement range:", float(np.min(sol[:, 0:2, ...])), float(np.max(sol[:, 0:2, ...])))
