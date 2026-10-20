"""Python frontend for the periodic Taylor--Green vortex."""

import numpy as np
import exasim

pde, mesh = exasim.initializeexasim()
pde["model"], pde["modelfile"] = "ModelD", "pdemodel"
pde["platform"], pde["mpiprocs"], pde["hybrid"] = "cpu", 8, 1
pde["porder"], pde["torder"], pde["nstage"] = 2, 3, 3
pde["dt"] = 0.16 * np.ones(200)
pde["soltime"] = np.arange(1, 201, dtype=int)
pde["visdt"] = 0.05
pde["physicsparam"] = np.array([1.4, 1600.0, 0.71, 0.2])
pde["tau"] = np.array([2.0])
pde["GMRESrestart"], pde["linearsolveriter"] = 50, 51
pde["linearsolvertol"], pde["NLiter"] = 1.0e-6, 2
pde["preconditioner"], pde["precMatrixType"], pde["ppdegree"] = 1, 2, 0

mesh["p"], mesh["t"] = exasim.Mesh.cubemesh(24, 24, 24, 1)[0:2]
mesh["p"] *= 2.0 * np.pi
mesh["boundaryexpr"] = [
    lambda p: np.abs(p[1, :]) < 1.0e-8,
    lambda p: np.abs(p[0, :] - 2.0*np.pi) < 1.0e-8,
    lambda p: np.abs(p[1, :] - 2.0*np.pi) < 1.0e-8,
    lambda p: np.abs(p[0, :]) < 1.0e-8,
    lambda p: np.abs(p[2, :]) < 1.0e-8,
    lambda p: np.abs(p[2, :] - 2.0*np.pi) < 1.0e-8,
]
mesh["boundarycondition"] = np.ones(6, dtype=int)
mesh["periodicexpr"] = [
    [2, lambda p: p[[1, 2], :], 4, lambda p: p[[1, 2], :]],
    [1, lambda p: p[[0, 2], :], 3, lambda p: p[[0, 2], :]],
    [5, lambda p: p[[0, 1], :], 6, lambda p: p[[0, 1], :]],
]

sol, pde, mesh = exasim.exasim(pde, mesh)[0:3]
print("Taylor--Green vortex completed with solution shape", sol.shape)
