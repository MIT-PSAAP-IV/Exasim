import numpy as np

import exasim


pde, mesh = exasim.initializeexasim()
pde["model"] = "ModelC"
pde["modelfile"] = "pdemodel"
pde["mpiprocs"] = 16
pde["hybrid"] = 0
pde["porder"] = 2
pde["torder"] = 3
pde["nstage"] = 3
final_time = np.sqrt(125.0)
pde["dt"] = 0.05 * np.ones(int(np.ceil(final_time / 0.05)))
pde["soltime"] = np.array([pde["dt"].size], dtype=int)
pde["visdt"] = 1.0
pde["linearsolveriter"] = 50
pde["GMRESrestart"] = 50
pde["NLiter"] = 2
pde["physicsparam"] = np.array([1.4, 0.5, 0.3, 1.5, 5.0, 5.0])
pde["tau"] = np.array([3.0])
pde["ALE"] = 1

mesh["p"], mesh["t"] = exasim.Mesh.SquareMesh(64, 48, 1)[0:2]
mesh["p"][0, :] *= 20.0
mesh["p"][1, :] *= 15.0
mesh["boundaryexpr"] = [
    lambda p: np.abs(p[1, :]) < 1.0e-8,
    lambda p: np.abs(p[0, :] - 20.0) < 1.0e-8,
    lambda p: np.abs(p[1, :] - 15.0) < 1.0e-8,
    lambda p: np.abs(p[0, :]) < 1.0e-8,
]
mesh["boundarycondition"] = np.ones(4, dtype=int)
mesh["periodicexpr"] = [
    [2, lambda p: p[1, :], 4, lambda p: p[1, :]],
    [1, lambda p: p[0, :], 3, lambda p: p[0, :]],
]

# The MATLAB case prepares this 16-rank run without launching it.
pde, mesh = exasim.preprocessing(pde, mesh)[0:2]
exasim.Gencode.gencode(pde)
print("Generated the mapped Euler-vortex input and source files.")
