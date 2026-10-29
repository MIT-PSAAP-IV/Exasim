"""Compressible Couette flow with viscous heating: one HDG rank and one LDG rank."""
import os
import numpy as np
import exasim

pde, mesh = exasim.initializeexasim()
pde["model"] = "ModelD"
pde["modelfile"] = "pdemodel"
pde["platform"] = "cpu"
pde["mpiprocs"] = 2
pde["porder"] = 2
pde["pgauss"] = 2 * pde["porder"]
pde["physicsparam"] = np.array([1.4,10,0.72,0.8,1,1,1.1])
pde["tau"] = np.array([3.])
pde["preconditioner"] = 1
pde["NLtol"] = 1e-10
pde["NLiter"] = 20
pde["linearsolvertol"] = 1e-8
pde["linearsolveriter"] = 2000
pde["GMRESrestart"] = 200
pde["GMRESortho"] = 1
pde["matvectol"] = 1e-3
pde["RBdim"] = 0
pde["ppdegree"] = 0

mesh["p"], mesh["t"] = exasim.Mesh.SquareMesh(4, 4, 1)
mesh["boundaryexpr"] = ["abs(y)<1e-12", "abs(x-1)<1e-12",
                        "abs(y-1)<1e-12", "abs(x)<1e-12"]
mesh["boundarycondition"] = np.array([1, 1, 1, 1])
centers = mesh["p"][0, mesh["t"]].mean(axis=0)
labels = (centers <= .5).astype(int)

# Optional working location; normal invocation needs no command-line arguments.
if os.environ.get("HDGLDG_RUN_DIR"):
    pde["hdgldgpath"] = os.environ["HDGLDG_RUN_DIR"]
sol, pde, mesh = exasim.hdgldgsolve(pde, mesh, labels, nhdg=1)
print("Navier-Stokes solution shape (nodes, u/q components, original elements):", sol.shape)
