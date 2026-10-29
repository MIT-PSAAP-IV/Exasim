"""Unit-cube Poisson: HDG on x<=1/2, LDG on x>=1/2, one rank each."""
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
pde["physicsparam"] = np.array([1., 0.])  # conductivity, Dirichlet value
pde["tau"] = np.array([1.])
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

N = 4
mesh["p"], mesh["t"] = exasim.Mesh.cubemesh(N, N, N, 1)
mesh["boundaryexpr"] = ["y<1e-8", "x>1-1e-8", "y>1-1e-8",
                        "x<1e-8", "z<1e-8", "z>1-1e-8"]
mesh["boundarycondition"] = np.ones(6, dtype=int)
centers = mesh["p"][0, mesh["t"]].mean(axis=0)
labels = (centers <= .5).astype(int)
if os.environ.get("HDGLDG_RUN_DIR"):
    pde["hdgldgpath"] = os.environ["HDGLDG_RUN_DIR"]
sol, pde, mesh = exasim.hdgldgsolve(pde, mesh, labels, nhdg=1)
print("Poisson3D solution shape (nodes, u/q components, original elements):", sol.shape)
