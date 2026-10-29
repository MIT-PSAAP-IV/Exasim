"""Steady NACA0012: HDG in one wall layer, LDG in the outer flow."""
import os
from pathlib import Path
import numpy as np
import exasim

pde, mesh = exasim.initializeexasim()
pde.update(model="ModelD", modelfile="pdemodel", platform="cpu", mpiprocs=2,
           porder=4, pgauss=8, preconditioner=1, NLtol=1e-6, NLiter=20,
           linearsolvertol=1e-7, linearsolveriter=400, GMRESrestart=400,
           GMRESortho=0, matvectol=1e-3, ppdegree=0, RBdim=0)
# gamma, Reynolds, Prandtl, Mach, and freestream conservative state.
pde["physicsparam"] = np.array([1.4, 100., .72, .2, 1., 1., 0., 45.1429])
pde["tau"] = np.array([3.])
mesh["p"], connectivity = exasim.Mesh.readmesh(str(Path(__file__).with_name("grid.bin")), 0)
mesh["t"] = connectivity - 1  # binary file is 1-based; native Python is 0-based
mesh["boundaryexpr"] = ["sqrt((x-0.5)^2+y^2)<3", "abs(x)<20"]
mesh["boundarycondition"] = np.array([1, 2])  # wall, far field
mesh["curvedboundary"] = np.array([1, 0])
mesh["curvedboundaryexpr"] = [
    "y^2-(5*0.01*12*(0.29690*sqrt(abs(x))-0.12600*x-0.35160*x^2+0.28430*x^3-0.10150*x^4))^2",
    "0"]
selection = {"seed_boundaries": [1], "layers": 1}
if os.environ.get("HDGLDG_RUN_DIR"):
    pde["hdgldgpath"] = os.environ["HDGLDG_RUN_DIR"]
sol, pde, mesh = exasim.hdgldgsolve(pde, mesh, selection, nhdg=1)
print("NACA solution shape (nodes, u/q components, original elements):", sol.shape)
