import numpy as np
import exasim

pde, mesh = exasim.initializeexasim()
pde["model"], pde["modelfile"], pde["platform"] = "ModelD", "pdemodel", "cpu"
pde["mpiprocs"], pde["hybrid"] = 1, 1
pde["porder"], pde["pgauss"] = 3, 6
pde["physicsparam"] = np.array([80.194, 488.806, 0.0, 125.0/16.0])
pde["tau"] = np.array([1000.0])
pde["linearsolvertol"], pde["ppdegree"], pde["RBdim"] = 1.0e-6, 10, 0
pde["GMRESrestart"], pde["linearsolveriter"], pde["preconditioner"] = 100, 100, 1
mesh["p"], mesh["t"] = exasim.Mesh.SquareMesh(10, 10, 1)[0:2]
xi, yi = mesh["p"].copy()
mesh["p"][0, :], mesh["p"][1, :] = 48.0*xi, yi*(44.0+16.0*xi)
xmin, xmax = mesh["p"][0, :].min(), mesh["p"][0, :].max()
mesh["boundaryexpr"] = [
    lambda p: np.abs(p[0,:]-xmin)<1.0e-6, lambda p: np.abs(p[0,:]-xmax)<1.0e-6,
    lambda p: p[1,:]<44.0+1.0e-6, lambda p: p[1,:]>44.0-1.0e-6,
]
mesh["boundarycondition"] = np.array([2,3,1,1])
sol, pde, mesh = exasim.exasim(pde, mesh)[0:3]
print("Deformed-coordinate range:", float(sol[:,0:2,...].min()), float(sol[:,0:2,...].max()))
