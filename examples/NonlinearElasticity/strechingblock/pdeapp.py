import numpy as np
import exasim

pde, mesh = exasim.initializeexasim()
pde["model"], pde["modelfile"], pde["platform"] = "ModelD", "pdemodel", "cpu"
pde["mpiprocs"], pde["hybrid"] = 1, 1
young, poisson = 1.25, 0.25
shear = young/(2.0*(1.0+poisson))
lame = poisson*young/((1.0+poisson)*(1.0-2.0*poisson)) + 2.0*shear/3.0
pde["porder"], pde["pgauss"] = 3, 6
pde["physicsparam"], pde["tau"] = np.array([shear,lame,1.25,0.0]), np.array([60.0])
pde["linearsolvertol"], pde["ppdegree"], pde["RBdim"] = 1e-6, 10, 0
pde["GMRESrestart"], pde["linearsolveriter"], pde["preconditioner"] = 100,100,1
mesh["p"], mesh["t"] = exasim.Mesh.SquareMesh(16,16,1)[0:2]
mesh["boundaryexpr"] = [lambda p: abs(p[1,:])<1e-6, lambda p: abs(p[0,:]-1)<1e-6,
    lambda p: abs(p[1,:]-1)<1e-6, lambda p: abs(p[0,:])<1e-6]
mesh["boundarycondition"] = np.array([1,3,1,2])
sol,pde,mesh = exasim.exasim(pde,mesh)[0:3]
print("Deformed-coordinate range:",float(sol[:,0:2,...].min()),float(sol[:,0:2,...].max()))
