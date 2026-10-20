import numpy as np
import exasim
pde,mesh=exasim.initializeexasim(); pde["model"]="ModelD"; pde["modelfile"]="pdemodel"; pde["mpiprocs"]=1; pde["hybrid"]=1; pde["porder"]=3; pde["physicsparam"]=np.array([1,1]); pde["tau"]=np.array([1.0]); pde["linearsolvertol"]=1e-3; pde["GMRESrestart"]=100
pde["dae_alpha"], pde["dae_beta"] = 0.0, 0.0
mesh["p"],mesh["t"]=exasim.Mesh.SquareMesh(4,4,1)[0:2]; mesh["boundaryexpr"]=[lambda p:abs(p[1,:])<1e-8,lambda p:abs(p[0,:]-1)<1e-8,lambda p:abs(p[1,:]-1)<1e-8,lambda p:abs(p[0,:])<1e-8]; mesh["boundarycondition"]=np.array([2,1,2,2])
sol,pde,mesh=exasim.exasim(pde,mesh)[0:3]; print("Solution range:",float(sol[:,0,...].min()),float(sol[:,0,...].max()))
