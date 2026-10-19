import numpy as np
import exasim
pde,mesh=exasim.initializeexasim(); pde["model"]="ModelD"; pde["modelfile"]="pdemodel"; pde["platform"]="cpu"; pde["mpiprocs"]=1; pde["hybrid"]=1
pde["dae_alpha"],pde["dae_beta"]=0.0,0.0; pde["porder"],pde["pgauss"]=3,6; pde["physicsparam"]=np.array([0.1,1e4]); pde["tau"]=np.array([2.0])
pde["linearsolvertol"],pde["ppdegree"],pde["RBdim"]=1e-6,10,0; pde["GMRESrestart"],pde["linearsolveriter"],pde["preconditioner"]=100,100,1
mesh["p"],mesh["t"]=exasim.Mesh.squaremesh(16,16,1,1)[0:2]
mesh["boundaryexpr"]=[lambda p:abs(p[1,:])<1e-6,lambda p:abs(p[0,:]-1)<1e-6,lambda p:abs(p[1,:]-1)<1e-6,lambda p:abs(p[0,:])<1e-6]
mesh["boundarycondition"]=np.ones(4,dtype=int)
sol,pde,mesh=exasim.exasim(pde,mesh)[0:3]
print("Kovasznay velocity range:",float(sol[:,0:2,...].min()),float(sol[:,0:2,...].max()))
