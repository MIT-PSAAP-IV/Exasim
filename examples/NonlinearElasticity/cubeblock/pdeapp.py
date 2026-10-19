import numpy as np
import exasim
pde,mesh=exasim.initializeexasim(); pde["model"]="ModelD"; pde["modelfile"]="pdemodel"; pde["platform"]="cpu"; pde["mpiprocs"]=2; pde["hybrid"]=1
pde["porder"],pde["pgauss"]=3,6; pde["physicsparam"]=np.array([1,1,3,0,0]); pde["tau"]=np.array([1e3]); pde["RBdim"]=0
pde["linearsolvertol"],pde["NLtol"],pde["NLiter"]=1e-6,1e-8,10; pde["GMRESortho"]=1; pde["GMRESrestart"]=50; pde["linearsolveriter"]=1000; pde["preconditioner"]=1; pde["ppdegree"]=10
mesh["p"],mesh["t"]=exasim.Mesh.cubemesh(16,4,4,1)[0:2]
mesh["boundaryexpr"]=[lambda p:abs(p[1,:])<1e-6,lambda p:abs(p[0,:]-1)<1e-6,lambda p:abs(p[1,:]-1)<1e-6,lambda p:abs(p[0,:])<1e-6,lambda p:abs(p[2,:])<1e-6,lambda p:abs(p[2,:]-1)<1e-6]
mesh["boundarycondition"]=np.array([1,3,1,2,1,1]); sol,pde,mesh=exasim.exasim(pde,mesh)[0:3]; print("Deformed-coordinate range:",float(sol[:,0:3,...].min()),float(sol[:,0:3,...].max()))
