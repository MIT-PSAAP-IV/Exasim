import numpy as np
import exasim
pde,mesh=exasim.initializeexasim(); pde["model"]="ModelD"; pde["modelfile"]="pdemodel"; pde["platform"]="cpu"; pde["mpiprocs"]=1; pde["hybrid"]=1
pde["porder"],pde["pgauss"]=3,6; pde["physicsparam"]=np.array([80.194,400889.806/100,0,0,250]); pde["tau"]=np.array([20000.0])
pde["linearsolvertol"],pde["ppdegree"],pde["RBdim"]=1e-6,10,0; pde["GMRESrestart"],pde["linearsolveriter"],pde["preconditioner"]=100,100,1
mesh["p"],mesh["t"]=exasim.Mesh.SquareMesh(32,32,1)[0:2]; mesh["p"][0,:]*=20; mesh["p"][1,:]*=10
mesh["boundaryexpr"]=[lambda p:abs(p[1,:])<1e-6,lambda p:abs(p[0,:]-20)<1e-6,lambda p:(abs(p[1,:]-10)<1e-6)&(p[0,:]>15-1e-3),lambda p:(abs(p[1,:]-10)<1e-6)&(p[0,:]>5-1e-3),lambda p:(abs(p[1,:]-10)<1e-6)&(p[0,:]>-1e-3),lambda p:abs(p[0,:])<1e-6]
mesh["boundarycondition"]=np.array([4,1,5,6,5,1]); sol,pde,mesh=exasim.exasim(pde,mesh)[0:3]; print("Deformed-coordinate range:",float(sol[:,0:2,...].min()),float(sol[:,0:2,...].max()))
