from pathlib import Path
import numpy as np
import exasim
from mesh_reader import read_msh3
pde,mesh=exasim.initializeexasim(); pde["model"]="ModelD"; pde["modelfile"]="pdemodel"; pde["platform"]="cpu"; pde["mpiprocs"]=8; pde["hybrid"]=1; pde["GMRESortho"]=0; pde["ppdegree"]=20
pde["porder"],pde["pgauss"]=2,4; pde["tau"]=np.array([1.0]); pde["linearsolvertol"]=1e-6; pde["NLiter"]=1; pde["RBdim"]=0; pde["linearsolveriter"]=50; pde["GMRESrestart"]=50
pde["physicsparam"]=np.array([1,1e-4,0.05,3e6,1.80955e-8,18.75e3,5e18,1e-2,4e-4])
mesh["p"],mesh["t"]=read_msh3(Path(__file__).with_name("streamer_16k_fixed.msh")); mesh["p"]/=mesh["p"][0,:].max()/125.0
mesh["boundaryexpr"]=[lambda p:abs(p[1,:])<1e-8,lambda p:abs(p[0,:]-125)<1e-8,lambda p:abs(p[1,:]-125)<1e-8,lambda p:abs(p[0,:])<1e-8]; mesh["boundarycondition"]=np.array([1,2,3,4])
sol,pde,mesh=exasim.exasim(pde,mesh)[0:3]; print("Potential range:",float(sol[:,0,...].min()),float(sol[:,0,...].max()))
