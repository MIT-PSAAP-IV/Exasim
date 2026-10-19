import sys
from pathlib import Path
import numpy as np
import exasim

sys.path.insert(0,str(Path(__file__).parents[2]/"MeshAdaptivity"/"isoq2d_idealgas"))
from isoq_mesh import make_isoq_mesh
pde,mesh=exasim.initializeexasim(); pde["model"]="ModelD"; pde["modelfile"]="pdemodel"; pde["platform"]="cpu"; pde["mpiprocs"]=1; pde["hybrid"]=1; pde["nd"]=2
pde["porder"],pde["pgauss"]=3,6; pde["physicsparam"]=np.array([1,0,1,1]); pde["tau"]=np.array([1.0]); pde["linearsolvertol"]=1e-8; pde["ppdegree"]=10; pde["GMRESrestart"]=50; pde["RBdim"]=0
mesh=make_isoq_mesh(pde["porder"]); mesh["boundarycondition"]=np.array([1,4,3,2])
sol,pde,mesh=exasim.exasim(pde,mesh)[0:3]; print("Potential range:",float(sol[:,0,...].min()),float(sol[:,0,...].max()))
