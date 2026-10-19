import numpy as np
import exasim
pde,mesh=exasim.initializeexasim(); pde["model"]="ModelD"; pde["modelfile"]="pdemodel"; pde["platform"]="cpu"; pde["mpiprocs"]=1; pde["hybrid"]=1; pde["nd"]=1; pde["Cxxpreprocessing"]=0
period=1800.0; pde["physicsparam"]=np.array([9.81,10000.0,3.0,-7.0,0.4,2*np.pi/period,1e-3]); pde["porder"]=1; pde["torder"]=2; pde["nstage"]=2; pde["tau"]=np.array([4.0])
pde["dt"]=np.ones(100); pde["saveSolFreq"]=1; pde["soltime"]=np.arange(1,101,dtype=int); pde["visdt"]=1.0
pde["GMRESrestart"],pde["linearsolveriter"],pde["linearsolvertol"],pde["NLtol"],pde["NLiter"]=30,30,1e-8,1e-8,3
mesh["p"],mesh["t"]=exasim.Mesh.linemesh(400); mesh["p"]*=10000.0
mesh["boundaryexpr"]=[lambda p:abs(p[0,:])<1e-8,lambda p:abs(p[0,:]-10000)<1e-8]; mesh["boundarycondition"]=np.array([1,2])
sol,pde,mesh=exasim.exasim(pde,mesh)[0:3]; print("Transformed-depth range:",float(sol[:,0,...].min()),float(sol[:,0,...].max()))
