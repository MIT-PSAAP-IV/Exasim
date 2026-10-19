import numpy as np
import exasim
pde,mesh=exasim.initializeexasim(); pde["model"]="ModelD"; pde["modelfile"]="pdemodel"; pde["platform"]="cpu"; pde["mpiprocs"]=1; pde["hybrid"]=1
pde["dae_alpha"],pde["dae_beta"]=0.0,0.0; pde["porder"],pde["pgauss"],pde["elemtype"]=3,6,1; pde["tau"]=np.array([100.0])
pde["linearsolvertol"],pde["ppdegree"],pde["RBdim"]=1e-6,10,0; pde["GMRESrestart"],pde["linearsolveriter"],pde["preconditioner"]=100,100,1
mesh["p"],mesh["t"]=exasim.Mesh.squaremesh(50,50,1,1)[0:2]; mesh["p"]=2*mesh["p"]-1
mesh["boundaryexpr"]=[lambda p:abs(p[1,:]+1)<1e-8,lambda p:abs(p[0,:]-1)<1e-8,lambda p:abs(p[1,:]-1)<1e-8,lambda p:abs(p[0,:]+1)<1e-8]
mesh["boundarycondition"]=np.ones(4,dtype=int)
a1,a2,radius=20.0,100.0,0.25; grid=np.linspace(-1,1,2001); xx,yy=np.meshgrid(grid,grid,indexing="ij")
theta=float(np.mean(1+a1/np.cosh(a2*(xx*xx+yy*yy-radius*radius))))
pde["physicsparam"]=np.array([0.025,1.0,theta,a1,a2,radius])
sol,pde,mesh=exasim.exasim(pde,mesh)[0:3]; print("Velocity range:",float(sol[:,0:2,...].min()),float(sol[:,0:2,...].max()))
