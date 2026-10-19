using Exasim
include("pdemodel.jl")
pde,mesh=Exasim.initializeexasim(); pde.model="ModelD"; pde.modelfile="pdemodel"; pde.platform="cpu"; pde.mpiprocs=2; pde.hybrid=1
pde.porder=3; pde.pgauss=6; pde.physicsparam=reshape([1,1,3,0,0],1,:); pde.tau=[1e3]; pde.RBdim=0; pde.linearsolvertol=1e-6; pde.NLtol=1e-8; pde.NLiter=10; pde.GMRESortho=1; pde.GMRESrestart=50; pde.linearsolveriter=1000; pde.preconditioner=1; pde.ppdegree=10
mesh.p,mesh.t=Exasim.Mesh.cubemesh(16,4,4,1)
mesh.boundaryexpr=[p->abs.(p[2,:]).<1e-6,p->abs.(p[1,:].-1).<1e-6,p->abs.(p[2,:].-1).<1e-6,p->abs.(p[1,:]).<1e-6,p->abs.(p[3,:]).<1e-6,p->abs.(p[3,:].-1).<1e-6]
mesh.boundarycondition=reshape([1,3,1,2,1,1],:,1); sol,pde,mesh=Exasim.exasim(pde,mesh)[1:3]; println("Deformed-coordinate range: ",extrema(sol[:,1:3,:]))
