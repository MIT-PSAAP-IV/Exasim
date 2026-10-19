using Exasim
include("pdemodel.jl")
pde,mesh=Exasim.initializeexasim(); pde.model="ModelD"; pde.modelfile="pdemodel"; pde.platform="cpu"; pde.mpiprocs=1; pde.hybrid=1; pde.porder=3; pde.pgauss=6
pde.physicsparam=reshape([80.194,400889.806/100,0,0,250],1,:); pde.tau=[20000.0]; pde.linearsolvertol=1e-6; pde.ppdegree=10; pde.RBdim=0; pde.GMRESrestart=100; pde.linearsolveriter=100; pde.preconditioner=1
mesh.p,mesh.t=Exasim.Mesh.SquareMesh(32,32,1); mesh.p[1,:].*=20; mesh.p[2,:].*=10
mesh.boundaryexpr=[p->abs.(p[2,:]).<1e-6,p->abs.(p[1,:].-20).<1e-6,p->(abs.(p[2,:].-10).<1e-6).&(p[1,:].>15-1e-3),p->(abs.(p[2,:].-10).<1e-6).&(p[1,:].>5-1e-3),p->(abs.(p[2,:].-10).<1e-6).&(p[1,:].>-1e-3),p->abs.(p[1,:]).<1e-6]
mesh.boundarycondition=reshape([4,1,5,6,5,1],:,1); sol,pde,mesh=Exasim.exasim(pde,mesh)[1:3]; println("Deformed-coordinate range: ",extrema(sol[:,1:2,:]))
