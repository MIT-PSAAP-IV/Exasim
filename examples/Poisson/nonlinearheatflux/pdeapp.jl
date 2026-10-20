using Exasim
include("pdemodel.jl"); pde,mesh=Exasim.initializeexasim(); pde.model="ModelD"; pde.modelfile="pdemodel"; pde.mpiprocs=1; pde.hybrid=1; pde.extendedW=1; pde.dae_alpha=0.0; pde.dae_beta=0.0; pde.porder=3; pde.physicsparam=reshape([1,1],1,:); pde.tau=[1.0]; pde.GMRESrestart=100
mesh.p,mesh.t=Exasim.Mesh.SquareMesh(8,8,1); mesh.boundaryexpr=[p->abs.(p[2,:]).<1e-8,p->abs.(p[1,:].-1).<1e-8,p->abs.(p[2,:].-1).<1e-8,p->abs.(p[1,:]).<1e-8]; mesh.boundarycondition=reshape([2,1,2,2],:,1)
sol,pde,mesh=Exasim.exasim(pde,mesh)[1:3]; println("Solution range: ",extrema(sol[:,1,:]))
