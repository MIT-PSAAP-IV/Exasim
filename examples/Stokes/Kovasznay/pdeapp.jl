using Exasim, LinearAlgebra
include("pdemodel.jl")
pde,mesh=Exasim.initializeexasim(); pde.model="ModelD"; pde.modelfile="pdemodel"; pde.platform="cpu"; pde.mpiprocs=1; pde.hybrid=1
pde.dae_alpha,pde.dae_beta=0.0,0.0; pde.porder,pde.pgauss=3,6; pde.physicsparam=[0.1 1e4]; pde.tau=[2.0]
pde.linearsolvertol,pde.ppdegree,pde.RBdim=1e-6,10,0; pde.GMRESrestart,pde.linearsolveriter,pde.preconditioner=100,100,1
mesh.p,mesh.t=Exasim.Mesh.squaremesh(16,16,1,1)
mesh.boundaryexpr=[p->abs.(p[2,:]).<1e-6,p->abs.(p[1,:].-1).<1e-6,p->abs.(p[2,:].-1).<1e-6,p->abs.(p[1,:]).<1e-6]
mesh.boundarycondition=reshape(ones(Int,4),:,1)
sol,pde,mesh=Exasim.exasim(pde,mesh)[1:3]
println("Kovasznay velocity range: ",extrema(sol[:,1:2,:]))
