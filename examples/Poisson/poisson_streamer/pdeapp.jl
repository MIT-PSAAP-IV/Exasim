using Exasim
include("pdemodel.jl"); include("mesh_reader.jl")
pde,mesh=Exasim.initializeexasim(); pde.model="ModelD"; pde.modelfile="pdemodel"; pde.platform="cpu"; pde.mpiprocs=8; pde.hybrid=1; pde.GMRESortho=0; pde.ppdegree=20; pde.porder=2; pde.pgauss=4; pde.tau=[1.0]; pde.linearsolvertol=1e-6; pde.NLiter=1; pde.RBdim=0; pde.linearsolveriter=50; pde.GMRESrestart=50
pde.physicsparam=reshape([1,1e-4,0.05,3e6,1.80955e-8,18.75e3,5e18,1e-2,4e-4],1,:)
mesh.p,mesh.t=read_msh3(joinpath(@__DIR__,"streamer_16k_fixed.msh")); mesh.p./=maximum(mesh.p[1,:])/125
mesh.boundaryexpr=[p->abs.(p[2,:]).<1e-8,p->abs.(p[1,:].-125).<1e-8,p->abs.(p[2,:].-125).<1e-8,p->abs.(p[1,:]).<1e-8]; mesh.boundarycondition=reshape([1,2,3,4],:,1)
sol,pde,mesh=Exasim.exasim(pde,mesh)[1:3]; println("Potential range: ",extrema(sol[:,1,:]))
