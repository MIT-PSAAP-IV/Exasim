using Exasim
include("pdemodel.jl"); include(joinpath(dirname(@__DIR__),"..","MeshAdaptivity","sharpb2_idealgas","sharpb2_mesh.jl"))
pde,mesh=Exasim.initializeexasim(); pde.model="ModelD"; pde.modelfile="pdemodel"; pde.platform="cpu"; pde.mpiprocs=1; pde.hybrid=1; pde.nd=2
pde.porder=3; pde.pgauss=6; pde.physicsparam=reshape([1,0,1,1],1,:); pde.tau=[1.0]; pde.linearsolvertol=1e-8; pde.ppdegree=10; pde.GMRESrestart=50; pde.RBdim=0
mesh=make_sharpb2_mesh(pde.porder;radial_shift=0.0); mesh.boundarycondition=reshape([1,3,2,4],:,1)
sol,pde,mesh=Exasim.exasim(pde,mesh)[1:3]; println("Potential range: ",extrema(sol[:,1,:]))
