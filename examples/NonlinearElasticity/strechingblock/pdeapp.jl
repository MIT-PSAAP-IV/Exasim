using Exasim
include("pdemodel.jl")
pde,mesh=Exasim.initializeexasim(); pde.model="ModelD"; pde.modelfile="pdemodel"; pde.platform="cpu"; pde.mpiprocs=1; pde.hybrid=1
young,poisson=1.25,0.25; shear=young/(2*(1+poisson)); lame=poisson*young/((1+poisson)*(1-2poisson))+2shear/3
pde.porder=3; pde.pgauss=6; pde.physicsparam=reshape([shear,lame,1.25,0.0],1,:); pde.tau=[60.0]
pde.linearsolvertol=1e-6; pde.ppdegree=10; pde.RBdim=0; pde.GMRESrestart=100; pde.linearsolveriter=100; pde.preconditioner=1
mesh.p,mesh.t=Exasim.Mesh.SquareMesh(16,16,1)
mesh.boundaryexpr=[p->abs.(p[2,:]).<1e-6,p->abs.(p[1,:].-1).<1e-6,p->abs.(p[2,:].-1).<1e-6,p->abs.(p[1,:]).<1e-6]
mesh.boundarycondition=reshape([1,3,1,2],:,1)
sol,pde,mesh=Exasim.exasim(pde,mesh)[1:3]; println("Deformed-coordinate range: ",extrema(sol[:,1:2,:]))
