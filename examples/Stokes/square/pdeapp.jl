using Exasim, LinearAlgebra, Statistics
include("pdemodel.jl")
pde,mesh=Exasim.initializeexasim(); pde.model="ModelD"; pde.modelfile="pdemodel"; pde.platform="cpu"; pde.mpiprocs=1; pde.hybrid=1
pde.dae_alpha,pde.dae_beta=0.0,0.0; pde.porder,pde.pgauss,pde.elemtype=3,6,1; pde.tau=[100.0]
pde.linearsolvertol,pde.ppdegree,pde.RBdim=1e-6,10,0; pde.GMRESrestart,pde.linearsolveriter,pde.preconditioner=100,100,1
mesh.p,mesh.t=Exasim.Mesh.squaremesh(50,50,1,1); mesh.p .= 2 .* mesh.p .- 1
mesh.boundaryexpr=[p->abs.(p[2,:].+1).<1e-8,p->abs.(p[1,:].-1).<1e-8,p->abs.(p[2,:].-1).<1e-8,p->abs.(p[1,:].+1).<1e-8]
mesh.boundarycondition=reshape(ones(Int,4),:,1)
a1,a2,radius=20.0,100.0,0.25; grid=range(-1,1,length=2001); theta=mean(1+a1*sech(a2*(x*x+y*y-radius^2)) for x in grid,y in grid)
pde.physicsparam=[0.025 1.0 theta a1 a2 radius]
sol,pde,mesh=Exasim.exasim(pde,mesh)[1:3]; println("Velocity range: ",extrema(sol[:,1:2,:]))
