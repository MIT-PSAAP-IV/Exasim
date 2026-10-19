using Exasim
include("pdemodel.jl")
pde,mesh=Exasim.initializeexasim(); pde.model="ModelD"; pde.modelfile="pdemodel"; pde.platform="cpu"; pde.mpiprocs=1; pde.hybrid=1; pde.nd=1; pde.Cxxpreprocessing=0
period=1800.0; pde.physicsparam=[9.81 10000.0 3.0 -7.0 0.4 2*pi/period 1e-3]; pde.porder=1; pde.torder=2; pde.nstage=2; pde.tau=[4.0]
pde.dt=ones(100); pde.saveSolFreq=1; pde.soltime=collect(1:100); pde.visdt=1.0
pde.GMRESrestart,pde.linearsolveriter,pde.linearsolvertol,pde.NLtol,pde.NLiter=30,30,1e-8,1e-8,3
mesh.p=reshape(collect(range(0.0,10000.0,length=401)),1,:); mesh.t=[collect(1:400)';collect(2:401)']
mesh.boundaryexpr=[p->abs.(p[1,:]).<1e-8,p->abs.(p[1,:].-10000).<1e-8]; mesh.boundarycondition=reshape([1,2],:,1)
sol,pde,mesh=Exasim.exasim(pde,mesh)[1:3]; println("Transformed-depth range: ",extrema(sol[:,1,:]))
