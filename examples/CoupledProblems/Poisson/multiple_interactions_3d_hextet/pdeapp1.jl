# Domain 1: x in [0, 0.25].
pde[1],mesh[1]=Exasim.initializeexasim()

# All four domains use the same Poisson model.
pde[1].model="ModelD"
pde[1].modelfile="pdemodel"
pde[1].Cxxpreprocessing=1
pde[1].platform="cpu"
pde[1].mpiprocs=domainranks[1]
pde[1].hybrid=1

# Discretization, physical parameters, and solver controls.
pde[1].porder=porder
pde[1].pgauss=2*porder
pde[1].physicsparam=reshape([1.0,1.0],2,1) # diffusion; numerical-flux interface equation
pde[1].tau=[1.0]
pde[1].NLiter=12
pde[1].NLtol=1e-12
pde[1].linearsolvertol=1e-12
pde[1].matvectol=1e-12
pde[1].GMRESrestart=250
pde[1].linearsolveriter=600
pde[1].datapath=run_directory
pde[1].builddir=joinpath(run_directory,".exasim")
pde[1].buildpath=pde[1].builddir

# Adjacent domains use different tangential resolutions.
nt=isodd(1) ? N : 2*N
mesh[1].p,mesh[1].t=Exasim.Mesh.cubemesh(N,nt,nt,families[1])
mesh[1].p[1,:] .= 0 .+ 0.25 .* mesh[1].p[1,:]

# Boundary order: y=0, right, y=1, left, z=0, z=1.
mesh[1].boundaryexpr=[
    x->abs.(x[2,:]).<1e-8,
    x->abs.(x[1,:].-0.25).<1e-8,
    x->abs.(x[2,:].-1).<1e-8,
    x->abs.(x[1,:].-0).<1e-8,
    x->abs.(x[3,:]).<1e-8,
    x->abs.(x[3,:].-1).<1e-8,
]
mesh[1].boundarycondition=reshape([1,3,1,1,1,1],:,1)
mesh[1].interfacecondition=zeros(Int,0,1)
