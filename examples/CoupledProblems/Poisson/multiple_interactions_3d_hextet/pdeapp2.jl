# Domain 2: x in [0.25, 0.5].
pde[2],mesh[2]=Exasim.initializeexasim()

# All four domains use the same Poisson model.
pde[2].model="ModelD"
pde[2].modelfile="pdemodel"
pde[2].Cxxpreprocessing=1
pde[2].platform="cpu"
pde[2].mpiprocs=domainranks[2]
pde[2].hybrid=1

# Discretization, physical parameters, and solver controls.
pde[2].porder=porder
pde[2].pgauss=2*porder
pde[2].physicsparam=reshape([1.0,0.0],2,1) # diffusion; trace interface equation
pde[2].tau=[1.0]
pde[2].NLiter=12
pde[2].NLtol=1e-12
pde[2].linearsolvertol=1e-12
pde[2].matvectol=1e-12
pde[2].GMRESrestart=250
pde[2].linearsolveriter=600
pde[2].datapath=run_directory
pde[2].builddir=joinpath(run_directory,".exasim")
pde[2].buildpath=pde[2].builddir

# Adjacent domains use different tangential resolutions.
nt=isodd(2) ? N : 2*N
mesh[2].p,mesh[2].t=Exasim.Mesh.cubemesh(N,nt,nt,families[2])
mesh[2].p[1,:] .= 0.25 .+ 0.25 .* mesh[2].p[1,:]

# Boundary order: y=0, right, y=1, left, z=0, z=1.
mesh[2].boundaryexpr=[
    x->abs.(x[2,:]).<1e-8,
    x->abs.(x[1,:].-0.5).<1e-8,
    x->abs.(x[2,:].-1).<1e-8,
    x->abs.(x[1,:].-0.25).<1e-8,
    x->abs.(x[3,:]).<1e-8,
    x->abs.(x[3,:].-1).<1e-8,
]
mesh[2].boundarycondition=reshape([1,3,1,2,1,1],:,1)
mesh[2].interfacecondition=zeros(Int,0,1)
