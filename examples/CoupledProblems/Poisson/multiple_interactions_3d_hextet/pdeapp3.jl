# Domain 3: x in [0.5, 0.75].
pde[3],mesh[3]=Exasim.initializeexasim()

# All four domains use the same Poisson model.
pde[3].model="ModelD"
pde[3].modelfile="pdemodel"
pde[3].Cxxpreprocessing=1
pde[3].platform="cpu"
pde[3].mpiprocs=domainranks[3]
pde[3].hybrid=1

# Discretization, physical parameters, and solver controls.
pde[3].porder=porder
pde[3].pgauss=2*porder
pde[3].physicsparam=reshape([1.0,1.0],2,1) # diffusion; numerical-flux interface equation
pde[3].tau=[1.0]
pde[3].NLiter=12
pde[3].NLtol=1e-12
pde[3].linearsolvertol=1e-12
pde[3].matvectol=1e-12
pde[3].GMRESrestart=250
pde[3].linearsolveriter=600
pde[3].datapath=run_directory
pde[3].builddir=joinpath(run_directory,".exasim")
pde[3].buildpath=pde[3].builddir

# Adjacent domains use different tangential resolutions.
nt=isodd(3) ? N : 2*N
mesh[3].p,mesh[3].t=Exasim.Mesh.cubemesh(N,nt,nt,families[3])
mesh[3].p[1,:] .= 0.5 .+ 0.25 .* mesh[3].p[1,:]

# Boundary order: y=0, right, y=1, left, z=0, z=1.
mesh[3].boundaryexpr=[
    x->abs.(x[2,:]).<1e-8,
    x->abs.(x[1,:].-0.75).<1e-8,
    x->abs.(x[2,:].-1).<1e-8,
    x->abs.(x[1,:].-0.5).<1e-8,
    x->abs.(x[3,:]).<1e-8,
    x->abs.(x[3,:].-1).<1e-8,
]
mesh[3].boundarycondition=reshape([1,3,1,2,1,1],:,1)
mesh[3].interfacecondition=zeros(Int,0,1)
