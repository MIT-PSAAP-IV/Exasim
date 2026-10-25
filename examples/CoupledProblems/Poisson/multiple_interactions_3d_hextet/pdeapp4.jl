# Domain 4: x in [0.75, 1].
pde[4],mesh[4]=Exasim.initializeexasim()

# All four domains use the same Poisson model.
pde[4].model="ModelD"
pde[4].modelfile="pdemodel"
pde[4].Cxxpreprocessing=1
pde[4].platform="cpu"
pde[4].mpiprocs=domainranks[4]
pde[4].hybrid=1

# Discretization, physical parameters, and solver controls.
pde[4].porder=porder
pde[4].pgauss=2*porder
pde[4].physicsparam=reshape([1.0,0.0],2,1) # diffusion; trace interface equation
pde[4].tau=[1.0]
pde[4].NLiter=12
pde[4].NLtol=1e-12
pde[4].linearsolvertol=1e-12
pde[4].matvectol=1e-12
pde[4].GMRESrestart=250
pde[4].linearsolveriter=600
pde[4].datapath=run_directory
pde[4].builddir=joinpath(run_directory,".exasim")
pde[4].buildpath=pde[4].builddir

# Adjacent domains use different tangential resolutions.
nt=isodd(4) ? N : 2*N
mesh[4].p,mesh[4].t=Exasim.Mesh.cubemesh(N,nt,nt,families[4])
mesh[4].p[1,:] .= 0.75 .+ 0.25 .* mesh[4].p[1,:]

# Boundary order: y=0, right, y=1, left, z=0, z=1.
mesh[4].boundaryexpr=[
    x->abs.(x[2,:]).<1e-8,
    x->abs.(x[1,:].-1).<1e-8,
    x->abs.(x[2,:].-1).<1e-8,
    x->abs.(x[1,:].-0.75).<1e-8,
    x->abs.(x[3,:]).<1e-8,
    x->abs.(x[3,:].-1).<1e-8,
]
mesh[4].boundarycondition=reshape([1,1,1,2,1,1],:,1)
mesh[4].interfacecondition=zeros(Int,0,1)
