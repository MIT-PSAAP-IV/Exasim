# Model 1 of the mixed triangle/quadrilateral Poisson example.
# Included by pdeapp.jl, which loads Exasim and defines squaremesh.
pde[1], mesh[1] = Exasim.initializeexasim();

pde[1].model = "ModelD";
pde[1].modelfile = "pdemodel1";    # module defined in pdemodel1.jl
include(joinpath(@__DIR__, "pdemodel1.jl"));
pde[1].Cxxpreprocessing = 1;
pde[1].mpiprocs = 1;
pde[1].hybrid = 1;
pde[1].NLiter = 3;
pde[1].NLtol = 1e-10;
pde[1].linearsolvertol = 1e-10;
pde[1].matvectol = 1e-10;
pde[1].GMRESrestart = 300;
pde[1].linearsolveriter = 1000;
pde[1].porder = 3;
pde[1].physicsparam = reshape([1.0], 1, 1);   # unit thermal conductivity
pde[1].tau = [1.0];                           # DG stabilization parameter
pde[1].interfacetype = 1;

triangleOnModel1 = @isdefined(triangleOnModel1) ? triangleOnModel1 : true;
mesh[1].p, mesh[1].t = squaremesh(8, 16, 1, triangleOnModel1 ? 0 : 1);
mesh[1].p[1,:] = 0.5 .* mesh[1].p[1,:];
mesh[1].boundaryexpr = [p -> abs.(p[2,:]) .< 1e-8, p -> abs.(p[1,:] .- 0.5) .< 1e-8,
    p -> abs.(p[2,:] .- 1) .< 1e-8, p -> abs.(p[1,:]) .< 1e-8];
mesh[1].boundarycondition = [1 2 1 1];
mesh[1].interfacecondition = [0 1 0 0];
pde[1].interfacefluxmap = [1];
