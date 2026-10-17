# Model 2 of the matching-interface coupled Poisson example: [0.5,1] x [0,1].
# Included by pdeapp.jl, which loads Exasim and defines squaremesh.
pde[2], mesh[2] = Exasim.initializeexasim();

pde[2].model = "ModelD";
pde[2].modelfile = "pdemodel2";    # module defined in pdemodel2.jl
include(joinpath(@__DIR__, "pdemodel2.jl"));
pde[2].Cxxpreprocessing = 1;
pde[2].mpiprocs = 4;
pde[2].hybrid = 1;
pde[2].linearsolvertol = 1e-6;
pde[2].NLiter = 1;
pde[2].porder = 3;
pde[2].physicsparam = reshape([1.0], 1, 1);   # unit thermal conductivity
pde[2].tau = [1.0];                           # DG stabilization parameter

mesh[2].p, mesh[2].t = squaremesh(8, 16, 1, 1);
mesh[2].p[1,:] = 0.5 .* mesh[2].p[1,:] .+ 0.5;
mesh[2].boundaryexpr = [p -> abs.(p[2,:]) .< 1e-8, p -> abs.(p[1,:] .- 1) .< 1e-8,
    p -> abs.(p[2,:] .- 1) .< 1e-8, p -> abs.(p[1,:] .- 0.5) .< 1e-8];
mesh[2].boundarycondition = [1 1 1 2];
