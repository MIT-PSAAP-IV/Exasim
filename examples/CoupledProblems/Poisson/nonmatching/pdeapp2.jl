# Model 2 of the non-matching coupled Poisson example: [0.5,1] x [0,1], 12 x 24
# quadrilaterals, so the two interface discretizations do not match (h ratio 3:2).
# Included by pdeapp.jl, which loads Exasim and defines squaremesh.
pde[2], mesh[2] = Exasim.initializeexasim();

pde[2].model = "ModelD";
pde[2].modelfile = "pdemodel2";    # module defined in pdemodel2.jl
include(joinpath(@__DIR__, "pdemodel2.jl"));
pde[2].Cxxpreprocessing = 1;
pde[2].mpiprocs = 1;
pde[2].hybrid = 1;
pde[2].NLiter = 3;
pde[2].NLtol = 1e-10;
pde[2].linearsolvertol = 1e-10;
pde[2].matvectol = 1e-10;
pde[2].GMRESrestart = 300;
pde[2].linearsolveriter = 1000;
pde[2].porder = 2;
pde[2].physicsparam = reshape([1.0], 1, 1);   # unit thermal conductivity
pde[2].tau = [1.0];                           # DG stabilization parameter
pde[2].interfacetype = 1;

mesh[2].p, mesh[2].t = squaremesh(12, 24, 1, 1);
mesh[2].p[1,:] = 0.5 .* mesh[2].p[1,:] .+ 0.5;
mesh[2].boundaryexpr = [p -> abs.(p[2,:]) .< 1e-8, p -> abs.(p[1,:] .- 1) .< 1e-8,
    p -> abs.(p[2,:] .- 1) .< 1e-8, p -> abs.(p[1,:] .- 0.5) .< 1e-8];
mesh[2].boundarycondition = [1 1 1 2];
mesh[2].interfacecondition = [0 0 0 1]   # boundary 4 is the coupled interface;
pde[2].interfacefluxmap = [1];
