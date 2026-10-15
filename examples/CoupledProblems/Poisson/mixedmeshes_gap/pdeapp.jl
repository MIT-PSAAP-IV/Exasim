# Julia version of the gapped mixed triangle/quadrilateral Poisson example (pdeapp.m).
# The mixed triangle/quadrilateral case with a 0.02 gap between the two
# subdomains around x = 0.5. The exact solution is u = sin(pi x) sin(pi y).
# Run with `julia --project=<repo>/frontends/Julia/Exasim pdeapp.jl`; data and the
# solver build go to ./julia_run.

include(joinpath(dirname(@__DIR__), "coupledpoisson.jl"))
using Exasim
using DelimitedFiles  # the frontend METIS partitioner (mpiprocs > 1) calls Main.DelimitedFiles

triangleOnModel1 = true; # false reverses the triangle/quad orientation
curved = false;
gap = 0.02;

pde = Array{Any, 1}(undef, 2);
mesh = Array{Any, 1}(undef, 2);
include(joinpath(@__DIR__, "pdeapp1.jl"));
include(joinpath(@__DIR__, "pdeapp2.jl"));

actualGap = minimum(mesh[2].p[1,:]) - maximum(mesh[1].p[1,:]);
abs(actualGap - gap) < 1e-12 || error("Expected interface gap $gap, found $actualGap.");
println("Mixed triangle/quad interface gap: ", actualGap);

run_directory = joinpath(@__DIR__, "julia_run");
for m = 1:2
    pde[m].datapath = run_directory;
    pde[m].builddir = joinpath(run_directory, ".exasim");
    pde[m].buildpath = pde[m].builddir;
end

# DG nodes in the global element order of each mesh, optionally mapped to the
# curved interface (the map fixes the outer boundaries). preprocessing() uses
# mesh[m].dgnodes when it is present.
for m = 1:2
    mesh[m].dgnodes = dgnodes_of(mesh[m], pde[m].porder, m, curved, 0.5 - gap / 2);
end

sol, pde, mesh, master, dmd, _, _ = Exasim.exasim(pde, mesh);

xdg = [dgnodes_of(mesh[m], pde[m].porder, m, curved, 0.5 - gap / 2) for m = 1:2];
sol = [global_order(sol[m], dmd[m]) for m = 1:2];
err = maximum(nodal_error(sol[m], xdg[m]) for m = 1:2);
println("Gapped mixed triangle/quad maximum nodal error: ", err);
save_results(joinpath(run_directory, "results"), sol, xdg);
