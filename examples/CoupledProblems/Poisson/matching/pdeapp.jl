# Julia version of the matching-interface coupled Poisson example (pdeapp.m).
# Two Poisson models on [0,0.5] x [0,1] (3 MPI ranks) and [0.5,1] x [0,1]
# (4 MPI ranks) are coupled through the matching interface x = 0.5.
# Run with `julia --project=<repo>/frontends/Julia/Exasim pdeapp.jl`; data and the
# solver build go to ./julia_run.

include(joinpath(dirname(@__DIR__), "coupledpoisson.jl"))
using Exasim
using DelimitedFiles  # the frontend METIS partitioner (mpiprocs > 1) calls Main.DelimitedFiles

pde = Array{Any, 1}(undef, 2);
mesh = Array{Any, 1}(undef, 2);
include(joinpath(@__DIR__, "pdeapp1.jl"));
include(joinpath(@__DIR__, "pdeapp2.jl"));

run_directory = joinpath(@__DIR__, "julia_run");
for m = 1:2
    pde[m].datapath = run_directory;
    pde[m].builddir = joinpath(run_directory, ".exasim");
    pde[m].buildpath = pde[m].builddir;
end

# set an interface condition for the interface of pde1
# the size of this array must be equal to the number of boundaries for pde1
# a nonzero element of this array corresponds to a particular fint of pdemodel1
mesh[1].interfacecondition = [0 1 0 0];

# a list of fbou components that are used to impose interface condition
pde[1].interfacefluxmap = [1];

# set an interface condition for the interface of pde2
mesh[2].interfacecondition = [0 0 0 1];
pde[2].interfacefluxmap = [1];

sol, pde, mesh, master, dmd, _, _ = Exasim.exasim(pde, mesh);

xdg = [dgnodes_of(mesh[m], pde[m].porder, m) for m = 1:2];
sol = [global_order(sol[m], dmd[m]) for m = 1:2];
err = maximum(nodal_error(sol[m], xdg[m]) for m = 1:2);
println("Maximum nodal error against sin(pi x) sin(pi y): ", err);
save_results(joinpath(run_directory, "results"), sol, xdg);
