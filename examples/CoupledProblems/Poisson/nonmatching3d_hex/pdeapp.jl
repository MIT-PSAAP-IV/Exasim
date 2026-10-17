# Julia version of the 3D non-matching coupled Poisson example (pdeapp.m).
# Two Poisson models on the unit cube (hexahedra, interfacetype = 1) coupled
# through the interface x = 0.5. The exact solution is
# u = exp(x) sin(pi y) sin(pi z). Both meshes come from cubemesh.
# Run with `julia --project=<repo>/frontends/Julia/Exasim pdeapp.jl`; data and the
# solver build go to ./julia_run.

# Use EXASIM_PREFIX if set, otherwise a sibling <repo>/../exasim_install, and
# put the source-tree Julia frontend on LOAD_PATH.
let repository = dirname(dirname(dirname(dirname(@__DIR__))))
    source_install = joinpath(dirname(repository), "exasim_install")
    if !haskey(ENV, "EXASIM_PREFIX") &&
       isfile(joinpath(source_install, "lib", "cmake", "Exasim", "ExasimConfig.cmake"))
        ENV["EXASIM_PREFIX"] = source_install
    end
    frontend = joinpath(repository, "frontends", "Julia", "Exasim")
    frontend in LOAD_PATH || pushfirst!(LOAD_PATH, frontend)
end
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

sol, pde, mesh, master, dmd, _, _ = Exasim.exasim(pde, mesh);

# Maximum nodal error against the exact solution. A one-rank model returns its
# solution in partition order, so it is put back into global element order first.
function nodal_error(sol, pde, mesh, dmd)
    elemtype = size(mesh.t, 1) == 8 ? 1 : 0
    f, _, _ = Exasim.Preprocessing.facenumbering(mesh.p, mesh.t, elemtype, mesh.boundaryexpr, [])
    xdg = Exasim.Preprocessing.createdgnodes(mesh.p, mesh.t, f, [], [], pde.porder)
    u = ndims(sol) == 4 ? sol[:, :, :, end] : sol
    if length(dmd) == 1
        ordered = similar(u)
        ordered[:, :, vec(dmd[1].elempart)[1:size(u, 3)]] .= u
        u = ordered
    end
    return maximum(abs.(u[:, 1, :] .- exp.(xdg[:, 1, :]) .* sin.(pi .* xdg[:, 2, :]) .*
        sin.(pi .* xdg[:, 3, :])))
end
err = maximum(nodal_error(sol[m], pde[m], mesh[m], dmd[m]) for m = 1:2);
println("Maximum nodal error against exp(x) sin(pi y) sin(pi z): ", err);
