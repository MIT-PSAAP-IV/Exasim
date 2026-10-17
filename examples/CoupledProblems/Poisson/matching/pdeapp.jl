# Julia version of the matching-interface coupled Poisson example (pdeapp.m).
# Two Poisson models on [0,0.5] x [0,1] (3 MPI ranks) and [0.5,1] x [0,1]
# (4 MPI ranks) are coupled through the matching interface x = 0.5.
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

"""
    squaremesh(m, n, parity=0, elemtype=0)

Regular mesh of the unit square, identical to MATLAB squaremesh.m: p is 2 x np,
t is nve x ne (one-based) with the same node and element ordering, so the three
frontends solve on the same mesh. elemtype 0 gives triangles (parity 0: SW-NE
diagonals, 1: NW-SE diagonals), elemtype 1 gives quadrilaterals.
"""
function squaremesh(m, n, parity=0, elemtype=0)
    mx, ny = m + 1, n + 1
    p = zeros(Float64, 2, mx * ny)
    for j in 0:ny-1, i in 0:mx-1
        p[1, 1 + i + j * mx] = i / (mx - 1)
        p[2, 1 + i + j * mx] = j / (ny - 1)
    end
    elements = Vector{Vector{Int}}()
    for j in 0:ny-2, i in 0:mx-2
        k = 1 + i + j * mx
        if elemtype == 1
            push!(elements, [k, k + 1, k + mx + 1, k + mx])
        elseif parity == 0
            push!(elements, [k, k + mx + 1, k + mx])
            push!(elements, [k, k + 1, k + mx + 1])
        else
            push!(elements, [k, k + 1, k + mx])
            push!(elements, [k + 1, k + mx + 1, k + mx])
        end
    end
    return p, reduce(hcat, elements)
end

"""
    dgnodes_of(mesh, porder, model, curved=false, width=0.5)

DG nodes in global element order, optionally mapped to a curved interface by
x += 0.05 sin(pi y) w, where w is the normalized distance from the outer
boundary of the subdomain (width = distance from that boundary to the
interface). The map fixes the outer boundaries and bends the interface.
"""
function dgnodes_of(mesh, porder, model, curved=false, width=0.5)
    elemtype = size(mesh.t, 1) == 4 ? 1 : 0
    f, _, _ = Exasim.Preprocessing.facenumbering(mesh.p, mesh.t, elemtype, mesh.boundaryexpr, [])
    dg = Exasim.Preprocessing.createdgnodes(mesh.p, mesh.t, f, [], [], porder)
    if curved
        w = model == 1 ? dg[:, 1, :] ./ width : (1.0 .- dg[:, 1, :]) ./ width
        dg[:, 1, :] .= dg[:, 1, :] .+ 0.05 .* sin.(pi .* dg[:, 2, :]) .* w
    end
    return dg
end

"""
    global_order(solution, dmd)

Solution in global element order. A one-rank model is returned in its partition
order (coupled interface elements last); several ranks are already assembled in
global order.
"""
function global_order(solution, dmd)
    if ndims(solution) == 4
        solution = solution[:, :, :, end]
    end
    if length(dmd) == 1
        elempart = vec(dmd[1].elempart)
        ordered = similar(solution)
        ordered[:, :, elempart[1:size(solution, 3)]] .= solution
        return ordered
    end
    return solution
end

"Maximum nodal error of u(:,1,:) against sin(pi x) sin(pi y)."
nodal_error(solution, dgnodes) = maximum(abs.(solution[:, 1, :] .-
    sin.(pi .* dgnodes[:, 1, :]) .* sin.(pi .* dgnodes[:, 2, :])))

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
