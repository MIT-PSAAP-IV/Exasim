# Helpers shared by the Julia versions of the coupled Poisson examples.
# Include this file before `using Exasim`: it puts the source-tree Julia
# frontend on LOAD_PATH and selects a sibling source install
# (<repo>/../exasim_install) unless EXASIM_PREFIX is set.

const COUPLEDPOISSON_REPOSITORY = dirname(dirname(dirname(@__DIR__)))
let source_install = joinpath(dirname(COUPLEDPOISSON_REPOSITORY), "exasim_install")
    if !haskey(ENV, "EXASIM_PREFIX") &&
       isfile(joinpath(source_install, "lib", "cmake", "Exasim", "ExasimConfig.cmake"))
        ENV["EXASIM_PREFIX"] = source_install
    end
end
let frontend = joinpath(COUPLEDPOISSON_REPOSITORY, "frontends", "Julia", "Exasim")
    frontend in LOAD_PATH || pushfirst!(LOAD_PATH, frontend)
end

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

"Write sol<m>.bin and xdg<m>.bin (ndims, dims, column-major data) for cross-frontend checks."
function save_results(directory, solutions, dgnodes)
    mkpath(directory)
    for m in eachindex(solutions)
        for (name, a) in (("sol", solutions[m]), ("xdg", dgnodes[m]))
            open(joinpath(directory, "$name$m.bin"), "w") do io
                write(io, Float64(ndims(a)))
                write(io, Float64.(collect(size(a))))
                write(io, Float64.(vec(a)))
            end
        end
    end
end
