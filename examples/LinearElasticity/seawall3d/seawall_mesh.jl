function extrude_quads(points2d, elements2d, z)
    nxy, ne2d = size(points2d, 2), size(elements2d, 2)
    points = Matrix{Float64}(undef, 3, nxy * length(z))
    for layer in eachindex(z)
        columns = (1:nxy) .+ (layer - 1) * nxy
        points[1:2, columns] = points2d
        points[3, columns] .= z[layer]
    end
    elements = Matrix{Int}(undef, 8, ne2d * (length(z) - 1))
    for layer in 1:length(z)-1
        columns = (1:ne2d) .+ (layer - 1) * ne2d
        elements[1:4, columns] = elements2d .+ (layer - 1) * nxy
        elements[5:8, columns] = elements2d .+ layer * nxy
    end
    points, elements
end

function map_xz(x, z, xmax, l1, radius, l2, l3)
    mapped_x, mapped_z = copy(x), copy(z)
    mask = (z .>= l2 + 1.0) .& (z .<= l2 + 1.0 + l3)
    mapped_x[mask] = xmax + radius .+ (z[mask] .- (l2 + 1.0)); mapped_z[mask] = l2 + radius + xmax .- x[mask]
    mask = (z .>= l2) .& (z .<= l2 + 1.0); angle = pi .- 0.5pi .* (z[mask] .- l2); radial = xmax + radius .- x[mask]
    mapped_x[mask] = xmax + radius .+ radial .* cos.(angle); mapped_z[mask] = l2 .+ radial .* sin.(angle)
    mask = (z .>= -1.0) .& (z .<= 0.0); angle = 1.5pi .- 0.5pi .* (z[mask] .+ 1.0); radial = xmax + radius .- x[mask]
    mapped_x[mask] = xmax + radius .+ radial .* cos.(angle); mapped_z[mask] = radial .* sin.(angle)
    mask = (z .>= -l1 - 1.0) .& (z .<= -1.0)
    mapped_x[mask] = xmax + radius .- (z[mask] .+ 1.0); mapped_z[mask] = -radius - xmax .+ x[mask]
    mapped_x, mapped_z
end

function make_seawall_mesh(pde)
    geometry = joinpath(@__DIR__, "seawall")
    points2d, elements2d = Exasim.Mesh.gmshcall(pde, geometry, 2, 1)
    elements2d = reverse(elements2d, dims=1)
    l1, radius, l2, l3 = 41.0, 4.0, 210.0, 41.0
    counts = [80, 20, 400, 20, 80]
    pieces = [range(0.0, l1, length=counts[1]), range(l1, l1 + 1.0, length=counts[2]),
        range(l1 + 1.0, l1 + 1.0 + l2, length=counts[3]),
        range(l1 + 1.0 + l2, l1 + 2.0 + l2, length=counts[4]),
        range(l1 + 2.0 + l2, l1 + 2.0 + l2 + l3, length=counts[5])]
    z = vcat(collect(pieces[1]), (collect(piece)[2:end] for piece in pieces[2:end])...) .- (l1 + 1.0)
    points, elements = extrude_quads(points2d, elements2d, z)
    dgnodes = Exasim.Preprocessing.createdgnodes(points, elements, zeros(Int, 6, size(elements, 2)), [], [], pde.porder)
    x0, z0 = copy(points[1, :]), copy(points[3, :])
    xmax = maximum(x0)
    points[1, :], points[3, :] = map_xz(x0, z0, xmax, l1, radius, l2, l3)
    dgnodes[:, 1, :], dgnodes[:, 3, :] = map_xz(dgnodes[:, 1, :], dgnodes[:, 3, :], xmax, l1, radius, l2, l3)
    points, elements, dgnodes
end
