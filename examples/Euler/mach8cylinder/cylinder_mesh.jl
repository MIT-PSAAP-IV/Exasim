"""Structured curved annulus used by the legacy Mach-8.98 cylinder case."""

loginc_values(x, a) = minimum(x) .+ (maximum(x) - minimum(x)) .* (
    exp.(a .* (x .- minimum(x)) ./ (maximum(x) - minimum(x))) .- 1.0
) ./ (exp(a) - 1.0)
logdec_values(x, a) = minimum(x) .+ (maximum(x) - minimum(x)) .* (
    1.0 .- exp.(-a .* (x .- minimum(x)) ./ (maximum(x) - minimum(x)))
) ./ (1.0 - exp(-a))

function quad_mesh(nx, ny)
    x = repeat(collect(range(0.0, 1.0, length=nx + 1)), ny + 1)
    y = repeat(collect(range(0.0, 1.0, length=ny + 1)), inner=nx + 1)
    points = vcat(reshape(x, 1, :), reshape(y, 1, :))
    elements = zeros(Int, 4, nx * ny)
    element = 1
    for iy in 0:(ny-1), ix in 0:(nx-1)
        node = ix + iy * (nx + 1) + 1
        elements[:, element] .= [node, node + 1, node + nx + 2, node + nx + 1]
        element += 1
    end
    return points, elements
end

function map_annulus(points; inner=1.0, outer=3.0)
    angle = pi .* points[1, :]
    radius = inner .+ (outer - inner) .* points[2, :]
    vcat(reshape(-radius .* sin.(angle), 1, :), reshape(-radius .* cos.(angle), 1, :))
end

function map_annulus_nodes(nodes; inner=1.0, outer=3.0)
    angle = pi .* nodes[:, 1, :]
    radius = inner .+ (outer - inner) .* nodes[:, 2, :]
    mapped = similar(nodes)
    mapped[:, 1, :] .= -radius .* sin.(angle)
    mapped[:, 2, :] .= -radius .* cos.(angle)
    mapped
end

function make_cylinder_mesh(porder; m=101, n=101)
    points, elements = quad_mesh(m - 1, n - 1)
    n1, n2 = ceil(Int, 5.5n / 10), ceil(Int, 3.05n / 10)
    n3 = n - n1 - n2
    radial = vcat(
        loginc_values(collect(range(0.0, 0.1, length=n1 + 1)), 3.0)[1:end-1],
        loginc_values(collect(range(0.1, 0.375, length=n2 + 1)), 1.0)[1:end-1],
        loginc_values(collect(range(0.375, 1.0, length=n3)), 2.0),
    )
    points[2, :] .= repeat(radial, m)
    left = points[1, :] .<= 0.5
    points[1, left] .= logdec_values(points[1, left], 1.0)
    points[1, .!left] .= loginc_values(points[1, .!left], 1.0)
    dgnodes = Exasim.Preprocessing.createdgnodes(
        points, elements, zeros(Int, 4, size(elements, 2)), [], [], porder
    )
    points = map_annulus(points)
    dgnodes = map_annulus_nodes(dgnodes)
    boundaryexpr = [
        p -> sqrt.(p[1, :].^2 .+ p[2, :].^2) .< 1.0 + 1.0e-6,
        p -> p[1, :] .> -1.0e-7,
        p -> abs.(p[1, :]) .< 20.0,
    ]
    faces, _, _ = Exasim.Preprocessing.facenumbering(points, elements, 1, boundaryexpr, [])
    _, mesh = Exasim.initializeexasim()
    mesh.p, mesh.t, mesh.f, mesh.dgnodes = points, elements, faces, dgnodes
    mesh.boundaryexpr = boundaryexpr
    mesh.boundarycondition = reshape([3, 6, 5], :, 1)
    return mesh
end

function wall_distance(mesh, porder)
    _, _, _, _, perm = Exasim.Preprocessing.Master.masternodes(porder, 2, 1)
    indices = findall(mesh.f .== 1)
    wall = reduce(vcat, [mesh.dgnodes[perm[:, i[1]], :, i[2]] for i in indices])
    npe, _, ne = size(mesh.dgnodes)
    distance = zeros(npe, 1, ne)
    for e in 1:ne, a in 1:npe
        delta = wall .- reshape(mesh.dgnodes[a, :, e], 1, 2)
        distance[a, 1, e] = minimum(sqrt.(sum(delta .* delta, dims=2)))
    end
    distance
end

curved_map_scale(mesh) = sqrt.(2.0 .* pi .* sqrt.(mesh.dgnodes[:, 1, :].^2 .+ mesh.dgnodes[:, 2, :].^2))

function initialize_solution(mesh, distance, parameters)
    npe, _, ne = size(mesh.dgnodes)
    solution = zeros(npe, 4, ne)
    solution[:, 1, :] .= parameters[5]
    solution[:, 2, :] .= parameters[6] .* tanh.(10.0 .* distance[:, 1, :])
    solution[:, 3, :] .= parameters[7] .* tanh.(10.0 .* distance[:, 1, :])
    temperature = parameters[9] .* (parameters[11] / parameters[10] - 1.0) .* exp.(-10.0 .* distance[:, 1, :]) .+ parameters[9]
    solution[:, 4, :] .= temperature .+ 0.5 .* (solution[:, 2, :].^2 .+ solution[:, 3, :].^2)
    solution
end
