"""Exact Julia translation of mkmesh_cyl.m and meshdist3 for this example."""

function logdec_values(values, alpha)
    a = minimum(values)
    b = maximum(values)
    return a .+ (b - a) .* (1.0 .- exp.(-alpha .* (values .- a) ./ (b - a))) ./
           (1.0 - exp(-alpha))
end

function map_halfcircle_points(points; a=1.0, b=3.0, c=4.7)
    theta = 1.5pi .+ points[2, :] .* (0.5pi - 1.5pi)
    outer_radius = -b .* cos.(theta) .+ c .* (1.0 .+ cos.(theta))
    radius = outer_radius .+ points[1, :] .* (a .- outer_radius)
    return vcat(
        reshape(radius .* cos.(theta), 1, :),
        reshape(radius .* sin.(theta), 1, :),
    )
end

function map_halfcircle_dgnodes(nodes; a=1.0, b=3.0, c=4.7)
    theta = 1.5pi .+ nodes[:, 2, :] .* (0.5pi - 1.5pi)
    outer_radius = -b .* cos.(theta) .+ c .* (1.0 .+ cos.(theta))
    radius = outer_radius .+ nodes[:, 1, :] .* (a .- outer_radius)
    mapped = similar(nodes)
    mapped[:, 1, :] .= radius .* cos.(theta)
    mapped[:, 2, :] .= radius .* sin.(theta)
    return mapped
end

function rectangular_quad_mesh(nx, ny)
    x = repeat(collect(range(0.0, 1.0, length=nx + 1)), ny + 1)
    y = repeat(collect(range(0.0, 1.0, length=ny + 1)), inner=nx + 1)
    p = vcat(reshape(x, 1, :), reshape(y, 1, :))
    t = zeros(Int, 4, nx * ny)
    element = 1
    for iy in 0:(ny-1), ix in 0:(nx-1)
        lower_left = ix + iy * (nx + 1) + 1
        t[:, element] .= [
            lower_left,
            lower_left + 1,
            lower_left + nx + 2,
            lower_left + nx + 1,
        ]
        element += 1
    end
    return p, t
end

function make_cylinder_mesh(
    porder; nx=31, ny=21, radial_decay=6.0, outer_offset=4.7
)
    p, t = rectangular_quad_mesh(nx, ny)
    f0 = zeros(Int, 4, size(t, 2))
    dgnodes = Exasim.Preprocessing.createdgnodes(p, t, f0, [], [], porder)

    p[1, :] .= logdec_values(p[1, :], radial_decay)
    dgnodes[:, 1, :] .= logdec_values(dgnodes[:, 1, :], radial_decay)
    p = map_halfcircle_points(p; c=outer_offset)
    dgnodes = map_halfcircle_dgnodes(dgnodes; c=outer_offset)

    boundaryexpr = [
        x -> sqrt.(x[1, :].^2 .+ x[2, :].^2) .< 1.0 + 1.0e-6,
        x -> x[1, :] .> -1.0e-7,
        x -> abs.(x[1, :]) .< 20.0,
    ]
    f, _, _ = Exasim.Preprocessing.facenumbering(p, t, 1, boundaryexpr, [])

    _, mesh = Exasim.initializeexasim()
    mesh.p = p
    mesh.t = t
    mesh.f = f
    mesh.dgnodes = dgnodes
    mesh.boundaryexpr = boundaryexpr
    mesh.boundarycondition = reshape([3, 6, 5], :, 1)
    mesh.periodicexpr = []
    mesh.curvedboundary = zeros(Int, 0, 0)
    mesh.curvedboundaryexpr = []
    mesh.periodicboundary = zeros(Int, 0, 0)
    return mesh
end

function wall_distance(mesh, porder)
    _, _, _, _, perm = Exasim.Preprocessing.Master.masternodes(porder, 2, 1)
    boundary_faces = findall(mesh.f .== 1)
    wall_nodes = reduce(
        vcat,
        [mesh.dgnodes[perm[:, index[1]], :, index[2]] for index in boundary_faces],
    )

    npe, _, ne = size(mesh.dgnodes)
    distance = zeros(Float64, npe, 1, ne)
    for elem in 1:ne, node in 1:npe
        point = mesh.dgnodes[node, :, elem]
        delta = wall_nodes .- reshape(point, 1, 2)
        distance[node, 1, elem] = minimum(sqrt.(sum(delta .* delta, dims=2)))
    end
    return distance
end

function initialize_solution(mesh, distance, physicsparam)
    rinf, ruinf, rvinf, rEinf = physicsparam[5:8]
    tinf, tref, twall = physicsparam[9:11]
    npe, _, ne = size(mesh.dgnodes)
    udg = zeros(Float64, npe, 4, ne)
    udg[:, 1, :] .= rinf
    udg[:, 2, :] .= ruinf .* tanh.(10.0 .* distance[:, 1, :])
    udg[:, 3, :] .= rvinf .* tanh.(10.0 .* distance[:, 1, :])
    t_near_wall = tinf .* (twall / tref - 1.0) .* exp.(-10.0 .* distance[:, 1, :]) .+ tinf
    udg[:, 4, :] .= t_near_wall .+ 0.5 .* (udg[:, 2, :].^2 .+ udg[:, 3, :].^2)
    return udg
end
