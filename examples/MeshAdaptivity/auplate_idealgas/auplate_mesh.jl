"""Frontend-neutral rounded-nose flat-plate mesh construction."""

const RNOSE = 8.0e-5
const YSHIFT = 1.0e-4
const XEND = 0.5
const HNOSE = 0.02
const HEND = 0.15

function structured_quads(nx, ny)
    points = zeros(2, (nx + 1) * (ny + 1))
    for j in 0:ny, i in 0:nx
        node = i + 1 + j * (nx + 1)
        points[:, node] .= (i / nx, j / ny)
    end
    elements = zeros(Int, 4, nx * ny)
    for j in 0:(ny - 1), i in 0:(nx - 1)
        element = i + 1 + j * nx
        lower_left = i + 1 + j * (nx + 1)
        elements[:, element] .= (lower_left, lower_left + 1,
                                 lower_left + nx + 2, lower_left + nx + 1)
    end
    return points, elements
end

function map_auplate(reference)
    xi, eta = reference[1, :], reference[2, :]
    x, y = similar(xi), similar(eta)
    nose = xi .<= 1.0 + 1.0e-12
    angle = 0.5pi .* xi[nose]
    height = HNOSE .* eta[nose]
    x[nose] .= RNOSE .- (RNOSE .+ height) .* cos.(angle)
    y[nose] .= (RNOSE .+ height) .* sin.(angle) .+ YSHIFT
    body = .!nose
    fraction = xi[body] .- 1.0
    wall_x = RNOSE .+ (XEND - RNOSE) .* fraction
    top_y = RNOSE + YSHIFT + HNOSE .+ (HEND - HNOSE) .* fraction
    x[body] .= wall_x
    y[body] .= (1.0 .- eta[body]) .* (RNOSE + YSHIFT) .+ eta[body] .* top_y
    return vcat(x', y')
end

function make_auplate_mesh(porder)
    points, elements = structured_quads(116, 41)
    reference_points = vcat(reshape(2.0 .* points[1, :], 1, :),
                            reshape(points[2, :], 1, :))
    reference_nodes = Exasim.Preprocessing.createdgnodes(
        reference_points, elements, zeros(Int, 4, size(elements, 2)), [], [], porder)
    mapped_points = map_auplate(reference_points)
    mapped_nodes = map_auplate(reshape(permutedims(reference_nodes, (2, 1, 3)), 2, :))
    dgnodes = permutedims(reshape(mapped_nodes, 2, size(reference_nodes, 1), size(reference_nodes, 3)), (2, 1, 3))

    center_x, center_y = RNOSE, YSHIFT
    outer_radius = RNOSE + HNOSE
    wall_y = RNOSE + YSHIFT
    boundaryexpr = [
        p -> (abs.(p[2, :] .- YSHIFT) .< 1.0e-6) .& (p[1, :] .<= 1.0e-6),
        p -> abs.(hypot.(p[1, :] .- center_x, p[2, :] .- center_y) .- outer_radius) .< 5.0e-6,
        p -> abs.(hypot.(p[1, :] .- center_x, p[2, :] .- center_y) .- RNOSE) .< 1.0e-6,
        p -> (abs.(p[2, :] .- wall_y) .< 1.0e-6) .& (p[1, :] .>= RNOSE - 1.0e-6),
        p -> p[1, :] .> XEND - 1.0e-6,
        p -> p[2, :] .> wall_y + 0.5HNOSE,
    ]
    faces, _, _ = Exasim.Preprocessing.facenumbering(mapped_points, elements, 1, boundaryexpr, [])
    _, mesh = Exasim.initializeexasim()
    mesh.p, mesh.t, mesh.f, mesh.dgnodes = mapped_points, elements, faces, dgnodes
    mesh.boundaryexpr = boundaryexpr
    mesh.boundarycondition = reshape([4, 1, 3, 3, 2, 1], :, 1)
    mesh.periodicexpr = []
    return mesh
end

function wall_distance(mesh)
    x, y = mesh.dgnodes[:, 1, :], mesh.dgnodes[:, 2, :]
    nose = abs.(hypot.(x .- RNOSE, y .- YSHIFT) .- RNOSE)
    plate = hypot.(max.(RNOSE .- x, 0.0), y .- (RNOSE + YSHIFT))
    return reshape(min.(nose, abs.(plate)), size(x, 1), 1, size(x, 2))
end
