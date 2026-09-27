"""Two-block ISOQ mesh shared by the Julia verification driver."""

using LinearAlgebra

function loginc_values(values, alpha)
    lower, upper = extrema(values)
    upper == lower && return copy(values)
    lower .+ (upper - lower) .* (
        exp.(alpha .* (values .- lower) ./ (upper - lower)) .- 1.0
    ) ./ (exp(alpha) - 1.0)
end

function logdec_values(values, alpha)
    lower, upper = extrema(values)
    upper == lower && return copy(values)
    lower .+ (upper - lower) .* (
        1.0 .- exp.(-alpha .* (values .- lower) ./ (upper - lower))
    ) ./ (1.0 - exp(-alpha))
end

function rectangular_quad_mesh(nx, ny, porder)
    x = repeat(collect(range(0.0, 1.0, length=nx + 1)), ny + 1)
    y = repeat(collect(range(0.0, 1.0, length=ny + 1)), inner=nx + 1)
    points = vcat(reshape(x, 1, :), reshape(y, 1, :))
    elements = zeros(Int, 4, nx * ny)
    element = 1
    for iy in 0:(ny-1), ix in 0:(nx-1)
        lower_left = ix + iy * (nx + 1) + 1
        elements[:, element] .= [
            lower_left, lower_left + 1, lower_left + nx + 2, lower_left + nx + 1
        ]
        element += 1
    end
    dgnodes = Exasim.Preprocessing.createdgnodes(
        points, elements, zeros(Int, 4, size(elements, 2)), [], [], porder
    )
    return points, elements, dgnodes
end

function inverse_bilinear(point, coefficients, initial)
    xi, eta = initial
    for _ in 1:100
        residual = [1.0, xi, eta, xi * eta]' * coefficients - point'
        jacobian = [
            coefficients[2, 1] + coefficients[4, 1] * eta coefficients[3, 1] + coefficients[4, 1] * xi
            coefficients[2, 2] + coefficients[4, 2] * eta coefficients[3, 2] + coefficients[4, 2] * xi
        ]
        update = jacobian \ (-vec(residual))
        xi += update[1]
        eta += update[2]
        norm(update) < 1.0e-10 && return [xi, eta]
    end
    error("inverse bilinear map did not converge")
end

function polynomial_fit(x, y, degree)
    vandermonde = hcat([x .^ power for power in degree:-1:0]...)
    vandermonde \ y
end

function polynomial_values(coefficients, x)
    value = zero(x)
    for coefficient in coefficients
        value .= value .* x .+ coefficient
    end
    value
end

function forward_bilinear(reference_points, coefficients)
    xi, eta = reference_points[:, 1], reference_points[:, 2]
    hcat(ones(length(xi)), xi, eta, xi .* eta) * coefficients
end

function surfmesh2d(lower_curve, upper_curve, nx, ny, porder)
    matrix = [
        1.0 0.0 0.0 0.0
        1.0 1.0 0.0 0.0
        1.0 1.0 1.0 1.0
        1.0 0.0 1.0 0.0
    ]
    corners = vcat(
        lower_curve[1:1, :], lower_curve[end:end, :],
        upper_curve[end:end, :], upper_curve[1:1, :]
    )
    coefficients = matrix \ corners
    lower_reference = reduce(
        vcat,
        [reshape(inverse_bilinear(lower_curve[i, :], coefficients,
                                  ((i - 1) / (size(lower_curve, 1) - 1), 0.0)), 1, 2)
         for i in axes(lower_curve, 1)],
    )
    upper_reference = reduce(
        vcat,
        [reshape(inverse_bilinear(upper_curve[i, :], coefficients,
                                  ((i - 1) / (size(upper_curve, 1) - 1), 1.0)), 1, 2)
         for i in axes(upper_curve, 1)],
    )
    lower_fit = polynomial_fit(
        lower_reference[:, 1], lower_reference[:, 2], min(size(lower_curve, 1) - 1, 12)
    )
    upper_fit = polynomial_fit(
        upper_reference[:, 1], upper_reference[:, 2], min(size(upper_curve, 1) - 1, 12)
    )

    points, elements, dgnodes = rectangular_quad_mesh(nx, ny, porder)
    vertex_reference = copy(points')
    node_reference = hcat(vec(dgnodes[:, 1, :]), vec(dgnodes[:, 2, :]))
    sides = (
        vertex_left=abs.(vertex_reference[:, 1]) .< 1.0e-6,
        vertex_right=abs.(vertex_reference[:, 1] .- 1.0) .< 1.0e-6,
        node_left=abs.(node_reference[:, 1]) .< 1.0e-6,
        node_right=abs.(node_reference[:, 1] .- 1.0) .< 1.0e-6,
    )
    for reference in (vertex_reference, node_reference)
        reference[:, 1] .= logdec_values(loginc_values(reference[:, 1], 2.0), 1.5)
        reference[:, 2] .= logdec_values(loginc_values(reference[:, 2], 3.0), 1.0e-8)
        lower = polynomial_values(lower_fit, reference[:, 1])
        upper = polynomial_values(upper_fit, reference[:, 1])
        endpoints = (abs.(reference[:, 1]) .< 1.0e-6) .|
                    (abs.(reference[:, 1] .- 1.0) .< 1.0e-6)
        lower[endpoints] .= 0.0
        upper[endpoints] .= 1.0
        reference[:, 2] .= lower .+ (upper .- lower) .* reference[:, 2]
    end
    physical_points = forward_bilinear(vertex_reference, coefficients)
    physical_nodes = forward_bilinear(node_reference, coefficients)
    dgnodes[:, 1, :] .= reshape(physical_nodes[:, 1], size(dgnodes, 1), size(dgnodes, 3))
    dgnodes[:, 2, :] .= reshape(physical_nodes[:, 2], size(dgnodes, 1), size(dgnodes, 3))
    return physical_points', elements, dgnodes, sides
end

function isoq_curves()
    nose_radius = 0.102
    shoulder_radius = nose_radius / 16.0
    end_x = 0.06
    nose_angle = 27.82pi / 180.0
    theta = collect(range(0.0, nose_angle, length=481))
    nose = hcat(nose_radius .* (1.0 .- cos.(theta)), nose_radius .* sin.(theta))
    initial = pi - nose_angle
    center = nose[end, :] - shoulder_radius .* [cos(initial), sin(initial)]
    theta = collect(range(initial, pi / 2.0, length=49))
    shoulder = reshape(center, 1, 2) .+
               shoulder_radius .* hcat(cos.(theta), sin.(theta))
    afterbody = hcat(
        collect(range(shoulder[end, 1], end_x, length=240)),
        fill(shoulder[end, 2], 240),
    )
    lower = vcat(nose[1:end-1, :], shoulder[1:end-1, :], afterbody)
    theta = collect(range(pi, pi / 2.0, length=2000))
    upper = hcat(
        3.7end_x .+ 4.4end_x .* cos.(theta), 3.6end_x .* sin.(theta)
    )
    upper = upper[upper[:, 1] .<= end_x, :]
    upper[end, 1] = end_x
    return lower, upper
end

function connect_blocks(first, second; tolerance=1.0e-5)
    p1, t1, dgnodes1, side1 = first
    p2, t2, dgnodes2, side2 = second
    average_vertices = 0.5 .* (
        p1[:, side1.vertex_right]' .+ p2[:, side2.vertex_left]'
    )
    p1[:, side1.vertex_right] .= average_vertices'
    p2[:, side2.vertex_left] .= average_vertices'
    flat1 = hcat(vec(dgnodes1[:, 1, :]), vec(dgnodes1[:, 2, :]))
    flat2 = hcat(vec(dgnodes2[:, 1, :]), vec(dgnodes2[:, 2, :]))
    average_nodes = 0.5 .* (flat1[side1.node_right, :] .+ flat2[side2.node_left, :])
    flat1[side1.node_right, :] .= average_nodes
    flat2[side2.node_left, :] .= average_nodes
    dgnodes1[:, 1, :] .= reshape(flat1[:, 1], size(dgnodes1, 1), size(dgnodes1, 3))
    dgnodes1[:, 2, :] .= reshape(flat1[:, 2], size(dgnodes1, 1), size(dgnodes1, 3))
    dgnodes2[:, 1, :] .= reshape(flat2[:, 1], size(dgnodes2, 1), size(dgnodes2, 3))
    dgnodes2[:, 2, :] .= reshape(flat2[:, 2], size(dgnodes2, 1), size(dgnodes2, 3))

    mapping = zeros(Int, size(p2, 2))
    keep = trues(size(p2, 2))
    next_index = size(p1, 2) + 1
    for index in axes(p2, 2)
        squared_distance = vec(sum((p1 .- p2[:, index]).^2, dims=1))
        nearest = argmin(squared_distance)
        if squared_distance[nearest] < tolerance^2
            mapping[index] = nearest
            keep[index] = false
        else
            mapping[index] = next_index
            next_index += 1
        end
    end
    points = hcat(p1, p2[:, keep])
    elements = hcat(t1, mapping[t2])
    dgnodes = cat(dgnodes1, dgnodes2, dims=3)
    return points, elements, dgnodes
end

function make_isoq_mesh(porder; radial_shift=5.0e-4)
    lower, upper = isoq_curves()
    upper_first = upper[upper[:, 1] .<= -0.02, :]
    upper_second = vcat(upper_first[end:end, :], upper[upper[:, 1] .> -0.02, :])
    lower_first = lower[lower[:, 1] .<= 0.013, :]
    lower_second = vcat(lower_first[end:end, :], lower[lower[:, 1] .> 0.013, :])
    points, elements, dgnodes = connect_blocks(
        surfmesh2d(lower_first, upper_first, 30, 60, porder),
        surfmesh2d(lower_second, upper_second, 25, 60, porder),
    )
    points[2, :] .+= radial_shift
    dgnodes[:, 2, :] .+= radial_shift
    lower_y, right_x = minimum(points[2, :]), maximum(points[1, :])
    boundaryexpr = [
        p -> abs.(p[2, :] .- lower_y) .< 1.0e-6,
        p -> p[1, :] .> right_x - 1.0e-4,
        p -> (p[1, :] .< -1.0e-3) .| (p[2, :] .> 0.1),
        p -> abs.(p[1, :]) .< 20.0 + 1.0e-6,
    ]
    faces, _, _ = Exasim.Preprocessing.facenumbering(points, elements, 1, boundaryexpr, [])
    _, mesh = Exasim.initializeexasim()
    mesh.p = points
    mesh.t = elements
    mesh.f = faces
    mesh.dgnodes = dgnodes
    mesh.boundaryexpr = boundaryexpr
    mesh.boundarycondition = reshape([5, 2, 1, 3], :, 1)
    mesh.periodicexpr = []
    mesh.curvedboundary = zeros(Int, 0, 0)
    mesh.curvedboundaryexpr = []
    mesh.periodicboundary = zeros(Int, 0, 0)
    return mesh
end

function wall_distance(mesh, porder)
    _, _, _, _, perm = Exasim.Preprocessing.Master.masternodes(porder, 2, 1)
    boundary_faces = findall(mesh.f .== 4)
    wall_nodes = reduce(
        vcat,
        [mesh.dgnodes[perm[:, index[1]], :, index[2]] for index in boundary_faces],
    )
    npe, _, ne = size(mesh.dgnodes)
    distance = zeros(Float64, npe, 1, ne)
    for element in 1:ne, node in 1:npe
        delta = wall_nodes .- reshape(mesh.dgnodes[node, :, element], 1, 2)
        distance[node, 1, element] = minimum(sqrt.(sum(delta .* delta, dims=2)))
    end
    distance
end
