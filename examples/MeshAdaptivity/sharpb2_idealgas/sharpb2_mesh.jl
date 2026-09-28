"""Two-block Sharp-B mesh for the Julia mesh-adaptivity driver."""

include(joinpath(dirname(@__DIR__), "isoq2d_idealgas", "isoq_mesh.jl"))

function circle_tangent_point(center_x, center_y, radius, point_x, point_y)
    vector = [point_x - center_x, point_y - center_y]
    distance = norm(vector)
    unit = vector / distance
    perpendicular = [-unit[2], unit[1]]
    ratio = radius / distance
    [center_x, center_y] + radius .* (
        ratio .* unit .+ sqrt(1.0 - ratio^2) .* perpendicular
    )
end

function sharpb2_curves()
    radius, nose_radius, length = 0.27, 0.0415, 1.81
    tangent = circle_tangent_point(nose_radius, 0.0, nose_radius, length, radius)
    theta = collect(range(0.0, asin(tangent[2] / nose_radius), length=400))
    nose = hcat(
        nose_radius .* (1.0 .- cos.(theta)), nose_radius .* sin.(theta)
    )
    cone_x = collect(range(tangent[1], length, length=400))
    cone_y = tangent[2] .+ (radius - tangent[2]) .* (
        cone_x .- tangent[1]
    ) ./ (length - tangent[1])
    lower = vcat(nose, hcat(cone_x[2:end], cone_y[2:end]))

    outer_radius = 0.17
    theta = collect(range(pi, pi / 1.75, length=400))
    outer_nose = hcat(
        0.14 .+ outer_radius .* cos.(theta), outer_radius .* sin.(theta)
    )
    outer_x = collect(range(outer_nose[end, 1], length, length=400))
    outer_y = outer_nose[end, 2] .+ (2.0radius - outer_nose[end, 2]) .* (
        outer_x .- outer_nose[end, 1]
    ) ./ (length - outer_nose[end, 1])
    upper = vcat(outer_nose, hcat(outer_x, outer_y))
    return lower, upper, tangent
end

function make_sharpb2_mesh(porder; radial_shift=1.0e-4)
    lower, upper, tangent = sharpb2_curves()
    lower_first = lower[lower[:, 1] .<= tangent[1], :]
    lower_second = vcat(
        lower_first[end:end, :],
        lower[(lower[:, 1] .> tangent[1]) .& (lower[:, 1] .<= 1.25), :],
    )
    upper_first = upper[upper[:, 1] .<= 0.0, :]
    upper_second = vcat(
        upper_first[end:end, :],
        upper[(upper[:, 1] .> 0.0) .& (upper[:, 1] .<= 1.22), :],
    )

    first = surfmesh2d(
        lower_first, upper_first, 36, 100, porder;
        streamwise_scaling=(2.0, 1.2), normal_scaling=(5.5, 0.0),
    )
    second = surfmesh2d(
        lower_second, upper_second, 120, 100, porder;
        streamwise_scaling=(3.5, 1.5), normal_scaling=(5.5, 0.0),
    )
    points, elements, dgnodes = connect_blocks(first, second)
    points[2, :] .+= radial_shift
    dgnodes[:, 2, :] .+= radial_shift

    minimum_x, minimum_y = minimum(points[1, :]), minimum(points[2, :])
    maximum_y_index = argmax(points[2, :])
    x_at_maximum_y, maximum_y = points[1, maximum_y_index], points[2, maximum_y_index]
    maximum_x_index = argmax(points[1, :])
    y_at_maximum_x = points[2, maximum_x_index]
    transition_x, transition_y = 0.0, 0.04
    boundaryexpr = [
        p -> abs.(p[2, :] .- minimum_y) .< 1.0e-6,
        p -> p[2, :] .> minimum_y .+ (transition_y - minimum_y) /
             (transition_x - minimum_x) .* (p[1, :] .- minimum_x) .- 1.0e-4,
        p -> p[2, :] .> transition_y .+ (maximum_y - transition_y) /
             (x_at_maximum_y - transition_x) .* (p[1, :] .- transition_x) .- 1.0e-4,
        p -> p[2, :] .< y_at_maximum_x + 1.0e-4,
        p -> abs.(p[1, :]) .< 20.0 + 1.0e-6,
    ]
    faces, _, _ = Exasim.Preprocessing.facenumbering(points, elements, 1, boundaryexpr, [])
    _, mesh = Exasim.initializeexasim()
    mesh.p = points
    mesh.t = elements
    mesh.f = faces
    mesh.dgnodes = dgnodes
    mesh.boundaryexpr = boundaryexpr
    mesh.boundarycondition = reshape([5, 1, 1, 3, 2], :, 1)
    mesh.periodicexpr = []
    mesh.curvedboundary = zeros(Int, 0, 0)
    mesh.curvedboundaryexpr = []
    mesh.periodicboundary = zeros(Int, 0, 0)
    mesh
end
