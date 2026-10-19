using Exasim
using LinearAlgebra
using NCDatasets

include("pdemodel.jl")

function main()
pde, mesh = Exasim.initializeexasim()
pde.model = "ModelD"
pde.modelfile = "pdemodel"
pde.porder = 2
pde.torder = 2
pde.nstage = 2
pde.dt = fill(0.05, 200)
pde.soltime = collect(0:199)
pde.visdt = 0.05
pde.pgauss = 2 * pde.porder
pde.tau = [1.0]
pde.linearsolvertol = 1.0e-4
pde.ppdegree = 6
pde.RBdim = 0
pde.GMRESrestart = 50
pde.linearsolveriter = 51
pde.precMatrixType = 2
pde.NLiter = 2

ion_mass = 16 * 1.67e-27
kB = 1.38e-23
pde.physicsparam = [ion_mass kB 1.6e-19 8.7]
pde.mpiprocs = 1
pde.hybrid = 1

earth_radius = 6378.0
inner_radius = (earth_radius + 100.0) / earth_radius
outer_radius = (earth_radius + 500.0) / earth_radius
mesh.p, mesh.t, mesh.dgnodes = cube_sphere(pde.porder, inner_radius, outer_radius, 15, 10)
mesh.boundaryexpr = [
    p -> abs.(sum(p .* p, dims=1) .- inner_radius^2) .< 1.0e-6,
    p -> abs.(sum(p .* p, dims=1) .- outer_radius^2) .< 1.0e-6,
]
mesh.boundarycondition = reshape([1, 2], :, 1)

npe, _, ne = size(mesh.dgnodes)
nvalues = npe * ne
input_file = joinpath(@__DIR__, "input.nc")
isfile(input_file) || error("ionTest requires $input_file, which is not distributed with the repository.")
dataset = NCDataset(input_file)
try
    mesh.vdg = zeros(npe, 14, ne)
    mesh.udg = zeros(npe, 5, ne)
    variable_names = ["EX", "EY", "EZ", "BX", "BY", "BZ", "U", "V", "W", "NU", "P", "L", "H", "C"]
    for (component, name) in enumerate(variable_names)
        mesh.vdg[:, component, :] = reshape(first_record(dataset[name], nvalues), npe, ne)
    end
    number_density = first_record(dataset["N"], nvalues)
    mesh.udg[:, 1, :] = reshape(number_density, npe, ne)
    mesh.udg[:, 2, :] = reshape(ion_mass .* number_density .* first_record(dataset["VX"], nvalues), npe, ne)
    mesh.udg[:, 3, :] = reshape(ion_mass .* number_density .* first_record(dataset["VY"], nvalues), npe, ne)
    mesh.udg[:, 4, :] = reshape(ion_mass .* number_density .* first_record(dataset["VZ"], nvalues), npe, ne)
    mesh.udg[:, 5, :] = reshape(kB .* number_density .* first_record(dataset["T"], nvalues), npe, ne)
finally
    close(dataset)
end

sol, pde, mesh = Exasim.exasim(pde, mesh)[1:3]
println("Done!")
end

function first_record(variable, count)
    values = Array(variable)
    length(values) == count && return vec(values)
    point_dimension = findfirst(==(count), size(values))
    point_dimension === nothing && error("NetCDF variable has no dimension matching the $count DG points.")
    indices = ntuple(dimension -> dimension == point_dimension ? Colon() : 1, ndims(values))
    vec(values[indices...])
end

function cube_sphere(order, inner_radius, outer_radius, tangential_elements, radial_elements)
    cube_points, cube_elements = Exasim.Mesh.cubemesh(tangential_elements, tangential_elements, radial_elements, 1)
    cube_points[3, :] = logarithmic_increment(cube_points[3, :], 3.0)
    cube_points[1:2, :] .= -pi / 4 .+ pi .* cube_points[1:2, :] ./ 2
    cube_points[3, :] .= inner_radius .+ (outer_radius - inner_radius) .* cube_points[3, :]

    cube_dgnodes = Exasim.Preprocessing.createdgnodes(cube_points, cube_elements, zeros(Int, 0, 0), Int[], [], order)
    npe, dimensions, elements = size(cube_dgnodes)
    flat_dgnodes = reshape(permutedims(cube_dgnodes, (2, 1, 3)), dimensions, npe * elements)
    points_per_patch, elements_per_patch = size(cube_points, 2), size(cube_elements, 2)
    dgnodes_per_patch = npe * elements
    points = zeros(3, 6 * points_per_patch)
    connectivity = zeros(Int, 8, 6 * elements_per_patch)
    dgnodes = zeros(3, 6 * dgnodes_per_patch)

    patch_points = cube_projection(cube_points)
    patch_dgnodes = cube_projection(flat_dgnodes)
    rotation_z = rotation_matrix([0.0, 0.0, 1.0])
    for patch = 1:4
        point_range = (patch-1)*points_per_patch+1:patch*points_per_patch
        element_range = (patch-1)*elements_per_patch+1:patch*elements_per_patch
        dgnode_range = (patch-1)*dgnodes_per_patch+1:patch*dgnodes_per_patch
        points[:, point_range] = patch_points
        dgnodes[:, dgnode_range] = patch_dgnodes
        connectivity[:, element_range] = cube_elements .+ (patch-1)*points_per_patch
        patch_points = rotation_z * patch_points
        patch_dgnodes = rotation_z * patch_dgnodes
    end
    rotation_y = rotation_matrix([0.0, 1.0, 0.0])
    for (patch, rotation) in ((5, transpose(rotation_y)), (6, rotation_y))
        point_range = (patch-1)*points_per_patch+1:patch*points_per_patch
        element_range = (patch-1)*elements_per_patch+1:patch*elements_per_patch
        dgnode_range = (patch-1)*dgnodes_per_patch+1:patch*dgnodes_per_patch
        points[:, point_range] = rotation * cube_projection(cube_points)
        dgnodes[:, dgnode_range] = rotation * cube_projection(flat_dgnodes)
        connectivity[:, element_range] = cube_elements .+ (patch-1)*points_per_patch
    end
    points, connectivity = merge_duplicate_nodes(points, connectivity)
    dgnodes = permutedims(reshape(dgnodes, 3, npe, 6 * elements), (2, 1, 3))
    points, connectivity, dgnodes
end

function rotation_matrix(axis)
    skew = [0.0 -axis[3] axis[2]; axis[3] 0.0 -axis[1]; -axis[2] axis[1] 0.0]
    Matrix{Float64}(I, 3, 3) + skew + skew * skew
end

function cube_projection(points)
    tangent_theta = tan.(points[1, :])
    tangent_phi = tan.(points[2, :])
    radial_x = points[3, :] ./ sqrt.(1 .+ tangent_theta.^2 .+ tangent_phi.^2)
    [transpose(radial_x); transpose(radial_x .* tangent_theta); transpose(radial_x .* tangent_phi)]
end

function logarithmic_increment(values, alpha)
    lower, upper = extrema(values)
    lower .+ (upper-lower) .* (exp.(alpha .* (values .- lower) ./ (upper-lower)) .- 1) ./ (exp(alpha)-1)
end

function merge_duplicate_nodes(points, connectivity)
    snap = 100 * maximum(maximum(points, dims=2) .- minimum(points, dims=2)) * 1024 * eps(Float64)
    node_map = Dict{NTuple{3,Int},Int}()
    merged_points = Vector{Vector{Float64}}()
    old_to_new = zeros(Int, size(points, 2))
    for node = axes(points, 2)
        key = Tuple(round.(Int, points[:, node] ./ snap))
        index = get(node_map, key, 0)
        if index == 0
            push!(merged_points, collect(points[:, node]))
            index = length(merged_points)
            node_map[key] = index
        end
        old_to_new[node] = index
    end
    hcat(merged_points...), old_to_new[connectivity]
end

main()
