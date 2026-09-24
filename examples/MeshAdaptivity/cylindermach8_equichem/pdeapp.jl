"""Julia version of the equilibrium-air Mach-8 mesh-adaptivity example."""

using DelimitedFiles

repository = dirname(dirname(dirname(@__DIR__)))
source_install = joinpath(dirname(repository), "exasim_install")
if !haskey(ENV, "EXASIM_PREFIX") &&
   isfile(joinpath(source_install, "lib", "cmake", "Exasim", "ExasimConfig.cmake"))
    ENV["EXASIM_PREFIX"] = source_install
end
pushfirst!(LOAD_PATH, joinpath(repository, "frontends", "Julia", "Exasim"))
using Exasim

include(joinpath(@__DIR__, "pdemodel.jl"))
include(joinpath(dirname(@__DIR__), "cylindermach8", "cylinder_mesh.jl"))

function read_material_database(filename)
    header = parse.(Int, split(strip(open(readline, filename))))
    if length(header) != 5 || header[1] != 2 || header[2] < 15
        error("Unexpected equilibrium-air database header in $filename")
    end
    rows = readdlm(filename, Float64; skipstart=1)
    if size(rows, 1) != prod(header[3:5])
        error("Material database row count does not match its header")
    end
    xi = unique(rows[:, 1])
    energy = unique(rows[:, 2])
    properties = zeros(Float64, length(xi), length(energy), header[2])
    xi_index = Dict(value => index for (index, value) in enumerate(xi))
    energy_index = Dict(value => index for (index, value) in enumerate(energy))
    for row in axes(rows, 1)
        properties[xi_index[rows[row, 1]], energy_index[rows[row, 2]], :] .=
            rows[row, 3:end]
    end
    return (xi=xi, energy=energy, properties=properties)
end

function interpolate_xi(database, xi)
    upper = searchsortedlast(database.xi, xi)
    lower = min(max(upper, 1), length(database.xi) - 1)
    weight = (xi - database.xi[lower]) /
             (database.xi[lower + 1] - database.xi[lower])
    return (1.0 - weight) .* database.properties[lower, :, :] .+
           weight .* database.properties[lower + 1, :, :]
end

function interpolate_energy(temperature_grid, energy_grid, temperature)
    result = similar(temperature)
    for index in eachindex(temperature)
        upper = searchsortedlast(temperature_grid, temperature[index])
        lower = min(max(upper, 1), length(temperature_grid) - 1)
        weight = (temperature[index] - temperature_grid[lower]) /
                 (temperature_grid[lower + 1] - temperature_grid[lower])
        result[index] = (1.0 - weight) * energy_grid[lower] +
                        weight * energy_grid[lower + 1]
    end
    return result
end

function initial_solution(mesh, distance, database, xi_inf, temperature_inf,
                          wall_temperature, energy_ref, velocity_x, velocity_y)
    temperature_grid = vec(interpolate_xi(database, xi_inf)[:, 2])
    target_temperature = temperature_inf .+
        (wall_temperature - temperature_inf) .* exp.(-10.0 .* distance[:, 1, :])
    if minimum(target_temperature) < minimum(temperature_grid) ||
       maximum(target_temperature) > maximum(temperature_grid)
        error("Initial temperature is outside the material database")
    end
    internal_energy = interpolate_energy(
        temperature_grid, database.energy, target_temperature
    ) ./ energy_ref
    density = ones(size(target_temperature))
    ux = velocity_x .* tanh.(10.0 .* distance[:, 1, :])
    uy = velocity_y .* tanh.(10.0 .* distance[:, 1, :])
    udg = zeros(Float64, size(distance, 1), 4, size(distance, 3))
    udg[:, 1, :] .= density
    udg[:, 2, :] .= density .* ux
    udg[:, 3, :] .= density .* uy
    udg[:, 4, :] .= density .* (internal_energy .+ 0.5 .* (ux .* ux .+ uy .* uy))
    return udg
end

function build_case(run_directory=joinpath(@__DIR__, "julia_run"))
    pde, _ = Exasim.initializeexasim()
    pde.model = "ModelD"
    pde.modelfile = ""
    pde.platform = "cpu"
    pde.mpiprocs = 4
    pde.hybrid = 1
    pde.ncw = 15
    pde.porder = 2
    pde.pgauss = 2 * pde.porder
    pde.tau = [1.0]
    pde.GMRESrestart = 200
    pde.linearsolvertol = 1.0e-7
    pde.linearsolveriter = 200
    pde.RBdim = 0
    pde.ppdegree = 20
    pde.NLtol = 1.0e-6
    pde.NLiter = 30
    pde.matvectol = 1.0e-6
    pde.flag = reshape(Int[], 1, 0)
    pde.problem = reshape(Int[], 1, 0)
    pde.factor = reshape(Float64[], 1, 0)
    pde.solversparam = reshape(Float64[], 1, 0)
    pde.stgdata = zeros(Float64, 0, 0)
    pde.stgparam = zeros(Float64, 0, 0)
    pde.stgib = zeros(Int, 0, 0)
    pde.dae_dt = Float64[]

    mkpath(run_directory)
    pde.datapath = run_directory
    pde.builddir = joinpath(run_directory, ".exasim")
    pde.buildpath = pde.builddir
    database_file = joinpath(
        repository, "apps", "materialdatabases", "equilibriumAir5logdensityexasim.dat"
    )
    pde.materialdatabase = database_file
    database = read_material_database(database_file)

    mach_inf = 8.03
    temperature_inf = 265.0
    wall_temperature = 300.0
    length_ref = 1.0
    chemistry_conductivity = 0.0
    outlet_relaxation = 0.0
    energy_inf_dimensional = -1.097328937016845e5
    pressure_inf_dimensional = 97.467348012325772
    density_inf_dimensional = 0.001276095150944
    sound_speed_inf = 3.232136939449924e2
    velocity_inf_dimensional = mach_inf * sound_speed_inf
    xi_inf = log(density_inf_dimensional)
    density_ref = density_inf_dimensional
    velocity_ref = velocity_inf_dimensional
    pressure_ref = density_ref * velocity_ref^2
    energy_ref = velocity_ref^2

    pde.AV = 1
    pde.AVcontinuationIter = 10
    pde.AVcontinuationLogScale = 1.5
    pde.AVcoeffStart = 0.060
    pde.AVcoeffEnd = 0.008
    pde.AVdistfunction = 1
    pde.distanceboundaryconditions = [3]
    pde.AVsmoothingMethod = 1
    pde.AVHelmholtzCoeff = 0.025

    pde.meshadaptenabled = 1
    pde.meshadaptfield = 1
    pde.meshadaptalpha = 0.5
    pde.meshadaptHelmholtzCoeff = 5.0e-2
    pde.meshadaptforcescale = 0.15
    pde.meshadaptsmoothingpasses = 30
    pde.meshadaptboundaryconditions = [2, 3, 3]

    energy_inf = energy_inf_dimensional / energy_ref
    pde.externalparam = reshape([1.0, 1.0, 0.0, energy_inf + 0.5], 1, :)
    pde.physicsparam = reshape(
        [
            density_ref,
            velocity_ref,
            pressure_ref,
            energy_ref,
            length_ref,
            1.0,
            chemistry_conductivity,
            wall_temperature,
            pressure_inf_dimensional,
            outlet_relaxation,
            2.0,
            30.0,
            pde.AVcoeffStart,
            pde.AVcoeffEnd,
        ],
        1,
        :,
    )

    mesh = make_cylinder_mesh(
        pde.porder; nx=51, ny=32, radial_decay=3.0, outer_offset=4.0
    )
    mesh.boundarycondition = reshape([3, 2, 1], :, 1)
    distance = wall_distance(mesh, pde.porder)
    mesh.odg = zeros(Float64, size(distance, 1), 2, size(distance, 3))
    mesh.odg[:, 1:1, :] .= distance
    mesh.udg = initial_solution(
        mesh,
        distance,
        database,
        xi_inf,
        temperature_inf,
        wall_temperature,
        energy_ref,
        1.0,
        0.0,
    )
    return pde, mesh
end

function main()
    pde, mesh = build_case()
    solution, pde, mesh, master, dmd, _, _, _ = Exasim.exasim(pde, mesh)
    println(
        "Julia equilibrium-air mesh adaptivity: ", size(solution),
        " output = ", joinpath(pde.datapath, "dataout"),
    )
    return solution, pde, mesh, master, dmd
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
