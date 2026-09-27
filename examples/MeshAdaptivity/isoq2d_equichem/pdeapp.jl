"""Julia version of the axisymmetric equilibrium-air ISOQ case."""

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
include(joinpath(dirname(@__DIR__), "isoq2d_idealgas", "isoq_mesh.jl"))

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

function interpolate_database(database, xi_value, energy_value)
    i = min(max(searchsortedlast(database.xi, xi_value), 1), length(database.xi) - 1)
    j = min(
        max(searchsortedlast(database.energy, energy_value), 1),
        length(database.energy) - 1,
    )
    tx = (xi_value - database.xi[i]) / (database.xi[i + 1] - database.xi[i])
    te = (energy_value - database.energy[j]) /
         (database.energy[j + 1] - database.energy[j])
    return (1.0 - tx) * (1.0 - te) .* database.properties[i, j, :] .+
           tx * (1.0 - te) .* database.properties[i + 1, j, :] .+
           (1.0 - tx) * te .* database.properties[i, j + 1, :] .+
           tx * te .* database.properties[i + 1, j + 1, :]
end

function interpolate_energy(temperature_grid, energy_grid, temperature)
    result = similar(temperature)
    for index in eachindex(temperature)
        lower = min(
            max(searchsortedlast(temperature_grid, temperature[index]), 1),
            length(temperature_grid) - 1,
        )
        weight = (temperature[index] - temperature_grid[lower]) /
                 (temperature_grid[lower + 1] - temperature_grid[lower])
        result[index] = (1.0 - weight) * energy_grid[lower] +
                        weight * energy_grid[lower + 1]
    end
    return result
end

function energy_for_temperature(database, xi_value, temperature)
    table_temperature = [
        interpolate_database(database, xi_value, value)[2]
        for value in database.energy
    ]
    values = temperature isa Number ? [temperature] : collect(temperature)
    if minimum(values) < minimum(table_temperature) ||
       maximum(values) > maximum(table_temperature)
        error("Initial temperature is outside the material database")
    end
    result = interpolate_energy(table_temperature, database.energy, values)
    return temperature isa Number ? result[1] : reshape(result, size(temperature))
end

function initial_solution(mesh, distance, database, xi_inf, temperature_inf,
                          wall_temperature, energy_ref, velocity_z, velocity_r,
                          slope)
    target_temperature = temperature_inf .+
        (wall_temperature - temperature_inf) .* exp.(-slope .* distance[:, 1, :])
    internal_energy =
        energy_for_temperature(database, xi_inf, target_temperature) ./ energy_ref
    uz = velocity_z .* tanh.(slope .* distance[:, 1, :])
    ur = velocity_r .* tanh.(slope .* distance[:, 1, :])
    udg = zeros(Float64, size(distance, 1), 4, size(distance, 3))
    udg[:, 1, :] .= 1.0
    udg[:, 2, :] .= uz
    udg[:, 3, :] .= ur
    udg[:, 4, :] .= internal_energy .+ 0.5 .* (uz .* uz .+ ur .* ur)
    return udg
end

function build_case(run_directory=joinpath(@__DIR__, "julia_run"))
    pde, _ = Exasim.initializeexasim()
    pde.model = "ModelD"
    pde.modelfile = ""
    pde.platform = "cpu"
    pde.mpiprocs = 4
    pde.porder = 2
    pde.pgauss = 2 * pde.porder
    pde.hybrid = 1
    pde.debugmode = 0
    pde.nd = 2
    pde.ncw = 15
    pde.saveParaview = 1
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

    mach_inf = 7.6
    reynolds = 1.56e5
    temperature_inf = 266.5
    wall_temperature = 300.0
    length_ref = 1.0
    chemistry_conductivity = 0.0
    outlet_relaxation = 0.0

    density_inf = 1.0e-3
    for _ in 1:12
        xi_inf = log(density_inf)
        energy_inf = energy_for_temperature(database, xi_inf, temperature_inf)
        properties_inf = interpolate_database(database, xi_inf, energy_inf)
        velocity_inf = mach_inf * properties_inf[6]
        density_new = reynolds * properties_inf[3] / (velocity_inf * length_ref)
        if abs(density_new - density_inf) <= 1.0e-12 * max(1.0, density_inf)
            density_inf = density_new
            break
        end
        density_inf = density_new
    end

    xi_inf = log(density_inf)
    energy_inf = energy_for_temperature(database, xi_inf, temperature_inf)
    properties_inf = interpolate_database(database, xi_inf, energy_inf)
    pressure_inf = properties_inf[1]
    sound_speed_inf = properties_inf[6]
    velocity_inf = mach_inf * sound_speed_inf

    density_ref = density_inf
    velocity_ref = velocity_inf
    pressure_ref = density_ref * velocity_ref^2
    energy_ref = velocity_ref^2
    transport_factor = 1.0

    pde.AV = 1
    pde.AVcontinuationIter = 10
    pde.AVcontinuationLogScale = 1.5
    pde.AVcoeffStart = 0.003
    pde.AVcoeffEnd = 0.000006
    pde.AVdistfunction = 1
    pde.distanceboundaryconditions = [3]
    pde.AVsmoothingMethod = 1
    pde.AVHelmholtzCoeff = 0.001
    av_max_divergence = 60.0
    av_distance_coefficient = 100.0

    pde.meshadaptenabled = 1
    pde.meshadaptfield = 2
    pde.meshadaptavcomponent = 1
    pde.meshadaptalpha = 0.5
    pde.meshadaptqmin = 0.2
    pde.meshadaptqmax = 0.8
    pde.meshadaptHelmholtzCoeff = 0.001
    pde.meshadaptforcescale = 0.25
    pde.meshadaptsmoothingpasses = 30
    pde.meshadaptboundaryconditions = [3, 3, 3, 2]

    specific_energy_inf = energy_inf / energy_ref
    pde.externalparam = reshape([1.0, 1.0, 0.0, specific_energy_inf + 0.5], 1, :)
    pde.physicsparam = reshape(
        [
            density_ref, velocity_ref, pressure_ref, energy_ref, length_ref,
            transport_factor, chemistry_conductivity, wall_temperature,
            pressure_inf, outlet_relaxation, av_max_divergence,
            av_distance_coefficient, pde.AVcoeffStart, pde.AVcoeffEnd,
        ],
        1, :,
    )

    pde.tau = [10.0]
    pde.GMRESrestart = 250
    pde.GMRESortho = 1
    pde.linearsolvertol = 1.0e-6
    pde.linearsolveriter = 500
    pde.preconditioner = 1
    pde.RBdim = 0
    pde.ppdegree = 0
    pde.NLtol = 1.0e-6
    pde.NLiter = 10
    pde.matvectol = 1.0e-6

    mesh = make_isoq_mesh(pde.porder; radial_shift=5.0e-4)
    mesh.boundarycondition = reshape([4, 2, 1, 3], :, 1)
    distance = wall_distance(mesh, pde.porder)
    mesh.odg = zeros(Float64, size(distance, 1), 2, size(distance, 3))
    mesh.odg[:, 1:1, :] .= distance
    mesh.udg = initial_solution(
        mesh, distance, database, xi_inf, temperature_inf, wall_temperature,
        energy_ref, 1.0, 0.0, av_distance_coefficient,
    )

    println("\nEquilibrium-air ISOQ setup")
    println(
        "  rho_inf = $(density_inf) kg/m^3, p_inf = $(pressure_inf) Pa, " *
        "T_inf = $(temperature_inf) K"
    )
    println(
        "  a_inf = $(sound_speed_inf) m/s, U_inf = $(velocity_inf) m/s, " *
        "Mach = $(mach_inf), Re = $(reynolds)"
    )
    return pde, mesh
end

function main()
    pde, mesh = build_case()
    solution, pde, mesh, master, dmd, _, _, _ = Exasim.exasim(pde, mesh)
    println(
        "Julia equilibrium-air ISOQ mesh adaptivity: ", size(solution),
        " output = ", joinpath(pde.datapath, "dataout"),
    )
    return solution, pde, mesh, master, dmd
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
