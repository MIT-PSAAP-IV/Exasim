"""Julia version of the axisymmetric ideal-gas ISOQ mesh-adaptivity case."""

repository = dirname(dirname(dirname(@__DIR__)))
source_install = joinpath(dirname(repository), "exasim_install")
if !haskey(ENV, "EXASIM_PREFIX") &&
   isfile(joinpath(source_install, "lib", "cmake", "Exasim", "ExasimConfig.cmake"))
    ENV["EXASIM_PREFIX"] = source_install
end
pushfirst!(LOAD_PATH, joinpath(repository, "frontends", "Julia", "Exasim"))
using DelimitedFiles
using Exasim

include(joinpath(@__DIR__, "pdemodel.jl"))
include(joinpath(@__DIR__, "isoq_mesh.jl"))
include(joinpath(@__DIR__, "postprocess_surfacequantities.jl"))

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
    pde.saveParaview = 1
    pde.saveSolBouFreq = 1
    pde.ibs = 3
    pde.saveSolBouLoc = 1
    mkpath(run_directory)
    pde.datapath = run_directory
    pde.builddir = joinpath(run_directory, ".exasim")
    pde.buildpath = pde.builddir

    gam, reynolds, prandtl, mach_inf = 1.4, 1.56e5, 0.71, 7.6
    reference_temperature, wall_temperature = 266.5, 300.0
    pressure_inf = 1.0 / (gam * mach_inf^2)
    temperature_inf = pressure_inf / (gam - 1.0)
    density_inf, momentum_x_inf, momentum_y_inf = 1.0, 1.0, 0.0
    energy_inf = 0.5 + pressure_inf / (gam - 1.0)

    pde.AV = 2
    pde.AVcontinuationIter = 6
    pde.AVcontinuationLogScale = 1.5
    pde.AVcoeffStart = 0.002
    pde.AVcoeffEnd = 0.00001
    pde.AVdistfunction = 1
    pde.distanceboundaryconditions = [3]
    pde.AVsmoothingMethod = 1
    pde.AVHelmholtzCoeff = 0.001
    av_max_divergence, av_distance_coefficient = 20.0, 100.0

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

    pde.physicsparam = reshape(
        [
            gam, reynolds, prandtl, mach_inf,
            density_inf, momentum_x_inf, momentum_y_inf, energy_inf,
            temperature_inf, reference_temperature, wall_temperature,
            av_max_divergence, av_distance_coefficient,
            pde.AVcoeffStart, pde.AVcoeffEnd,
        ],
        1, :,
    )
    pde.tau = [4.0]
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
    distance = wall_distance(mesh, pde.porder)
    # ODG layout: wall distance, filtered AV sensor, mesh-adaptation pressure.
    mesh.odg = zeros(Float64, size(distance, 1), 3, size(distance, 3))
    mesh.odg[:, 1:1, :] .= distance
    npe, _, ne = size(mesh.dgnodes)
    mesh.udg = zeros(Float64, npe, 4, ne)
    mesh.udg[:, 1, :] .= density_inf
    mesh.udg[:, 2, :] .= momentum_x_inf .* tanh.(
        av_distance_coefficient .* distance[:, 1, :]
    )
    mesh.udg[:, 3, :] .= momentum_y_inf
    near_wall_temperature = temperature_inf .* (
        wall_temperature / reference_temperature - 1.0
    ) .* exp.(-av_distance_coefficient .* distance[:, 1, :]) .+ temperature_inf
    mesh.udg[:, 4, :] .= near_wall_temperature .+ 0.5 .* (
        mesh.udg[:, 2, :].^2 .+ mesh.udg[:, 3, :].^2
    )
    return pde, mesh
end

function main()
    pde, mesh = build_case()
    solution, pde, mesh, master, dmd, _, _ = Exasim.exasim(pde, mesh)
    println(
        "Julia ISOQ mesh adaptivity: ", size(solution),
        " output = ", joinpath(pde.datapath, "dataout"),
    )
    postprocess_surfacequantities(pde)
    return solution, pde, mesh, master, dmd
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
