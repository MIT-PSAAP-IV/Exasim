"""Julia frontend for the Mach-5.38 rounded-nose flat-plate example."""

repository = dirname(dirname(dirname(@__DIR__)))
source_install = joinpath(dirname(repository), "exasim_install")
if !haskey(ENV, "EXASIM_PREFIX") && isfile(joinpath(source_install, "lib", "cmake", "Exasim", "ExasimConfig.cmake"))
    ENV["EXASIM_PREFIX"] = source_install
end
pushfirst!(LOAD_PATH, joinpath(repository, "frontends", "Julia", "Exasim"))
using Exasim

include(joinpath(@__DIR__, "pdemodel.jl"))
include(joinpath(@__DIR__, "auplate_mesh.jl"))

function build_case()
    pde, _ = Exasim.initializeexasim()
    pde.model = "ModelD"
    pde.modelfile = ""
    pde.platform = "cpu"
    pde.mpiprocs = 8
    pde.hybrid = 1
    pde.porder = 2

    run_directory = joinpath(@__DIR__, "julia_run")
    pde.datapath = run_directory
    pde.builddir = joinpath(run_directory, ".exasim")
    pde.buildpath = pde.builddir
    # Match the MATLAB/Python empty optional tails rather than serializing the
    # Julia frontend's legacy two-entry placeholders.
    pde.flag = reshape(Int[], 1, 0)
    pde.problem = reshape(Int[], 1, 0)
    pde.factor = reshape(Float64[], 1, 0)
    pde.solversparam = reshape(Float64[], 1, 0)
    pde.stgdata = zeros(Float64, 0, 0)
    pde.stgparam = zeros(Float64, 0, 0)
    pde.stgib = zeros(Int, 0, 0)
    pde.dae_dt = Float64[]

    gam = 1.4
    reynolds = 8.9e6
    prandtl = 0.71
    mach_inf = 5.38
    tref = 1020.0
    twall = 296.0
    pinf = 1.0 / (gam * mach_inf^2)
    tinf = pinf / (gam - 1.0)
    rinf = 1.0
    ruinf = 1.0
    rvinf = 0.0
    rEinf = 0.5 + pinf / (gam - 1.0)

    pde.tau = [10.0]
    pde.GMRESrestart = 250
    pde.linearsolvertol = 1.0e-6
    pde.linearsolveriter = 500
    pde.RBdim = 0
    pde.ppdegree = 0
    pde.NLtol = 1.0e-6
    pde.NLiter = 5
    pde.matvectol = 1.0e-6

    pde.AV = 1
    pde.AVcontinuationIter = 10
    pde.AVcontinuationLogScale = 2.0
    pde.AVcoeffStart = 0.02
    pde.AVcoeffEnd = 1.0e-5
    pde.AVdistfunction = 1
    pde.distanceboundaryconditions = [3]
    pde.AVsmoothingMethod = 1
    pde.AVHelmholtzCoeff = 1.0e-4

    av_max_divergence = 4.0
    av_distance_coefficient = 1.5e3
    pde.physicsparam = reshape(
        [
            gam,
            reynolds,
            prandtl,
            mach_inf,
            rinf,
            ruinf,
            rvinf,
            rEinf,
            tinf,
            tref,
            twall,
            av_max_divergence,
            av_distance_coefficient,
            pde.AVcoeffStart,
            pde.AVcoeffEnd,
        ],
        1,
        :,
    )

    mesh = make_auplate_mesh(pde.porder)
    distance = wall_distance(mesh)
    mesh.odg = zeros(Float64, size(distance, 1), 2, size(distance, 3))
    mesh.odg[:, 1:1, :] .= distance
    initial = zeros(Float64, size(mesh.dgnodes, 1), 4, size(mesh.dgnodes, 3))
    initial[:, 1, :] .= rinf
    initial[:, 2, :] .= ruinf .* tanh.(100.0 .* distance[:, 1, :])
    initial[:, 3, :] .= rvinf .* tanh.(100.0 .* distance[:, 1, :])
    near_wall_temperature = tinf .* (twall / tref - 1.0) .* exp.(-100.0 .* distance[:, 1, :]) .+ tinf
    initial[:, 4, :] .= near_wall_temperature .+ 0.5 .* (initial[:, 2, :].^2 .+ initial[:, 3, :].^2)
    mesh.udg = initial

    return pde, mesh
end

function main()
    pde, mesh = build_case()
    solution, pde, mesh, master, dmd, _, _ = Exasim.exasim(pde, mesh)
    density = solution[:, 1, :, end]
    println("Julia rounded-nose flat-plate solution: ", size(solution), " rho range = ", extrema(density))
    return solution, pde, mesh, master, dmd
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
