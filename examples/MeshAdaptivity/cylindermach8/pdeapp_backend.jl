"""Julia backend mesh-adaptivity counterpart of pdeapp_backend.m."""

repository = dirname(dirname(dirname(@__DIR__)))
source_install = joinpath(dirname(repository), "exasim_install")
if !haskey(ENV, "EXASIM_PREFIX") && isfile(joinpath(source_install, "lib", "cmake", "Exasim", "ExasimConfig.cmake"))
    ENV["EXASIM_PREFIX"] = source_install
end
pushfirst!(LOAD_PATH, joinpath(repository, "frontends", "Julia", "Exasim"))
using Exasim

include(joinpath(@__DIR__, "pdemodel.jl"))
include(joinpath(@__DIR__, "cylinder_mesh.jl"))

function build_backend_case()
    pde, _ = Exasim.initializeexasim()
    pde.model = "ModelD"
    pde.modelfile = ""
    pde.platform = "cpu"
    pde.mpiprocs = 1
    pde.hybrid = 1
    pde.porder = 2

    run_directory = joinpath(@__DIR__, "julia_backend_run")
    mkpath(run_directory)
    pde.datapath = run_directory
    pde.builddir = joinpath(run_directory, ".exasim")
    pde.buildpath = pde.builddir
    pde.flag = reshape(Int[], 1, 0)
    pde.problem = reshape(Int[], 1, 0)
    pde.factor = reshape(Float64[], 1, 0)
    pde.solversparam = reshape(Float64[], 1, 0)
    pde.stgdata = zeros(Float64, 0, 0)
    pde.stgparam = zeros(Float64, 0, 0)
    pde.stgib = zeros(Int, 0, 0)
    pde.dae_dt = Float64[]

    gam = 1.4
    reynolds = 1.835e5
    prandtl = 0.71
    mach_inf = 8.03
    tref = 265.0
    twall = 300.0
    pinf = 1.0 / (gam * mach_inf^2)
    tinf = pinf / (gam - 1.0)
    rinf = 1.0
    ruinf = 1.0
    rvinf = 0.0
    rEinf = 0.5 + pinf / (gam - 1.0)

    pde.tau = [1.0]
    pde.GMRESrestart = 200
    pde.linearsolvertol = 1.0e-6
    pde.linearsolveriter = 200
    pde.RBdim = 0
    pde.ppdegree = 0
    pde.NLtol = 1.0e-6
    pde.NLiter = 10
    pde.matvectol = 1.0e-6

    pde.AV = 1
    pde.AVcontinuationIter = 10
    pde.AVcontinuationLogScale = 1.5
    pde.AVcoeffStart = 0.060
    pde.AVcoeffEnd = 0.01
    pde.AVdistfunction = 1
    pde.distanceboundaryconditions = [3]
    pde.AVsmoothingMethod = 1
    pde.AVHelmholtzCoeff = 0.025

    av_max_divergence = 2.0
    av_distance_coefficient = 100.0
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

    pde.meshadaptenabled = 1
    pde.meshadaptfield = 3
    pde.meshadaptalpha = 0.5
    pde.meshadaptHelmholtzCoeff = 5.0e-2
    pde.meshadaptforcescale = 0.2
    pde.meshadaptsmoothingpasses = 30
    pde.meshadaptboundaryconditions = [2, 3, 3]

    mesh = make_cylinder_mesh(
        pde.porder; nx=51, ny=32, radial_decay=3.0, outer_offset=4.0
    )
    distance = wall_distance(mesh, pde.porder)
    mesh.odg = zeros(Float64, size(distance, 1), 2, size(distance, 3))
    mesh.odg[:, 1:1, :] .= distance
    mesh.udg = initialize_solution(mesh, distance, vec(pde.physicsparam))
    return pde, mesh
end

function main_backend()
    pde, mesh = build_backend_case()
    old_verification = get(ENV, "EXASIM_MESHADAPT_VERIFY", nothing)
    ENV["EXASIM_MESHADAPT_VERIFY"] = "0"
    try
        solution, pde, mesh, master, dmd, _, _ = Exasim.exasim(pde, mesh)
        adapted_file = joinpath(
            pde.datapath, "dataout", "out_meshadapt_xdg_np0.bin"
        )
        adapted_nodes = reshape(
            collect(reinterpret(Float64, read(adapted_file))), size(mesh.dgnodes)
        )
        mesh.dgnodes .= adapted_nodes
        println(
            "Julia backend mesh adaptivity: ", size(solution),
            " adapted x range = ", extrema(adapted_nodes[:, 1, :]),
        )
        return solution, pde, mesh, master, dmd
    finally
        if isnothing(old_verification)
            delete!(ENV, "EXASIM_MESHADAPT_VERIFY")
        else
            ENV["EXASIM_MESHADAPT_VERIFY"] = old_verification
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_backend()
end
