"""Julia axisymmetric Poisson solve on the tracked Orion geometry."""

repository = dirname(dirname(dirname(@__DIR__)))
pushfirst!(LOAD_PATH, joinpath(repository, "frontends", "Julia", "Exasim"))
using Exasim
include(joinpath(@__DIR__, "pdemodel.jl"))

function build_case(run_directory=joinpath(@__DIR__, "julia_run"))
    pde, mesh = Exasim.initializeexasim()
    pde.model = "ModelD"
    pde.platform, pde.mpiprocs, pde.hybrid = "cpu", 4, 1
    pde.porder, pde.pgauss = 3, 6
    pde.physicsparam, pde.tau = [1.0 0.0 1.0 1.0], [1.0]
    pde.linearsolvertol, pde.ppdegree, pde.RBdim = 1.0e-8, 1, 0
    mkpath(run_directory)
    pde.datapath, pde.builddir = run_directory, joinpath(run_directory, ".exasim")

    geometry = joinpath(run_directory, "domain6.geo")
    cp(joinpath(@__DIR__, "domain6.geo"), geometry; force=true)
    mesh.p, mesh.t = Exasim.Mesh.gmshcall(pde, splitext(geometry)[1], 2, 1)
    xmin, xmax = extrema(mesh.p[1, :])
    ymin = minimum(mesh.p[2, :])
    mesh.boundaryexpr = [
        p -> abs.(p[2, :] .- ymin) .< 1.0e-6,
        p -> abs.(p[1, :] .- xmax) .< 1.0e-6,
        p -> p[1, :] .< xmin + 1.0e-6,
        p -> abs.(p[1, :]) .< xmax + 1.0,
    ]
    mesh.boundarycondition = reshape([1, 4, 3, 2], :, 1)
    return pde, mesh
end

function main()
    pde, mesh = build_case()
    solution, pde, mesh = Exasim.exasim(pde, mesh)[1:3]
    println("Julia Orion solution: ", size(solution))
    return solution, pde, mesh
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
