"""Julia 3D ISOQ-shell Poisson example."""

repository = dirname(dirname(dirname(@__DIR__)))
pushfirst!(LOAD_PATH, joinpath(repository, "frontends", "Julia", "Exasim"))
using Exasim
include(joinpath(@__DIR__, "pdemodel.jl"))

function build_case(run_directory=joinpath(@__DIR__, "julia_run"))
    pde, mesh = Exasim.initializeexasim()
    pde.model = "ModelD"; pde.platform = "cpu"; pde.mpiprocs = 4; pde.hybrid = 1
    pde.porder = 2; pde.pgauss = 4; pde.physicsparam = [1.0 0.0 1.0]; pde.tau = [2.0]
    pde.linearsolveriter = 151; pde.GMRESrestart = 150; pde.NLtol = 1.0e-4
    pde.linearsolvertol = 1.0e-6; pde.GMRESortho = 1; pde.RBdim = 0
    pde.NLiter = 2; pde.ppdegree = 20
    mkpath(run_directory); pde.datapath = run_directory; pde.builddir = joinpath(run_directory, ".exasim")
    geometry = joinpath(run_directory, "isoq3d.geo")
    cp(joinpath(@__DIR__, "isoq3d.geo"), geometry; force=true)
    mesh.p, mesh.t = Exasim.Mesh.gmshcall(pde, splitext(geometry)[1], 3, 0)
    maximum_radius = maximum(hypot.(mesh.p[2, :], mesh.p[3, :]))
    mesh.boundaryexpr = [
        p -> abs.(p[2, :]) .< 1.0e-6,
        p -> abs.(p[3, :]) .< 1.0e-6,
        p -> abs.(p[1, :] .- 1.0) .< 1.0e-6,
        p -> abs.(hypot.(p[2, :], p[3, :]) .- 0.05) .< 1.0e-5,
        p -> abs.(hypot.(p[2, :], p[3, :]) .- maximum_radius) .< 1.0e-5,
    ]
    mesh.boundarycondition = reshape([3, 3, 3, 1, 2], :, 1)
    return pde, mesh
end

function main()
    pde, mesh = build_case(); solution, pde, mesh = Exasim.exasim(pde, mesh)[1:3]
    println("Julia ISOQ3D solution: ", size(solution)); return solution, pde, mesh
end
if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
