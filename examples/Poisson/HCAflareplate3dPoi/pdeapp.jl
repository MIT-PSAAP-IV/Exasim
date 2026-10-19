"""Julia Poisson solve on a periodic HCA flared-plate extrusion."""

repository = dirname(dirname(dirname(@__DIR__)))
pushfirst!(LOAD_PATH, joinpath(repository, "frontends", "Julia", "Exasim"))
using Exasim
include(joinpath(@__DIR__, "pdemodel.jl"))

function build_case(run_directory=joinpath(@__DIR__, "julia_run"))
    pde, mesh = Exasim.initializeexasim()
    pde.model = "ModelD"; pde.platform = "cpu"; pde.mpiprocs = 4; pde.hybrid = 1
    pde.porder = 2; pde.pgauss = 4; pde.physicsparam = [1.0 0.0 1.0 1.0]; pde.tau = [1.0]
    pde.linearsolvertol = 1.0e-8; pde.ppdegree = 0; pde.linearsolveriter = 400
    pde.GMRESrestart = 200; pde.preconditioner = 1; pde.RBdim = 0
    mkpath(run_directory); pde.datapath = run_directory; pde.builddir = joinpath(run_directory, ".exasim")
    geometry = joinpath(run_directory, "hca_flare.geo")
    cp(joinpath(@__DIR__, "hca_flare.geo"), geometry; force=true)
    mesh.p, mesh.t = Exasim.Mesh.gmshcall(pde, splitext(geometry)[1], 3, 1)
    xmin, xmax = extrema(mesh.p[1, :]); ymax = maximum(mesh.p[2, :])
    corner_x, lower_y = 0.04988, 0.10645
    lower_slope = (0.17000 - lower_y) / (xmax - corner_x)
    mesh.boundaryexpr = [
        p -> abs.(p[3, :]) .< 1.0e-7,
        p -> abs.(p[3, :] .- 0.05) .< 1.0e-7,
        p -> (abs.(p[2, :] .- lower_y) .< 1.0e-7) .& (p[1, :] .<= corner_x + 1.0e-7),
        p -> abs.(p[2, :] .- (lower_y .+ lower_slope .* (p[1, :] .- corner_x))) .< 1.0e-7,
        p -> p[1, :] .> xmax - 1.0e-6,
        p -> (p[2, :] .> ymax - 1.0e-6) .& (p[1, :] .>= corner_x - 1.0e-7),
        p -> (p[2, :] .> ymax - 1.0e-6) .& (p[1, :] .<= corner_x + 1.0e-7),
        p -> p[1, :] .< xmin + 1.0e-6,
    ]
    mesh.boundarycondition = reshape([3, 3, 1, 1, 3, 2, 2, 2], :, 1)
    return pde, mesh
end

function main()
    pde, mesh = build_case(); solution, pde, mesh = Exasim.exasim(pde, mesh)[1:3]
    println("Julia HCA flared-plate Poisson solution: ", size(solution)); return solution, pde, mesh
end
if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
