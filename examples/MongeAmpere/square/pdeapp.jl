"""Julia two-stage density-equalizing mesh-adaptivity example."""

using Exasim
using LinearAlgebra

const CASEDIR = @__DIR__

function monitor_values(x, y)
    argument = 100.0 .* (x .* x .+ y .* y .- 0.25^2)
    density = 1.0 .+ 20.0 ./ cosh.(argument)
    scale = -4000.0 .* sinh.(argument) ./ cosh.(argument).^2
    return density, scale .* x, scale .* y
end

function mean_monitor()
    # Deterministic midpoint integration is more than sufficient for the
    # smooth monitor normalization used by this demonstration.
    n = 2048
    points = range(-1.0 + 1.0 / n, 1.0 - 1.0 / n, length=n)
    total = 0.0
    for x in points, y in points
        total += monitor_values(x, y)[1]
    end
    return total / n^2
end

function base_case(run_directory)
    pde, mesh = Exasim.initializeexasim()
    pde.model = "ModelD"
    pde.platform = "cpu"
    pde.mpiprocs = 1
    pde.hybrid = 1
    pde.porder = 4
    pde.pgauss = 8
    pde.linearsolvertol = 1.0e-8
    pde.NLtol = 1.0e-8
    pde.GMRESrestart = 200
    pde.linearsolveriter = 400
    pde.preconditioner = 1
    pde.ppdegree = 10
    pde.datapath = run_directory

    mesh.p, mesh.t = Exasim.Mesh.SquareMesh(50, 50, 1)
    mesh.p .= 2.0 .* mesh.p .- 1.0
    mesh.boundaryexpr = [
        p -> abs.(p[2, :] .+ 1.0) .< 1.0e-8,
        p -> abs.(p[1, :] .- 1.0) .< 1.0e-8,
        p -> abs.(p[2, :] .- 1.0) .< 1.0e-8,
        p -> abs.(p[1, :] .+ 1.0) .< 1.0e-8,
    ]
    mesh.boundarycondition = ones(Int, 1, 4)
    mesh.f = Exasim.Preprocessing.facenumbering(mesh.p, mesh.t, 1, mesh.boundaryexpr, [])
    mesh.dgnodes = Exasim.Preprocessing.createdgnodes(mesh.p, mesh.t, mesh.f, [], [], pde.porder)
    return pde, mesh
end

function main()
    run_directory = joinpath(CASEDIR, "julia_run")
    mkpath(run_directory)
    pde, mesh = base_case(run_directory)
    theta = mean_monitor()
    x, y = mesh.dgnodes[:, 1, :], mesh.dgnodes[:, 2, :]
    density, density_x, density_y = monitor_values(x, y)

    include("pdemodel_poisson.jl")
    pde.modelfile = "pdemodel_poisson"
    pde.physicsparam = [1.0 theta]
    pde.tau = [2.0]
    pde.builddir = joinpath(run_directory, ".exasim", "poisson")
    mesh.vdg = cat(density ./ theta, density_x ./ theta, density_y ./ theta; dims=2)
    poisson, _, mesh = Exasim.exasim(pde, mesh)[1:3]
    ndims(poisson) == 4 && (poisson = poisson[:, :, :, end])

    original = copy(mesh.dgnodes)
    adapted = copy(original)
    include("pdemodel_transport.jl")
    for coordinate in 1:2
        transport, transport_mesh = base_case(run_directory)
        transport.modelfile = "pdemodel_transport"
        transport.physicsparam = [1.0e-4 theta]
        transport.tau = [1.0]
        transport.dt = 0.02 .* ones(50)
        transport.soltime = collect(1:50)
        transport.torder = 2
        transport.nstage = 2
        transport.builddir = joinpath(run_directory, ".exasim", "transport")
        transport_mesh.dgnodes = copy(original)
        transport_mesh.vdg = cat(density ./ theta, density_x ./ theta,
                                  density_y ./ theta, poisson[:, 2, :],
                                  poisson[:, 3, :]; dims=2)
        transport_mesh.udg = original[:, coordinate:coordinate, :]
        moved = Exasim.exasim(transport, transport_mesh)[1]
        adapted[:, coordinate, :] .= moved[:, 1, :, end]
    end

    open(joinpath(run_directory, "adapted_mesh.bin"), "w") do io
        write(io, adapted)
    end
    println("Julia Monge-Ampere square mesh: ", size(adapted))
    return adapted
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
