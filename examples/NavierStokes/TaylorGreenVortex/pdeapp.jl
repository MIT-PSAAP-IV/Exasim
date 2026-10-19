"""Julia frontend for the periodic Taylor--Green vortex."""

using Exasim
include("pdemodel.jl")

pde, mesh = Exasim.initializeexasim()
pde.model, pde.modelfile = "ModelD", "pdemodel"
pde.platform, pde.mpiprocs, pde.hybrid = "cpu", 8, 1
pde.porder, pde.torder, pde.nstage = 2, 3, 3
pde.dt = fill(0.16, 200)
pde.soltime = collect(1:200)
pde.visdt = 0.05
pde.physicsparam = [1.4 1600.0 0.71 0.2]
pde.tau = [2.0]
pde.GMRESrestart, pde.linearsolveriter = 50, 51
pde.linearsolvertol, pde.NLiter = 1.0e-6, 2
pde.preconditioner, pde.precMatrixType, pde.ppdegree = 1, 2, 0

mesh.p, mesh.t = Exasim.Mesh.cubemesh(24, 24, 24, 1)
mesh.p .*= 2.0*pi
mesh.boundaryexpr = [
    p -> abs.(p[2, :]) .< 1.0e-8,
    p -> abs.(p[1, :] .- 2.0*pi) .< 1.0e-8,
    p -> abs.(p[2, :] .- 2.0*pi) .< 1.0e-8,
    p -> abs.(p[1, :]) .< 1.0e-8,
    p -> abs.(p[3, :]) .< 1.0e-8,
    p -> abs.(p[3, :] .- 2.0*pi) .< 1.0e-8,
]
mesh.boundarycondition = reshape(ones(Int, 6), :, 1)
mesh.periodicexpr = [
    2 p -> p[[2,3], :] 4 p -> p[[2,3], :];
    1 p -> p[[1,3], :] 3 p -> p[[1,3], :];
    5 p -> p[[1,2], :] 6 p -> p[[1,2], :]
]

sol, pde, mesh = Exasim.exasim(pde, mesh)[1:3]
println("Taylor--Green vortex completed with solution size ", size(sol))
