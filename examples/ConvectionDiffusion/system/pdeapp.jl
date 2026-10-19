using Exasim

include("pdemodel.jl")
pde, mesh = Exasim.initializeexasim()
pde.model = "ModelD"
pde.modelfile = "pdemodel"
pde.mpiprocs = 1
pde.hybrid = 1
pde.Cxxpreprocessing = 0
pde.porder = 3
pde.physicsparam = reshape([0.1], 1, :)
pde.tau = [1.0]
pde.GMRESrestart = 50
pde.ppdegree = 20
pde.linearsolvertol = 1.0e-6

mesh.p, mesh.t = Exasim.Mesh.SquareMesh(32, 32, 1)
mesh.boundaryexpr = [
    p -> abs.(p[2, :]) .< 1.0e-8,
    p -> abs.(p[1, :] .- 1.0) .< 1.0e-8,
    p -> abs.(p[2, :] .- 1.0) .< 1.0e-8,
    p -> abs.(p[1, :]) .< 1.0e-8,
]
mesh.boundarycondition = reshape(ones(Int, 4), :, 1)

sol, pde, mesh = Exasim.exasim(pde, mesh)[1:3]
pde.visscalars = ["temperature1", 1, "temperature2", 2]
pde.visvectors = ["temperature gradient 1", [3, 5], "temperature gradient 2", [4, 6]]
Exasim.vis(sol, pde, mesh)
println("Done!")
