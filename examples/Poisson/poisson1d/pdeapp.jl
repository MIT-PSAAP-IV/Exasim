using Exasim

include("pdemodel.jl")
pde, mesh = Exasim.initializeexasim()
pde.model = "ModelD"
pde.modelfile = "pdemodel"
pde.platform = "cpu"
pde.mpiprocs = 1
pde.hybrid = 1
pde.debugmode = 0
pde.nd = 1
pde.porder = 2
pde.pgauss = 2 * pde.porder
pde.physicsparam = reshape([1.0], 1, :)
pde.tau = [1.0]
pde.linearsolvertol = 1.0e-8
pde.ppdegree = 1
pde.RBdim = 0
pde.Cxxpreprocessing = 0

mesh.p = reshape(collect(range(0.0, 1.0, length=6)), 1, :)
mesh.t = [collect(1:5)'; collect(2:6)']
mesh.boundaryexpr = [p -> abs.(p[1, :]) .< 1.0e-8, p -> abs.(p[1, :] .- 1.0) .< 1.0e-8]
mesh.boundarycondition = reshape([1, 1], :, 1)

sol, pde, mesh = Exasim.exasim(pde, mesh)[1:3]
println("Poisson 1D solution range: ", extrema(sol[:, 1, :]))
