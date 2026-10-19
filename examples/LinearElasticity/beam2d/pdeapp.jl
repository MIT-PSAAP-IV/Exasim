using Exasim
include("pdemodel.jl")
pde, mesh = Exasim.initializeexasim()
pde.model = "ModelD"
pde.modelfile = "pdemodel"
pde.platform = "cpu"
pde.mpiprocs = 1
pde.hybrid = 1
pde.debugmode = 0
pde.porder = 3
pde.pgauss = 2 * pde.porder
pde.physicsparam = [1.0 1.0 0.0 0.002]
pde.tau = [2.0]
pde.linearsolvertol = 1.0e-6
pde.ppdegree = 10
pde.RBdim = 0
pde.GMRESrestart = 100
pde.linearsolveriter = 100
pde.preconditioner = 1

mesh.p, mesh.t = Exasim.Mesh.SquareMesh(64, 8, 1)
mesh.p[1, :] .*= 8.0
mesh.boundaryexpr = [
    p -> abs.(p[2, :]) .< 1.0e-6,
    p -> abs.(p[1, :] .- 8.0) .< 1.0e-6,
    p -> abs.(p[2, :] .- 1.0) .< 1.0e-6,
    p -> abs.(p[1, :]) .< 1.0e-6,
]
mesh.boundarycondition = reshape([1, 1, 3, 2], :, 1)

sol, pde, mesh = Exasim.exasim(pde, mesh)[1:3]
pde.visscalars = []
pde.visvectors = ["displacement", [1, 2]]
Exasim.vis(sol, pde, mesh)
println("Done!")
