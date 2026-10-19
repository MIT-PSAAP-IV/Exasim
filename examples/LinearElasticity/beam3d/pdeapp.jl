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
pde.physicsparam = [1.0 1.0 0.0 0.0 0.01]
pde.tau = [2.0]
pde.RBdim = 0
pde.linearsolvertol = 1.0e-6
pde.NLtol = 1.0e-8
pde.GMRESortho = 1
pde.GMRESrestart = 100
pde.linearsolveriter = 1000
pde.preconditioner = 1
pde.ppdegree = 0

mesh.p, mesh.t = Exasim.Mesh.cubemesh(20, 4, 4, 1)
mesh.p[1, :] .*= 5.0
mesh.boundaryexpr = [
    p -> abs.(p[2, :]) .< 1.0e-6,
    p -> abs.(p[1, :] .- 5.0) .< 1.0e-6,
    p -> abs.(p[2, :] .- 1.0) .< 1.0e-6,
    p -> abs.(p[1, :]) .< 1.0e-6,
    p -> abs.(p[3, :]) .< 1.0e-6,
    p -> abs.(p[3, :] .- 1.0) .< 1.0e-6,
]
mesh.boundarycondition = reshape([1, 1, 1, 2, 1, 3], :, 1)

sol, pde, mesh = Exasim.exasim(pde, mesh)[1:3]
pde.visscalars = []
pde.visvectors = ["displacement", [1, 2, 3]]
Exasim.vis(sol, pde, mesh)
println("Done!")
