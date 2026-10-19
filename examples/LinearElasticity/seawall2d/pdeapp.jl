using Exasim

include("pdemodel.jl")
pde, mesh = Exasim.initializeexasim()
pde.model = "ModelD"
pde.modelfile = "pdemodel"
pde.platform = "cpu"
pde.mpiprocs = 1
pde.hybrid = 1
pde.debugmode = 0
pde.porder = 1
pde.pgauss = 2
pde.physicsparam = reshape([1.25, 0.833, 0.0, 0.1], 1, :)
pde.tau = [2.0]
pde.linearsolvertol = 1.0e-6
pde.ppdegree = 1
pde.RBdim = 0
pde.GMRESrestart = 1000
pde.linearsolveriter = 1000
pde.preconditioner = 1
pde.NLiter = 2

geometry = joinpath(@__DIR__, "seawall")
mesh.p, mesh.t = Exasim.Mesh.gmshcall(pde, geometry, 2, 1)
mesh.t = reverse(mesh.t, dims=1)
ymin, ymax = extrema(mesh.p[2, :])
mesh.boundaryexpr = [
    p -> abs.(p[2, :] .- ymin) .< 1.0e-3,
    p -> abs.(p[2, :] .- ymax) .< 1.0e-3,
    p -> p[1, :] .< 1.0e-3,
    p -> abs.(p[1, :] .- 0.51816) .< 1.0e-3,
    p -> abs.(p[2, :]) .< 100.0,
]
mesh.boundarycondition = reshape([2, 1, 3, 1, 2], :, 1)

sol, pde, mesh = Exasim.exasim(pde, mesh)[1:3]
println("Displacement range: ", extrema(sol[:, 1:2, :]))
