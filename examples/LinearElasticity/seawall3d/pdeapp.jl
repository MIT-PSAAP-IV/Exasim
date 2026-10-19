using Exasim

include("pdemodel.jl")
include("seawall_mesh.jl")
pde, mesh = Exasim.initializeexasim()
pde.model = "ModelD"
pde.modelfile = "pdemodel"
pde.platform = "cpu"
pde.mpiprocs = 8
pde.hybrid = 1
pde.debugmode = 0
pde.porder = 2
pde.pgauss = 4
pde.physicsparam = reshape([1.25, 0.833, 0.015, 0.0, -0.015], 1, :)
pde.tau = [5.0]
pde.RBdim = 0
pde.linearsolvertol = 1.0e-6
pde.NLtol = 1.0e-6
pde.GMRESortho = 1
pde.GMRESrestart = 100
pde.linearsolveriter = 200
pde.preconditioner = 1
pde.ppdegree = 0

mesh.p, mesh.t, mesh.dgnodes = make_seawall_mesh(pde)
ymin, ymax = extrema(mesh.p[2, :])
zmin, zmax = extrema(mesh.p[3, :])
mesh.boundaryexpr = [
    p -> abs.(p[3, :] .- zmin) .< 1.0e-3,
    p -> abs.(p[3, :] .- zmax) .< 1.0e-3,
    p -> abs.(p[2, :] .- ymin) .< 1.0e-3,
    p -> abs.(p[2, :] .- ymax) .< 1.0e-3,
    p -> p[1, :] .< 1.0e-3,
    p -> abs.(p[1, :] .- 0.34544) .< 1.0e-3,
    p -> abs.(p[2, :] .- 1.9812) .< 1.0e-3,
    p -> abs.(p[2, :]) .< 100.0,
]
mesh.boundarycondition = reshape([2, 2, 2, 1, 3, 1, 1, 2], :, 1)

sol, pde, mesh = Exasim.exasim(pde, mesh)[1:3]
println("Displacement range: ", extrema(sol[:, 1:3, :]))
