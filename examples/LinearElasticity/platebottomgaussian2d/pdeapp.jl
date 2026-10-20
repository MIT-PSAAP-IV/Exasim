using Exasim

include("pdemodel.jl")
include("plate_mesh.jl")
pde, mesh = Exasim.initializeexasim()
pde.model = "ModelD"
pde.modelfile = "pdemodel"
pde.platform = "cpu"
pde.mpiprocs = 1
pde.hybrid = 1
pde.debugmode = 0

youngs_modulus, poisson_ratio = 1.0, 0.30
shear_modulus = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
lame_first = poisson_ratio * youngs_modulus / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
pde.porder = 2
pde.pgauss = 2 * pde.porder
pde.physicsparam = reshape([shear_modulus, lame_first, 0.08, 0.0, 0.15], 1, :)
pde.tau = [2.0 * lame_first]
pde.linearsolvertol = 1.0e-8
pde.ppdegree = 12
pde.RBdim = 0
pde.GMRESrestart = 100
pde.linearsolveriter = 200
pde.preconditioner = 1

mesh.p, mesh.t = make_plate_mesh()
mesh.boundaryexpr = [
    p -> abs.(p[2, :]) .< 1.0e-8,
    p -> abs.(p[1, :] .- 0.5) .< 1.0e-8,
    p -> abs.(p[2, :] .- 0.2) .< 1.0e-8,
    p -> abs.(p[1, :] .+ 0.5) .< 1.0e-8,
]
mesh.boundarycondition = reshape([2, 1, 1, 1], :, 1)

sol, pde, mesh = Exasim.exasim(pde, mesh)[1:3]
println("Displacement range: ", extrema(sol[:, 1:2, :]))
