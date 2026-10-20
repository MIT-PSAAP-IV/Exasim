using Exasim

include("pdemodel.jl")
pde, mesh = Exasim.initializeexasim()
pde.model = "ModelD"
pde.modelfile = "pdemodel"
pde.platform = "cpu"
pde.mpiprocs = 1
pde.hybrid = 1
pde.debugmode = 0

youngs_modulus = 240.565
poisson_ratio = 0.495
shear_modulus = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
lame_first = poisson_ratio * youngs_modulus / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
pde.porder = 3
pde.pgauss = 2 * pde.porder
pde.physicsparam = reshape([shear_modulus, lame_first, 0.0, 16.0], 1, :)
pde.tau = [2.0 * shear_modulus]
pde.linearsolvertol = 1.0e-6
pde.ppdegree = 10
pde.RBdim = 0
pde.GMRESrestart = 100
pde.linearsolveriter = 100
pde.preconditioner = 1

mesh.p, mesh.t = Exasim.Mesh.SquareMesh(10, 10, 1)
xi, eta = copy(mesh.p[1, :]), copy(mesh.p[2, :])
mesh.p[1, :] .= 48.0 .* xi
mesh.p[2, :] .= eta .* (44.0 .+ 16.0 .* xi)
xmin, xmax = extrema(mesh.p[1, :])
mesh.boundaryexpr = [
    p -> abs.(p[1, :] .- xmin) .< 1.0e-6,
    p -> abs.(p[1, :] .- xmax) .< 1.0e-6,
    p -> p[2, :] .< 44.0 + 1.0e-6,
    p -> p[2, :] .> 44.0 - 1.0e-6,
]
mesh.boundarycondition = reshape([2, 3, 1, 1], :, 1)

sol, pde, mesh = Exasim.exasim(pde, mesh)[1:3]
println("Displacement range: ", extrema(sol[:, 1:2, :]))
