using Exasim

include("pdemodel.jl")
pde, mesh = Exasim.initializeexasim()
pde.model = "ModelD"
pde.modelfile = "pdemodel"
pde.platform = "cpu"
pde.mpiprocs = 1
pde.hybrid = 1
pde.debugmode = 0
youngs_modulus, poisson_ratio, amplitude = 1.0, 0.30, 0.10
shear_modulus = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
lame_first = poisson_ratio * youngs_modulus / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
pde.porder = 2
pde.pgauss = 2 * pde.porder
pde.physicsparam = reshape([shear_modulus, lame_first, amplitude], 1, :)
pde.tau = [2.0 * shear_modulus]
pde.linearsolvertol = 1.0e-8
pde.ppdegree = 4
pde.RBdim = 0
pde.GMRESrestart = 100
pde.linearsolveriter = 200
pde.preconditioner = 1

geometry = joinpath(@__DIR__, "platefluid")
mesh.p, mesh.t = Exasim.Mesh.gmshcall(pde, geometry, 2, 0)
mesh.boundaryexpr = [
    p -> abs.(p[2, :] .+ 4.975) .< 1.0e-6,
    p -> abs.(p[1, :] .- 5.5) .< 1.0e-6,
    p -> abs.(p[2, :] .- 5.025) .< 1.0e-6,
    p -> abs.(p[1, :] .+ 4.5) .< 1.0e-6,
    p -> (p[1, :] .>= -1.0e-6) .& (p[1, :] .<= 1.0 + 1.0e-6) .& (p[2, :] .>= -1.0e-6) .& (p[2, :] .<= 0.05 + 1.0e-6),
]
mesh.boundarycondition = reshape([1, 1, 1, 1, 2], :, 1)

sol, pde, mesh = Exasim.exasim(pde, mesh)[1:3]
println("Displacement range: ", extrema(sol[:, 1:2, :]))
