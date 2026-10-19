"""Julia frontend for the legacy Mach-8.98 cylinder setup."""

using Exasim
include("pdemodel.jl")
include("cylinder_mesh.jl")

pde, _ = Exasim.initializeexasim()
pde.model, pde.modelfile = "ModelD", "pdemodel"
pde.platform, pde.mpiprocs = "gpu", 1
pde.porder, pde.pgauss = 2, 4
pde.torder, pde.nstage = 1, 1
pde.dt = 1.0e-5 .* ones(10000)
pde.visdt, pde.saveSolFreq = pde.dt[1], 100
pde.soltime = collect(100:100:length(pde.dt))

gamma, mach, reynolds, prandtl = 1.4, 8.98, 0.71, 23319.0
tref, twall = 200.0, 500.0
pinf = 1.0 / (gamma * mach^2)
tinf = pinf / (gamma - 1.0)
pde.physicsparam = reshape([
    gamma, reynolds, prandtl, mach, 1.0, 1.0, 0.0,
    0.5 + pinf / (gamma - 1.0), tinf, tref, twall,
    1.5, 0.5, 0.0, pde.porder, 0.001, 2.5,
], 1, :)
pde.tau = [4.0]
pde.GMRESrestart, pde.linearsolveriter = 29, 30
pde.linearsolvertol, pde.NLtol, pde.NLiter = 1.0e-4, 1.0e-10, 3
pde.precMatrixType, pde.ptcMatrixType = 2, 0
pde.matvectol, pde.AV = 1.0e-6, 1

mesh = make_cylinder_mesh(pde.porder)
distance = wall_distance(mesh, pde.porder)
npe, _, ne = size(mesh.dgnodes)
mesh.odg = zeros(npe, 2, ne)
mesh.odg[:, 2, :] .= curved_map_scale(mesh)
mesh.odg[:, 2, :] .*= tanh.(100.0 .* distance[:, 1, :])
mesh.udg = initialize_solution(mesh, distance, vec(pde.physicsparam))

pde, mesh = Exasim.preprocessing(pde, mesh)[1:2]
Exasim.Gencode.gencode(pde)
Exasim.Gencode.compilecode(pde)
println("Generated and compiled the Julia Mach-8.98 cylinder application.")
