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
pde.Cxxpreprocessing = 0
pde.porder = 3
pde.pgauss = 2 * pde.porder
pde.torder = 1
pde.nstage = 1
pde.tau = [1.0]

temperature = 2000.0
boltzmann = 8.617e-5
length_scale = 4.0e-5
diffusivity = 9.28e-6 * exp(-1.46 / (boltzmann * temperature))
time_scale = 0.5 * length_scale^2 / diffusivity
pde.physicsparam = reshape([time_scale * diffusivity / length_scale^2, 18.0 * time_scale, 1.0], 1, :)
pde.ppdegree = 1
pde.RBdim = 0
pde.linearsolvertol = 1.0e-12

dt = 5.0e-3
pde.dt = dt .* ones(200)
pde.soltime = collect(1:200)
pde.visdt = dt
mesh.p = reshape(collect(range(0.0, 1.0, length=301)), 1, :)
mesh.t = [collect(1:300)'; collect(2:301)']
mesh.boundaryexpr = [p -> abs.(p[1, :]) .< 1.0e-8, p -> abs.(p[1, :] .- 1.0) .< 1.0e-8]
mesh.boundarycondition = reshape([1, 2], :, 1)

sol, pde, mesh = Exasim.exasim(pde, mesh)[1:3]
println("Concentration range: ", extrema(selectdim(sol, 2, 1)))
