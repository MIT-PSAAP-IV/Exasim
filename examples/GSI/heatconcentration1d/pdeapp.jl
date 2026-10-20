using Exasim

include("pdemodel.jl")
pde, mesh = Exasim.initializeexasim()
pde.model = "ModelD"
pde.modelfile = "pdemodel"
pde.platform = "cpu"
pde.mpiprocs = 1
pde.hybrid = 1
pde.Cxxpreprocessing = 0
pde.porder = 3
pde.pgauss = 2 * pde.porder
pde.torder = 1
pde.nstage = 1
pde.tau = [1.0]

length_scale = 5.0e-2
wall_temperature = 500.0
temperature_scale = 1000.0
heat_capacity = 170.0 * 13000.0
thermal_diffusivity = 15.0 / heat_capacity
time_scale = length_scale^2 / thermal_diffusivity
mu = [length_scale, wall_temperature, temperature_scale, 8.314, 8.617e-5,
      1.0e-5, 1.415e10, -104.9e3, 7.45e4, 9.28e-6, 1.46, time_scale,
      time_scale * 9.28e-6 / length_scale^2,
      time_scale * thermal_diffusivity / length_scale^2, heat_capacity, 1.5]
pde.physicsparam = reshape(mu, 1, :)
pde.ppdegree = 10
pde.RBdim = 0
pde.linearsolvertol = 1.0e-8
pde.dt = 0.001 .* collect(1.0:200.0).^2
pde.soltime = collect(1:200)

mesh.p = reshape(collect(range(0.0, 1.0, length=301)), 1, :)
mesh.t = [collect(1:300)'; collect(2:301)']
mesh.boundaryexpr = [p -> abs.(p[1, :]) .< 1.0e-8, p -> abs.(p[1, :] .- 1.0) .< 1.0e-8]
mesh.boundarycondition = reshape([1, 2], :, 1)

sol, pde, mesh = Exasim.exasim(pde, mesh)[1:3]
println("Temperature range: ", extrema(selectdim(sol, 2, 1)))
println("Concentration range: ", extrema(selectdim(sol, 2, 2)))
