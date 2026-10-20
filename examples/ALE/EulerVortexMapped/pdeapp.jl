using Exasim

include("pdemodel.jl")
pde, mesh = Exasim.initializeexasim()
pde.model = "ModelC"
pde.modelfile = "pdemodel"
pde.mpiprocs = 16
pde.hybrid = 0
pde.porder = 2
pde.torder = 3
pde.nstage = 3
final_time = sqrt(125.0)
pde.dt = 0.05 .* ones(Int(ceil(final_time / 0.05)))
pde.soltime = [length(pde.dt)]
pde.visdt = 1.0
pde.linearsolveriter = 50
pde.GMRESrestart = 50
pde.NLiter = 2
pde.physicsparam = reshape([1.4, 0.5, 0.3, 1.5, 5.0, 5.0], 1, :)
pde.tau = [3.0]
pde.ALE = 1

mesh.p, mesh.t = Exasim.Mesh.SquareMesh(64, 48, 1)
mesh.p[1, :] .*= 20.0
mesh.p[2, :] .*= 15.0
mesh.boundaryexpr = [
    p -> abs.(p[2, :]) .< 1.0e-8,
    p -> abs.(p[1, :] .- 20.0) .< 1.0e-8,
    p -> abs.(p[2, :] .- 15.0) .< 1.0e-8,
    p -> abs.(p[1, :]) .< 1.0e-8,
]
mesh.boundarycondition = reshape(ones(Int, 4), :, 1)
mesh.periodicexpr = [2 p -> p[2, :] 4 p -> p[2, :]; 1 p -> p[1, :] 3 p -> p[1, :]]

pde, mesh = Exasim.preprocessing(pde, mesh)[1:2]
Exasim.Gencode.gencode(pde)
println("Generated the mapped Euler-vortex input and source files.")
