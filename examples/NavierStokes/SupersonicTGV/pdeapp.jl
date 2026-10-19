using Exasim

include("pdemodel.jl")

pde, mesh = Exasim.initializeexasim()
pde.model = "ModelD"
pde.modelfile = "pdemodel"
pde.mpiprocs = 456
pde.hybrid = 1

pde.porder = 2
pde.torder = 3
pde.nstage = 3
delta_t = 1.0e-2
pde.dt = fill(delta_t, round(Int, 20.0 / delta_t))
pde.saveSolFreq = 20
pde.saveSolOpt = 0

nspatial = 128
gam, reynolds, prandtl, mach_inf = 1.4, 1600.0, 0.71, 1.25
reference_density = 1.0
sensor_length = 2.0 * pi / nspatial
av_coefficient = 2.0e-3
pde.physicsparam = [gam reynolds prandtl mach_inf reference_density sensor_length av_coefficient pde.porder]
pde.tau = [5.0]
pde.GMRESortho = 1
pde.GMRESrestart = 24
pde.linearsolvertol = 1.0e-7
pde.linearsolveriter = 24
pde.preconditioner = 1
pde.precMatrixType = 2
pde.NLiter = 1
pde.NLtol = 1.0e-8
pde.ppdegree = 0
pde.RBdim = 5
pde.AV = 1
pde.frozenAVflag = 1
pde.AVsmoothingInter = 2

# This production-scale case mirrors the MATLAB preprocessing/export setup.
# It intentionally does not launch the 456-rank solver from the frontend.
pde.exportapp = "supersonic-tgv"
pde.buildandrun = 0

mesh.p, mesh.t = Exasim.Mesh.cubemesh(nspatial, nspatial, nspatial, 1)
mesh.p .*= 2.0 * pi
mesh.boundaryexpr = [
    p -> abs.(p[2, :]) .< 1.0e-8,
    p -> abs.(p[1, :] .- 2.0*pi) .< 1.0e-8,
    p -> abs.(p[2, :] .- 2.0*pi) .< 1.0e-8,
    p -> abs.(p[1, :]) .< 1.0e-8,
    p -> abs.(p[3, :]) .< 1.0e-8,
    p -> abs.(p[3, :] .- 2.0*pi) .< 1.0e-8,
]
mesh.boundarycondition = reshape(ones(Int, 6), :, 1)
mesh.periodicexpr = [
    2 p -> p[[2,3], :] 4 p -> p[[2,3], :];
    1 p -> p[[1,3], :] 3 p -> p[[1,3], :];
    5 p -> p[[1,2], :] 6 p -> p[[1,2], :]
]

pde, mesh, master, dmd = Exasim.Preprocessing.preprocessing(pde, mesh)
println("Prepared the SupersonicTGV input for ", pde.mpiprocs, " MPI ranks.")
