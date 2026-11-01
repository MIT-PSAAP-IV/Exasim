# Run with the installed Exasim Julia project active.
using Exasim
include(joinpath(@__DIR__,"pdemodel.jl"))
pde,mesh = Exasim.initializeexasim()
pde.model = "ModelD"; pde.modelfile = "Main"
pde.platform = "cpu"; pde.mpiprocs = 2
pde.ncu = 4; pde.nco = 0; pde.ncw = 0
pde.porder = 4; pde.pgauss = 8
pde.physicsparam = [1.4 100.0 0.72 0.2 1.0 1.0 0.0 45.1429]
pde.tau = [3.0]
pde.NLtol = 1e-6; pde.NLiter = 20
pde.linearsolvertol = 1e-7; pde.linearsolveriter = 400; pde.GMRESrestart = 400
pde.GMRESortho = 0; pde.matvectol = 1e-3
pde.preconditioner = 1; pde.ppdegree = 0; pde.RBdim = 0
mesh.p,mesh.t = Exasim.Mesh.readmesh(joinpath(@__DIR__,"grid.bin"),0)
mesh.boundaryexpr = ["sqrt((x-0.5)^2+y^2)<3", "abs(x)<20"]
mesh.boundarycondition = [1 2] # no-slip adiabatic wall, far field
mesh.curvedboundary = [1 0]
mesh.curvedboundaryexpr = [
    "y^2-(5*0.01*12*(0.29690*sqrt(abs(x))-0.12600*x-0.35160*x^2+0.28430*x^3-0.10150*x^4))^2" "0"]
selection = (seed_boundaries=[1], layers=1)
if haskey(ENV,"HDGLDG_RUN_DIR")
    pde.datapath = abspath(ENV["HDGLDG_RUN_DIR"])
    pde.builddir = joinpath(pde.datapath,".exasim")
    pde.buildpath = pde.builddir
end
sol,pde,mesh = Exasim.hdgldgsolve(pde,mesh,selection)
println("NACA solution size: ",size(sol))
