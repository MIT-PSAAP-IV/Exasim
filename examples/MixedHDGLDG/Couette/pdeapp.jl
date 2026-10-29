# Run directly: julia --project=/path/to/Exasim/frontends/Julia/Exasim pdeapp.jl
using Exasim
include(joinpath(@__DIR__,"pdemodel.jl"))
pde,mesh = Exasim.initializeexasim()
pde.model = "ModelD"; pde.modelfile = "Main"
pde.platform = "cpu"; pde.mpiprocs = 2
pde.ncu = 4; pde.nco = 0; pde.ncw = 0
pde.porder = 2; pde.pgauss = 2*pde.porder
pde.physicsparam = [1.4 10 0.72 0.8 1 1 1.1]; pde.tau = [3.0]
pde.preconditioner = 1; pde.NLtol = 1e-10; pde.NLiter = 20
pde.linearsolvertol = 1e-8; pde.linearsolveriter = 2000; pde.GMRESrestart = 200
pde.GMRESortho = 1; pde.matvectol = 1e-3
pde.RBdim = 0; pde.ppdegree = 0
if haskey(ENV,"HDGLDG_RUN_DIR")
    pde.datapath = abspath(ENV["HDGLDG_RUN_DIR"])
    pde.builddir = joinpath(pde.datapath,".exasim")
    pde.buildpath = pde.builddir
end
N = 4
mesh.p,mesh.t = Exasim.Mesh.SquareMesh(N,N,1)
mesh.boundaryexpr = ["abs(y)<1e-8","abs(x-1)<1e-8","abs(y-1)<1e-8","abs(x)<1e-8"]
mesh.boundarycondition = [1 1 1 1]
mesh.curvedboundary = [0 0 0 0]
mesh.curvedboundaryexpr = ["" "" "" ""]
centers = vec(sum(reshape(mesh.p[1,vec(mesh.t)],size(mesh.t)),dims=1))/size(mesh.t,1)
labels = Int.(centers .<= 0.5)
sol,pde,mesh = Exasim.hdgldgsolve(pde,mesh,labels)
println("Mixed Navier-Stokes solution size: ",size(sol))
println("Native output: ",pde.dataoutpath)
