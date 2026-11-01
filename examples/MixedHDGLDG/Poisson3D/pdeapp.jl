# Run directly with the usual installed Exasim project.
using Exasim
include(joinpath(@__DIR__,"pdemodel.jl"))
pde,mesh = Exasim.initializeexasim()
pde.model = "ModelD"; pde.modelfile = "Main"
pde.platform = "cpu"; pde.mpiprocs = 2
pde.ncu = 1; pde.nco = 0; pde.ncw = 0
pde.porder = 2; pde.pgauss = 2*pde.porder
pde.physicsparam = [1.0 0.0]; pde.tau = [1.0]
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
mesh.p,mesh.t = Exasim.Mesh.cubemesh(N,N,N,1)
mesh.boundaryexpr = ["y<1e-8","x>1-1e-8","y>1-1e-8","x<1e-8","z<1e-8","z>1-1e-8"]
mesh.boundarycondition = [1 1 1 1 1 1]
mesh.curvedboundary = [0 0 0 0 0 0]
mesh.curvedboundaryexpr = ["" "" "" "" "" ""]
centers = vec(sum(reshape(mesh.p[1,vec(mesh.t)],size(mesh.t)),dims=1))/size(mesh.t,1)
labels = Int.(centers .<= 0.5)
sol,pde,mesh = Exasim.hdgldgsolve(pde,mesh,labels)
println("Mixed Poisson3D solution size: ",size(sol))
println("Native output: ",pde.dataoutpath)
