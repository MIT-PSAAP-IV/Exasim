using Exasim
include("pdemodel.jl")
pde, mesh = Exasim.initializeexasim()
pde.model, pde.modelfile, pde.platform = "ModelD", "pdemodel", "cpu"
pde.mpiprocs, pde.hybrid = 1, 1
pde.porder, pde.pgauss = 3, 6
pde.physicsparam = reshape([80.194,488.806,0.0,125.0/16.0],1,:)
pde.tau = [1000.0]
pde.linearsolvertol, pde.ppdegree, pde.RBdim = 1.0e-6, 10, 0
pde.GMRESrestart, pde.linearsolveriter, pde.preconditioner = 100, 100, 1
mesh.p, mesh.t = Exasim.Mesh.SquareMesh(10,10,1)
xi, yi = copy(mesh.p[1,:]), copy(mesh.p[2,:])
mesh.p[1,:] .= 48.0.*xi; mesh.p[2,:] .= yi.*(44.0.+16.0.*xi)
xmin, xmax = extrema(mesh.p[1,:])
mesh.boundaryexpr = [p->abs.(p[1,:].-xmin).<1e-6, p->abs.(p[1,:].-xmax).<1e-6,
    p->p[2,:].<44.0+1e-6, p->p[2,:].>44.0-1e-6]
mesh.boundarycondition = reshape([2,3,1,1],:,1)
sol, pde, mesh = Exasim.exasim(pde,mesh)[1:3]
println("Deformed-coordinate range: ", extrema(sol[:,1:2,:]))
