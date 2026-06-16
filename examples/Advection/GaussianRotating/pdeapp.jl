# import the Exasim frontend (installed via `cmake --install` under
# <prefix>/share/exasim/julia, or Pkg.develop'd from frontends/Julia/Exasim)
using Exasim

# create pde structure and mesh structure
pde, mesh = Exasim.initializeexasim();

# Define PDE model: governing equations, initial solutions, and boundary conditions
pde.model = "ModelC";            # ModelC, ModelD, ModelW
include("pdemodel.jl");          # include the PDE model file

# Set discretization parameters, physical parameters, and solver parameters
pde.porder = 4;          # polynomial degree
pde.torder = 3;          # time-stepping order of accuracy
pde.nstage = 3;          # time-stepping number of stages
pde.physicsparam = [1 1];    # convective velocity
pde.tau = [1.0];               # DG stabilization parameter
pde.dt = 0.025*ones(400);   # time step sizes
pde.soltime = collect(1:length(pde.dt)); # steps at which solution are collected
pde.visdt = 0.025; # visualization timestep size

# Choose computing platform and set number of processors
#pde.platform = "gpu";           # choose this option if NVIDIA GPUs are available
pde.mpiprocs = 4;                # number of MPI processors

# create a linear mesh for a square domain
mesh.p, mesh.t = Exasim.Mesh.SquareMesh(20,20,1); # a mesh of 8 by 8 quadrilaterals
mesh.p = mesh.p .- 0.5; 
# expressions for disjoint boundaries
mesh.boundaryexpr = [p -> (p[2,:] .< -0.5+1e-3), p -> (p[1,:] .> 0.5-1e-3), p -> (p[2,:] .> 0.5-1e-3), p -> (p[1,:] .< -0.5+1e-3)];
mesh.boundarycondition = [1 1 1 1]; # Set boundary condition for each disjoint boundary

# call exasim to generate and run C++ code to solve the PDE model
sol, pde, mesh,~,~,~,~  = Exasim.exasim(pde,mesh);

# visualize the numerical solution of the PDE model using Paraview
pde.visscalars = ["temperature", 1];  # list of scalar fields for visualization
pde.visvectors = []; # list of vector fields for visualization
Exasim.vis(sol,pde,mesh); # visualize the numerical solution
print("Done!");
