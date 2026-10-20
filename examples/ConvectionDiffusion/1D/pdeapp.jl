# import the Exasim frontend (installed via `cmake --install` under
# <prefix>/share/exasim/julia, or Pkg.develop'd from frontends/Julia/Exasim)
using Exasim

# create pde structure and mesh structure
pde, mesh = Exasim.initializeexasim();

# Define PDE model: governing equations, initial solutions, and boundary conditions
pde.model = "ModelD";            # ModelC, ModelD, ModelW
include("pdemodel.jl");          # include the PDE model file

# Set discretization parameters, physical parameters, and solver parameters
pde.porder = 3;                  # polynomial degree
pde.physicsparam = [1.0 2*pi];    # thermal conductivity and boundary value
pde.tau = [2*pi];                 # DG stabilization parameter

# Choose computing platform and set number of processors
#pde.platform = "gpu";           # choose this option if NVIDIA GPUs are available
pde.mpiprocs = 1;                # number of MPI processors

# Create the same 16-element unit-interval mesh as the MATLAB and Python cases.
mesh.p = reshape(collect(range(0.0, 1.0, length=17)), 1, :);
mesh.t = [collect(1:16)'; collect(2:17)'];
# expressions for disjoint boundaries
mesh.boundaryexpr = [p -> (p[1,:] .> 1-1e-3), p -> (p[1,:] .< 1e-3)];
mesh.boundarycondition = [1 1]; # Set boundary condition for each disjoint boundary

# call exasim to generate and run C++ code to solve the PDE model
sol, pde, mesh,~,~,~,~  = Exasim.exasim(pde,mesh);

# visualize the numerical solution of the PDE model using Paraview
pde.visscalars = ["temperature", 1];  # list of scalar fields for visualization
pde.visvectors = ["temperature gradient", [2]]; # list of vector fields for visualization
Exasim.vis(sol,pde,mesh); # visualize the numerical solution
print("Done!");
