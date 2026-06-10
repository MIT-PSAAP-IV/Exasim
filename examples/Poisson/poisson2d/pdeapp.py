# import external modules
import numpy

# import the Exasim frontend (see README, "Using the frontends")
import exasim

# Create pde object and mesh object
pde, mesh = exasim.initializeexasim();

# Define a PDE model: governing equations and boundary conditions
pde['model'] = "ModelD";       # ModelC, ModelD, ModelW
pde['modelfile'] = "pdemodel"; # name of a file defining the PDE model

# Choose computing platform and set number of processors
#pde['platform'] = "gpu";   # choose this option if NVIDIA GPUs are available
pde['mpiprocs'] = 2;        # number of MPI processors
pde['hybrid'] = 1;          # 0 -> LDG, 1 -> HDG

# Set discretization parameters, physical parameters, and solver parameters
pde['porder'] = 3;         # polynomial degree
pde['physicsparam'] = numpy.array([1.0]);   # unit thermal conductivity
pde['tau'] = numpy.array([1.0]);            # DG stabilization parameter
pde['saveSolBouFreq'] = 1;
pde['ibs'] = 1;

# create a mesh of 16 by 16 quads on a square domain
mesh['p'], mesh['t'] = exasim.Mesh.SquareMesh(16,16,1)[0:2];
# expressions for domain boundaries
mesh['boundaryexpr'] = [lambda p: (p[1,:] < 1e-3), lambda p: (p[0,:] > 1-1e-3), lambda p: (p[1,:] > 1-1e-3), lambda p: (p[0,:] < 1e-3)];
mesh['boundarycondition'] = numpy.array([1, 1, 1, 1]); # Set boundary condition for each boundary

# call exasim to generate and run C++ code to solve the PDE model
sol, pde, mesh  = exasim.exasim(pde,mesh)[0:3];

# visualize the numerical solution of the PDE model using Paraview
pde['visscalars'] = ["temperature", 0]; # list of scalar fields for visualization
pde['visvectors'] = ["temperature gradient", numpy.array([1, 2]).astype(int)]; # list of vector fields for visualization
exasim.vis(sol,pde,mesh); # visualize the numerical solution
print("Done!");
