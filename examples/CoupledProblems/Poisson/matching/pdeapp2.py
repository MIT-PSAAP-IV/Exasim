# Model 2 of the matching-interface coupled Poisson example: [0.5,1] x [0,1].
# Executed by pdeapp.py, which defines pde, mesh, exasim, numpy and squaremesh.
pde[1], mesh[1] = exasim.initializeexasim()

# Define a PDE model: governing equations, initial solutions, and boundary conditions
pde[1]['model'] = "ModelD"          # ModelC, ModelD, ModelW
pde[1]['modelfile'] = "pdemodel2"   # name of a file defining the PDE model
pde[1]['Cxxpreprocessing'] = 1

# Choose computing platform and set number of processors
#pde[1]['platform'] = "gpu"         # choose this option if NVIDIA GPUs are available
pde[1]['mpiprocs'] = 4              # number of MPI processors
pde[1]['hybrid'] = 1
pde[1]['linearsolvertol'] = 1e-6
pde[1]['NLiter'] = 1

# Set discretization parameters, physical parameters, and solver parameters
pde[1]['porder'] = 3                          # polynomial degree
pde[1]['physicsparam'] = numpy.array([1.0])   # unit thermal conductivity
pde[1]['tau'] = numpy.array([1.0])            # DG stabilization parameter

# create a grid of 8 by 16 quadrilaterals on [0.5,1] x [0,1]
mesh[1]['p'], mesh[1]['t'] = squaremesh(8, 16, 1, 1)
mesh[1]['p'][0, :] = 0.5 * mesh[1]['p'][0, :] + 0.5

# expressions for domain boundaries
mesh[1]['boundaryexpr'] = [lambda p: abs(p[1, :]) < 1e-8, lambda p: abs(p[0, :] - 1) < 1e-8,
                           lambda p: abs(p[1, :] - 1) < 1e-8, lambda p: abs(p[0, :] - 0.5) < 1e-8]
mesh[1]['boundarycondition'] = numpy.array([1, 1, 1, 2])  # Set boundary condition for each boundary
