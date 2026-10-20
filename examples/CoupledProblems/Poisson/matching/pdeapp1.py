# Model 1 of the matching-interface coupled Poisson example: [0,0.5] x [0,1].
# Executed by pdeapp.py, which defines pde, mesh, exasim, numpy and squaremesh.
pde[0], mesh[0] = exasim.initializeexasim()

# Define a PDE model: governing equations, initial solutions, and boundary conditions
pde[0]['model'] = "ModelD"          # ModelC, ModelD, ModelW
pde[0]['modelfile'] = "pdemodel1"   # name of a file defining the PDE model
pde[0]['Cxxpreprocessing'] = 1

# Choose computing platform and set number of processors
#pde[0]['platform'] = "gpu"         # choose this option if NVIDIA GPUs are available
pde[0]['mpiprocs'] = 3              # number of MPI processors
pde[0]['hybrid'] = 1
pde[0]['linearsolvertol'] = 1e-6
pde[0]['NLiter'] = 1

# Set discretization parameters, physical parameters, and solver parameters
pde[0]['porder'] = 3                          # polynomial degree
pde[0]['physicsparam'] = numpy.array([1.0])   # unit thermal conductivity
pde[0]['tau'] = numpy.array([1.0])            # DG stabilization parameter

# create a grid of 8 by 16 quadrilaterals on [0,0.5] x [0,1]
mesh[0]['p'], mesh[0]['t'] = squaremesh(8, 16, 1, 1)
mesh[0]['p'][0, :] = 0.5 * mesh[0]['p'][0, :]

# expressions for domain boundaries
mesh[0]['boundaryexpr'] = [lambda p: abs(p[1, :]) < 1e-8, lambda p: abs(p[0, :] - 0.5) < 1e-8,
                           lambda p: abs(p[1, :] - 1) < 1e-8, lambda p: abs(p[0, :]) < 1e-8]
mesh[0]['boundarycondition'] = numpy.array([1, 2, 1, 1])  # Set boundary condition for each boundary
