# Model 2 of the non-matching coupled Poisson example: [0.5,1] x [0,1].
# Executed by pdeapp.py, which defines pde, mesh, exasim, numpy and squaremesh.
pde[1], mesh[1] = exasim.initializeexasim()

pde[1]['model'] = "ModelD"
pde[1]['modelfile'] = "pdemodel2"
pde[1]['Cxxpreprocessing'] = 1
pde[1]['mpiprocs'] = 1
pde[1]['hybrid'] = 1
pde[1]['porder'] = 2                  # both models use the same polynomial order
pde[1]['physicsparam'] = numpy.array([1.0])
pde[1]['tau'] = numpy.array([1.0])
pde[1]['NLiter'] = 3
pde[1]['NLtol'] = 1e-10
pde[1]['linearsolvertol'] = 1e-10
pde[1]['matvectol'] = 1e-10
pde[1]['GMRESrestart'] = 300
pde[1]['linearsolveriter'] = 1000
pde[1]['interfacetype'] = 1           # must equal model 1

# 12 x 24 quadrilaterals: interface faces of length 1/24, so the two interface
# discretizations do not match (h ratio 3:2).
mesh[1]['p'], mesh[1]['t'] = squaremesh(12, 24, 1, 1)
mesh[1]['p'][0, :] = 0.5 * mesh[1]['p'][0, :] + 0.5
mesh[1]['boundaryexpr'] = [lambda p: abs(p[1, :]) < 1e-8, lambda p: abs(p[0, :] - 1) < 1e-8,
                           lambda p: abs(p[1, :] - 1) < 1e-8, lambda p: abs(p[0, :] - 0.5) < 1e-8]
mesh[1]['boundarycondition'] = numpy.array([1, 1, 1, 2])
mesh[1]['interfacecondition'] = numpy.array([0, 0, 0, 1])   # boundary 4 is the coupled interface
pde[1]['interfacefluxmap'] = numpy.array([1])
