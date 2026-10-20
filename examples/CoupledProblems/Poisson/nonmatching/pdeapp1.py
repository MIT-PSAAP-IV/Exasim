# Model 1 of the non-matching coupled Poisson example: [0,0.5] x [0,1].
# Executed by pdeapp.py, which defines pde, mesh, exasim, numpy and squaremesh.
pde[0], mesh[0] = exasim.initializeexasim()

pde[0]['model'] = "ModelD"
pde[0]['modelfile'] = "pdemodel1"
pde[0]['Cxxpreprocessing'] = 1
pde[0]['mpiprocs'] = 1
pde[0]['hybrid'] = 1
pde[0]['porder'] = 2
pde[0]['physicsparam'] = numpy.array([1.0])
pde[0]['tau'] = numpy.array([1.0])
pde[0]['NLiter'] = 3
pde[0]['NLtol'] = 1e-10
pde[0]['linearsolvertol'] = 1e-10
pde[0]['matvectol'] = 1e-10
pde[0]['GMRESrestart'] = 300
pde[0]['linearsolveriter'] = 1000

# Non-matching interface (app.problem[34] = 1): the backend locates donor points on
# the other model's independently generated mesh; no frontend matching is needed.
pde[0]['interfacetype'] = 1

# 8 x 16 quadrilaterals: interface faces of length 1/16.
mesh[0]['p'], mesh[0]['t'] = squaremesh(8, 16, 1, 1)
mesh[0]['p'][0, :] = 0.5 * mesh[0]['p'][0, :]
mesh[0]['boundaryexpr'] = [lambda p: abs(p[1, :]) < 1e-8, lambda p: abs(p[0, :] - 0.5) < 1e-8,
                           lambda p: abs(p[1, :] - 1) < 1e-8, lambda p: abs(p[0, :]) < 1e-8]
mesh[0]['boundarycondition'] = numpy.array([1, 2, 1, 1])
mesh[0]['interfacecondition'] = numpy.array([0, 1, 0, 0])   # boundary 2 is the coupled interface
pde[0]['interfacefluxmap'] = numpy.array([1])
