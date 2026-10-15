# Model 2 of the mixed triangle/quadrilateral Poisson example.
# Executed by pdeapp.py, which defines pde, mesh, exasim, numpy and squaremesh.
pde[1], mesh[1] = exasim.initializeexasim()

pde[1]['model'] = "ModelD"
pde[1]['modelfile'] = "pdemodel2"
pde[1]['Cxxpreprocessing'] = 1
pde[1]['mpiprocs'] = 1
pde[1]['hybrid'] = 1
pde[1]['porder'] = 3
pde[1]['physicsparam'] = numpy.array([1.0])
pde[1]['tau'] = numpy.array([1.0])
pde[1]['NLiter'] = 3
pde[1]['NLtol'] = 1e-10
pde[1]['linearsolvertol'] = 1e-10
pde[1]['matvectol'] = 1e-10
pde[1]['GMRESrestart'] = 300
pde[1]['linearsolveriter'] = 1000
pde[1]['interfacetype'] = 1

triangleOnModel1 = globals().get('triangleOnModel1', True)
elemtype = 1 if triangleOnModel1 else 0
mesh[1]['p'], mesh[1]['t'] = squaremesh(12, 24, 1, elemtype)
mesh[1]['p'][0, :] = 0.5 * mesh[1]['p'][0, :] + 0.5
mesh[1]['boundaryexpr'] = [lambda p: abs(p[1, :]) < 1e-8, lambda p: abs(p[0, :] - 1) < 1e-8,
                           lambda p: abs(p[1, :] - 1) < 1e-8, lambda p: abs(p[0, :] - 0.5) < 1e-8]
mesh[1]['boundarycondition'] = numpy.array([1, 1, 1, 2])
mesh[1]['interfacecondition'] = numpy.array([0, 0, 0, 1])
pde[1]['interfacefluxmap'] = numpy.array([1])
