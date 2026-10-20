# Model 1 of the mixed triangle/quadrilateral Poisson example.
# Executed by pdeapp.py, which defines pde, mesh, exasim, numpy and squaremesh.
pde[0], mesh[0] = exasim.initializeexasim()

pde[0]['model'] = "ModelD"
pde[0]['modelfile'] = "pdemodel1"
pde[0]['Cxxpreprocessing'] = 1
pde[0]['mpiprocs'] = 1
pde[0]['hybrid'] = 1
pde[0]['porder'] = 3
pde[0]['physicsparam'] = numpy.array([1.0])
pde[0]['tau'] = numpy.array([1.0])
pde[0]['NLiter'] = 3
pde[0]['NLtol'] = 1e-10
pde[0]['linearsolvertol'] = 1e-10
pde[0]['matvectol'] = 1e-10
pde[0]['GMRESrestart'] = 300
pde[0]['linearsolveriter'] = 1000
pde[0]['interfacetype'] = 1

triangleOnModel1 = globals().get('triangleOnModel1', True)
elemtype = 0 if triangleOnModel1 else 1
mesh[0]['p'], mesh[0]['t'] = squaremesh(8, 16, 1, elemtype)
mesh[0]['p'][0, :] = 0.5 * mesh[0]['p'][0, :]
mesh[0]['boundaryexpr'] = [lambda p: abs(p[1, :]) < 1e-8, lambda p: abs(p[0, :] - 0.5) < 1e-8,
                           lambda p: abs(p[1, :] - 1) < 1e-8, lambda p: abs(p[0, :]) < 1e-8]
mesh[0]['boundarycondition'] = numpy.array([1, 2, 1, 1])
mesh[0]['interfacecondition'] = numpy.array([0, 1, 0, 0])
pde[0]['interfacefluxmap'] = numpy.array([1])
