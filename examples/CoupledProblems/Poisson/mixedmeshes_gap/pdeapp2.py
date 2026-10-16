# Model 2: [0.5+gap/2,1] x [0,1].
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
gap = globals().get('gap', 0.02)
elemtype = 1 if triangleOnModel1 else 0
mesh[1]['p'], mesh[1]['t'] = squaremesh(12, 24, 1, elemtype)
interface = 0.5 + gap / 2
mesh[1]['p'][0, :] = (1 - interface) * mesh[1]['p'][0, :] + interface
# x0=interface binds the current value of interface.
mesh[1]['boundaryexpr'] = [lambda p: abs(p[1, :]) < 1e-8, lambda p: abs(p[0, :] - 1) < 1e-8,
                           lambda p: abs(p[1, :] - 1) < 1e-8, lambda p, x0=interface: abs(p[0, :] - x0) < 1e-8]
mesh[1]['boundarycondition'] = numpy.array([1, 1, 1, 2])
mesh[1]['interfacecondition'] = numpy.array([0, 0, 0, 1])
pde[1]['interfacefluxmap'] = numpy.array([1])
