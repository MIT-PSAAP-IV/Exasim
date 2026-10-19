import numpy

import exasim


pde, mesh = exasim.initializeexasim()

pde["model"] = "ModelD"
pde["modelfile"] = "pdemodel"
pde["mpiprocs"] = 1
pde["hybrid"] = 0
pde["Cxxpreprocessing"] = 0

pde["porder"] = 3
pde["physicsparam"] = numpy.array([1.0, 2.0 * numpy.pi])
pde["tau"] = numpy.array([2.0 * numpy.pi])

# Python's linemesh argument is the number of points, so 17 gives the same
# 16-element unit-interval mesh as linemesh(16) in MATLAB.
mesh["p"], mesh["t"] = exasim.Mesh.linemesh(17)
mesh["boundaryexpr"] = [
    lambda p: numpy.abs(p[0, :] - 1.0) < 1.0e-8,
    lambda p: numpy.abs(p[0, :]) < 1.0e-8,
]
mesh["boundarycondition"] = numpy.array([1, 1])

sol, pde, mesh = exasim.exasim(pde, mesh)[0:3]

pde["visscalars"] = ["temperature", 0]
pde["visvectors"] = []
exasim.vis(sol, pde, mesh)
print("Done!")
