# Text2Code export fixture.
#
# This script exports a small Poisson application as a Text2Code package. It
# deliberately uses string boundary expressions because Text2Code input files
# cannot represent arbitrary Python lambda/function objects.
import os
from copy import deepcopy

import numpy
import exasim
from exasim.Preprocessing.createdgnodes import createdgnodes


pde, mesh = exasim.initializeexasim()

pde["model"] = "ModelD"
pde["modelfile"] = "pdemodel"
pde["mpiprocs"] = 1
pde["hybrid"] = 1
pde["porder"] = 1
pde["pgauss"] = 2
pde["physicsparam"] = numpy.array([1.0])
pde["physicsparamsweep"] = numpy.array([[1.0], [2.0]])
pde["tau"] = numpy.array([1.0])
# Exported to pdeapp.txt and applied by text2code (grid + xdg/udg/vdg/wdg are refined there).
pde["uniformrefinementlevel"] = 1
pde["AV"] = 1
pde["AVdistfunction"] = 1
pde["distanceboundaryconditions"] = numpy.array([1], dtype=int)
pde["AVsmoothingMethod"] = 1
pde["AVHelmholtzCoeff"] = 0.375
pde["avparam1"] = numpy.array([9.0, 8.0])
pde["avparam2"] = numpy.array([7.0, 6.0])
pde["AVcontinuationIter"] = 5
pde["AVcontinuationLogScale"] = 1.0
pde["AVcoeffStart"] = 0.06
pde["AVcoeffEnd"] = 0.015

# Include external and auxiliary field dimensions so the exported package
# exercises vdg.bin and wdg.bin in addition to mesh/grid and initial solution.
pde["nco"] = 1
pde["ncw"] = 1

mesh["p"], mesh["t"] = exasim.Mesh.SquareMesh(2, 2, 1)[0:2]
mesh["boundaryexpr"] = [
    "abs(y)<1e-8",
    "abs(x-1)<1e-8",
    "abs(y-1)<1e-8",
    "abs(x)<1e-8",
]
mesh["boundarycondition"] = numpy.array([1, 1, 1, 1])

mesh["dgnodes"] = createdgnodes(mesh["p"], mesh["t"], numpy.zeros((4, mesh["t"].shape[1])), [], [], pde["porder"])
npe = mesh["dgnodes"].shape[0]
ne = mesh["dgnodes"].shape[2]
mesh["udg"] = numpy.zeros((npe, pde["ncu"], ne))
mesh["vdg"] = numpy.ones((npe, pde["nco"], ne))
mesh["wdg"] = 2.0 * numpy.ones((npe, pde["ncw"], ne))

# Produce the same application through the ordinary frontend preprocessing
# path so the regression can compare its AV blocks with Text2Code's.
native_dir = os.path.join(os.getcwd(), "native_preprocessing")
pde_native = deepcopy(pde)
pde_native["datapath"] = native_dir
mesh_native = deepcopy(mesh)
mesh_native["boundaryexpr"] = [
    lambda p: p[1, :] < 1.0e-8,
    lambda p: p[0, :] > 1.0 - 1.0e-8,
    lambda p: p[1, :] > 1.0 - 1.0e-8,
    lambda p: p[0, :] < 1.0e-8,
]
exasim.preprocessing(pde_native, mesh_native)

dest = os.path.join(os.getcwd(), "text2code_package")
exasim.exporttext2code(pde, mesh, dest)
print(f"TEXT2CODE EXPORT PACKAGE: {dest}")
