# Export this Poisson 2D application as a Text2Code package.
#
# Usage from this directory:
#
#   python pdeapp_exporttext2code.py
#   cd text2code_package
#   export EXASIM_PREFIX=/path/to/exasim-prefix
#   /path/to/exasim-prefix/bin/text2code pdeapp.txt --out-dir generated
#
# The exported package contains pdemodel.txt, pdeapp.txt, grid.bin, xdg.bin,
# udg.bin, vdg.bin, and wdg.bin. The vdg/wdg fields are included here to
# demonstrate optional field export; this Poisson model does not use them in
# the governing equation.
import os

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

dest = os.path.join(os.getcwd(), "text2code_package")
exasim.exporttext2code(pde, mesh, dest)
print(f"Text2Code package written to {dest}")
