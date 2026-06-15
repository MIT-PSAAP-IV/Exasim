# Data-transfer app example (Poisson 2D, HDG).
#
# This is the standard Poisson2D frontend example with ONE extra line:
#
#     pde['exportapp'] = "<dest>"
#
# When set, exasim() additionally packages a self-contained, relocatable
# "data-transfer app" bundle at <dest>: the binary inputs (datain/, with
# boundary conditions already resolved), the generated kernels, a relocatable
# CMakeLists.txt + main.cpp, a generated text2code pdemodel.txt, a run.sh, and a
# manifest. The bundle can be copied to any machine that has an Exasim install
# and built + run there with no frontend involved:
#
#     EXASIM_ROOT=/path/to/exasim/install ./run.sh
#
# It is arch-independent — retarget the build machine's variant with, e.g.,
#     EXASIM_ROOT=/path/to/install EXASIM_VARIANT=gpu ./run.sh
#
# By default (build=True) the bundle is also built and run locally in a
# throwaway dir to verify it works before hand-off, leaving the bundle pristine.
import os
import numpy
import exasim

pde, mesh = exasim.initializeexasim()

# Define the PDE model: governing equations and boundary conditions
pde['model'] = "ModelD"
pde['modelfile'] = "pdemodel"

# Computing platform / parallelism
pde['mpiprocs'] = 1          # serial; bundle stays variant "cpu"
pde['hybrid'] = 1            # 0 -> LDG, 1 -> HDG

# Discretization / physics / solver parameters
pde['porder'] = 3
pde['physicsparam'] = numpy.array([1.0])     # unit thermal conductivity
pde['tau'] = numpy.array([1.0])              # DG stabilization parameter

# >>> The only line that distinguishes this from the plain Poisson2D example:
# package a relocatable data-transfer app bundle next to this script.
pde['exportapp'] = os.path.join(os.getcwd(), "poisson2d-bundle")

# Mesh: 16x16 quads on the unit square, with boundary expressions
mesh['p'], mesh['t'] = exasim.Mesh.SquareMesh(16, 16, 1)[0:2]
mesh['boundaryexpr'] = [lambda p: (p[1, :] < 1e-3), lambda p: (p[0, :] > 1 - 1e-3),
                        lambda p: (p[1, :] > 1 - 1e-3), lambda p: (p[0, :] < 1e-3)]
mesh['boundarycondition'] = numpy.array([1, 1, 1, 1])

# Generate + run the solver AND export the data-transfer app bundle.
sol, pde, mesh = exasim.exasim(pde, mesh)[0:3]

print("Done! Data-transfer app bundle: " + pde['exportapp'])
