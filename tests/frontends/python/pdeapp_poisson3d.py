# Cheap frontend app-regression case: Poisson 3D (HDG) on the unit cube.
# Adds 3D coverage to CI (the other frontend cases are 2D SquareMesh only), so a
# regression in the 3D codegen / assembly path is caught. Small hex mesh (6^3) at
# porder 3 to stay cheap; manufactured solution sin(pi x)sin(pi y)sin(pi z) gives an
# analytic (u-uexact)^2 QoI gate (no golden baseline). 3D p3 on h=1/6 leaves a
# discretization error^2 ~ (1/6)^8 ~ 6e-7, so the harness runs this with QOI_TOL=1e-6
# (a real broken solve gives O(1) error and still fails loudly).
import os
import numpy
import exasim

pde, mesh = exasim.initializeexasim()

pde['model'] = "ModelD"
pde['modelfile'] = "pdemodel_poisson3d"
pde['mpiprocs'] = 1
pde['hybrid'] = 1
pde['porder'] = 3
pde['physicsparam'] = numpy.array([1.0])
pde['tau'] = numpy.array([1.0])

mesh['p'], mesh['t'] = exasim.Mesh.cubemesh(6, 6, 6, 1)[0:2]
# 6 cube faces: z=0, z=1, y=0, y=1, x=0, x=1 -- all Dirichlet (u=0 = uexact on ∂Ω).
mesh['boundaryexpr'] = [lambda p: (p[2, :] < 1e-3), lambda p: (p[2, :] > 1 - 1e-3),
                        lambda p: (p[1, :] < 1e-3), lambda p: (p[1, :] > 1 - 1e-3),
                        lambda p: (p[0, :] < 1e-3), lambda p: (p[0, :] > 1 - 1e-3)]
mesh['boundarycondition'] = numpy.array([1, 1, 1, 1, 1, 1])

sol, pde, mesh = exasim.exasim(pde, mesh)[0:3]

if pde.get('exportapp'):
    print("FRONTEND TEST (export mode): skipped in-app QoI gate")
else:
    qoifile = os.path.join(pde['datapath'], "dataout", "outqoi.txt")
    with open(qoifile) as f:
        last = f.readlines()[-1].split()
    err2 = float(last[1])
    tol = float(os.environ.get("QOI_TOL", "1e-6"))
    print(f"L2 error^2 = {err2:.6e} (tol {tol:.1e})")
    assert err2 < tol, f"QoI gate failed: {err2} >= {tol}"
    print("FRONTEND TEST PASSED (Poisson3D)")
