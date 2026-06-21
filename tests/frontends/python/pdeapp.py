# Frontend end-to-end test: Poisson 2D on the unit square (serial, HDG).
# Generates kernels with the exasim package, builds the solver against the
# installed Exasim via the external-model path, runs it, and gates the QoI
# (Domain_QoI1 = integral of (u - uexact)^2) against QOI_TOL.
import os
import numpy
import exasim

pde, mesh = exasim.initializeexasim()

pde['model'] = "ModelD"
pde['modelfile'] = "pdemodel"
pde['mpiprocs'] = 1
pde['hybrid'] = 1
pde['porder'] = 3
pde['physicsparam'] = numpy.array([1.0])
pde['tau'] = numpy.array([1.0])

mesh['p'], mesh['t'] = exasim.Mesh.SquareMesh(16, 16, 1)[0:2]
mesh['boundaryexpr'] = [lambda p: (p[1, :] < 1e-3), lambda p: (p[0, :] > 1 - 1e-3),
                        lambda p: (p[1, :] > 1 - 1e-3), lambda p: (p[0, :] < 1e-3)]
mesh['boundarycondition'] = numpy.array([1, 1, 1, 1])

sol, pde, mesh = exasim.exasim(pde, mesh)[0:3]

if pde.get('exportapp'):
    # export mode: exasim() packages the bundle without running the simulation
    # locally, so there is no main-dir QoI to gate here (the test harness
    # validates the exported bundle by relocating and running it).
    print("FRONTEND TEST (export mode): skipped in-app QoI gate")
else:
    qoifile = os.path.join(pde['datapath'], "dataout", "outqoi.txt")
    with open(qoifile) as f:
        last = f.readlines()[-1].split()
    err2 = float(last[1])
    tol = float(os.environ.get("QOI_TOL", "1e-8"))
    print(f"L2 error^2 = {err2:.6e} (tol {tol:.1e})")
    assert err2 < tol, f"QoI gate failed: {err2} >= {tol}"
    print("FRONTEND TEST PASSED")
