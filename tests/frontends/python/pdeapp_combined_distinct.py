# Combined multi-PDE regression for the slot -> model-ID mapping: TWO generated
# models in ONE exasimapp where slot 1's PDE differs from built-in model 1.
# Slot 0 is the standard Poisson model (mu = 1). Slot 1 uses
# pdemodel_combined_distinct (source scaled by mu) with mu = 2; its exact
# solution is still sin(pi*x)*sin(pi*y), but built-in model 1 would give half
# of it (L2 error^2 ~ 6e-2). Both slots' QoIs must therefore be at
# discretization-error level; slot 1 fails if its kernels are not model 101's.
import os
import numpy
import exasim


def make(modelfile, mu):
    pde, mesh = exasim.initializeexasim()
    pde['model'] = "ModelD"
    pde['modelfile'] = modelfile
    pde['mpiprocs'] = 1
    pde['hybrid'] = 1
    pde['porder'] = 3
    pde['physicsparam'] = numpy.array([mu])
    pde['tau'] = numpy.array([1.0])
    mesh['p'], mesh['t'] = exasim.Mesh.SquareMesh(16, 16, 1)[0:2]
    mesh['boundaryexpr'] = [lambda p: (p[1, :] < 1e-3), lambda p: (p[0, :] > 1 - 1e-3),
                            lambda p: (p[1, :] > 1 - 1e-3), lambda p: (p[0, :] < 1e-3)]
    mesh['boundarycondition'] = numpy.array([1, 1, 1, 1])
    return pde, mesh


pde0, mesh0 = make("pdemodel", 1.0)
pde1, mesh1 = make("pdemodel_combined_distinct", 2.0)

sol, pde, mesh, master, dmd, cstr, rstr, res = exasim.exasim([pde0, pde1], [mesh0, mesh1])

assert pde[0]['modelid'] == 100 and pde[1]['modelid'] == 101, \
    f"expected modelids [100, 101], got {[pde[0]['modelid'], pde[1]['modelid']]}"

tol = float(os.environ.get("QOI_TOL", "1e-8"))
for m, sub in ((0, ""), (1, "1")):
    qoifile = os.path.join(pde[m]['datapath'], "dataout", sub, "outqoi.txt")
    err2 = float(open(qoifile).readlines()[-1].split()[1])
    print(f"slot {m} (id {pde[m]['modelid']}): L2 error^2 = {err2:.6e} (tol {tol:.1e})")
    assert err2 < tol, f"slot {m} QoI gate failed: {err2} >= {tol}"

print("COMBINED DISTINCT MULTI-PDE TEST PASSED")
