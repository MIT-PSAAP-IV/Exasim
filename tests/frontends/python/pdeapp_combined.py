# Combined multi-PDE test: TWO generated models linked into ONE exasimapp and
# run together via the solver's multi-model path. exasim() is given a LIST of
# models; each occupies a slot (modelnumber 0,1 -> modelid 100,101), generates
# into its own kernel dir, and the single provider dispatches both ids. The run
# emits one (datain,dataout) pair per model; both QoIs must be correct.
import os
import numpy
import exasim


def make():
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
    return pde, mesh


pde0, mesh0 = make()
pde1, mesh1 = make()

# In export mode, also package a relocatable combined data-transfer bundle.
if os.environ.get("COMBINED_EXPORT") == "1":
    pde0['exportapp'] = os.path.join(os.getcwd(), "bundle")

sol, pde, mesh, master, dmd, cstr, rstr, res = exasim.exasim([pde0, pde1], [mesh0, mesh1])

assert pde[0]['modelid'] == 100 and pde[1]['modelid'] == 101, \
    f"expected modelids [100, 101], got {[pde[0]['modelid'], pde[1]['modelid']]}"

if pde[0].get('exportapp'):
    # export mode: no local run, so no per-model main-dir QoI to gate here; the
    # harness validates the exported combined bundle by relocating + running it.
    print("COMBINED MULTI-PDE TEST (export mode): skipped in-app QoI gate")
else:
    tol = float(os.environ.get("QOI_TOL", "1e-8"))
    for m, sub in ((0, ""), (1, "1")):
        qoifile = os.path.join(pde[m]['datapath'], "dataout", sub, "outqoi.txt")
        err2 = float(open(qoifile).readlines()[-1].split()[1])
        print(f"slot {m} (id {pde[m]['modelid']}): L2 error^2 = {err2:.6e} (tol {tol:.1e})")
        assert err2 < tol, f"slot {m} QoI gate failed: {err2} >= {tol}"

    print("COMBINED MULTI-PDE TEST PASSED")
