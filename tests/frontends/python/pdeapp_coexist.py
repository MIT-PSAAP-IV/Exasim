# Coexistence test: build and run TWO independent generated models from the
# SAME working directory. Before per-model isolation they clobbered each other
# (shared .exasim/kernels, .exasim/build, datain/, dataout/, and both modelid
# 100). With isolation, model 0 keeps the flat layout (.exasim, datain,
# dataout, modelid 100) and model 1 nests (.exasim/models/1, datain/1,
# dataout/1, modelid 101). Both must build and produce a correct QoI.
import os
import numpy
import exasim


def poisson(modelnumber):
    pde, mesh = exasim.initializeexasim()
    pde['model'] = "ModelD"
    pde['modelfile'] = "pdemodel"
    pde['modelnumber'] = modelnumber            # 0 -> flat/id100, 1 -> models/1/id101
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
    assert pde['modelid'] == 100 + modelnumber, \
        f"model {modelnumber}: expected modelid {100 + modelnumber}, got {pde['modelid']}"
    return pde


tol = float(os.environ.get("QOI_TOL", "1e-8"))
for n, sub in ((0, ""), (1, "1")):
    pde = poisson(n)
    qoifile = os.path.join(pde['datapath'], "dataout", sub, "outqoi.txt")
    with open(qoifile) as f:
        err2 = float(f.readlines()[-1].split()[1])
    print(f"model {n} (id {pde['modelid']}): L2 error^2 = {err2:.6e} (tol {tol:.1e})")
    assert err2 < tol, f"model {n} QoI gate failed: {err2} >= {tol}"

print("COEXISTENCE TEST PASSED")
