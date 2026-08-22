"""Export the Supersonic Taylor-Green vortex as a Text2Code app.

This is the Python equivalent of ``pdeapp_text2code.m``. It writes the
Text2Code package to ``apps/navierstokes/supersonicTGV`` without running the
solver.
"""

from pathlib import Path
import sys

import numpy

exasimroot = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(exasimroot / "frontends" / "Python"))

import exasim


def main():
    pde, mesh = exasim.initializeexasim()

    pde["model"] = "ModelD"
    pde["modelfile"] = "pdemodel"
    pde["ncu"] = 5
    pde["nco"] = 1

    pde["mpiprocs"] = 12
    pde["hybrid"] = 1

    pde["porder"] = 2
    pde["pgauss"] = 2 * pde["porder"]
    pde["torder"] = 3
    pde["nstage"] = 3

    deltat = 1.0e-2
    nsteps = round(20.0 / deltat)
    pde["dt"] = deltat * numpy.ones(nsteps)
    pde["saveSolFreq"] = 20
    pde["saveSolOpt"] = 0

    nspatial = 16
    gam = 1.4
    Re = 1600.0
    Pr = 0.71
    Minf = 1.25
    rhoRef = 1.0
    hm = 2.0 * numpy.pi / nspatial
    avcoeff = 2.0e-3
    pde["physicsparam"] = numpy.array([gam, Re, Pr, Minf, rhoRef, hm, avcoeff, pde["porder"]])

    pde["tau"] = numpy.array([5.0])
    pde["GMRESortho"] = 1
    pde["GMRESrestart"] = 24
    pde["linearsolvertol"] = 1.0e-7
    pde["linearsolveriter"] = 24
    pde["preconditioner"] = 1
    pde["precMatrixType"] = 2
    pde["NLiter"] = 1
    pde["NLtol"] = 1.0e-8
    pde["ppdegree"] = 0
    pde["RBdim"] = 5
    pde["gencode"] = 1

    pde["AV"] = 1
    pde["frozenAVflag"] = 1
    pde["AVsmoothingIter"] = 2

    mesh["p"], mesh["t"] = exasim.Mesh.cubemesh(nspatial, nspatial, nspatial, 1)[0:2]
    mesh["p"] = 2.0 * numpy.pi * mesh["p"]
    mesh["t"] = mesh["t"] + 1

    mesh["boundaryexpr"] = [
        "abs(y)<1e-8",
        "abs(x-2*pi)<1e-8",
        "abs(y-2*pi)<1e-8",
        "abs(x)<1e-8",
        "abs(z)<1e-8",
        "abs(z-2*pi)<1e-8",
    ]
    mesh["boundarycondition"] = numpy.array([1, 1, 1, 1, 1, 1])

    # Python exporttext2code consumes mesh["periodicboundary"] and expands
    # xy/xz/yz into the text2code coordinate lists used by the MATLAB exporter.
    mesh["periodicboundary"] = [
        [2, "yz", 4, "yz"],
        [1, "xz", 3, "xz"],
        [5, "xy", 6, "xy"],
    ]

    exportdir = exasimroot / "apps" / "navierstokes" / "supersonicTGV"
    exasim.exporttext2code(pde, mesh, exportdir)


if __name__ == "__main__":
    main()
