"""Python coupled Mach-8 cylinder heat-transfer example (non-matching meshes).

Model 1 is the fluid (pdemodel1.py), model 2 the solid (pdemodel2.py). A fluid
precursor with artificial-viscosity continuation is solved first; the coupled
fluid/solid solve then starts from it.
Run with `python3 pdeapp.py`; data and the solver build go to ./python_run.
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPOSITORY = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
# The frontend imports pdemodel1.py and pdemodel2.py by module name.
sys.path[0:0] = [HERE, os.path.join(REPOSITORY, "frontends", "Python")]
import exasim  # noqa: E402
from exasim.Preprocessing.createdgnodes import createdgnodes  # noqa: E402
from exasim.Preprocessing.facenumbering import facenumbering  # noqa: E402
from exasim.Preprocessing.masternodes import masternodes  # noqa: E402

CASE = "nonmatching"


def _stretch(values, alpha, increasing):
    lo, hi = np.min(values), np.max(values)
    s = (values - lo) / (hi - lo)
    if increasing:
        mapped = (np.exp(alpha * s) - 1.0) / (np.exp(alpha) - 1.0)
    else:
        mapped = (1.0 - np.exp(-alpha * s)) / (1.0 - np.exp(-alpha))
    return lo + (hi - lo) * mapped


def _logical_mesh(nx, ny, elemtype):
    x = np.tile(np.linspace(0.0, 1.0, nx + 1), ny + 1)
    y = np.repeat(np.linspace(0.0, 1.0, ny + 1), nx + 1)
    p = np.vstack((x, y))
    cells = []
    for iy in range(ny):
        for ix in range(nx):
            ll = ix + iy * (nx + 1)
            if elemtype == 1:
                cells.append([ll, ll + 1, ll + nx + 2, ll + nx + 1])
            else:
                # Match mkmesh_square(..., parity=1): NW-SE diagonal.
                cells.extend(([ll, ll + 1, ll + nx + 1], [ll + 1, ll + nx + 2, ll + nx + 1]))
    return p, np.asarray(cells, dtype=np.int64).T


def _half_circle(points, a, b, c):
    theta = 1.5 * np.pi - np.pi * points[1, ...]
    outer = -b * np.cos(theta) + c * (1.0 + np.cos(theta))
    radius = outer + points[0, ...] * (a - outer)
    return np.stack((radius * np.cos(theta), radius * np.sin(theta)), axis=0)


def make_cylinder_mesh(porder, nx, ny, elemtype, solid=False):
    p, t = _logical_mesh(nx, ny, elemtype)
    nfe = 4 if elemtype == 1 else 3
    dgnodes = createdgnodes(p, t, np.zeros((nfe, t.shape[1]), dtype=np.int64), [], [], porder)
    p[0, :] = _stretch(p[0, :], 2.0 if solid else 5.0, solid)
    dgnodes[:, 0, :] = _stretch(dgnodes[:, 0, :], 2.0 if solid else 5.0, solid)
    if solid:
        map_args = (0.85, 1.0, 1.0)
    else:
        map_args = (1.0, 3.0, 4.7)
    p = _half_circle(p, *map_args)
    dgnodes = np.transpose(_half_circle(np.transpose(dgnodes, (1, 0, 2)), *map_args), (1, 0, 2))
    if solid:
        boundaryexpr = [
            lambda z: np.sqrt(z[0, :] ** 2 + z[1, :] ** 2) > 1.0 - 1.0e-6,
            lambda z: z[0, :] > -1.0e-7,
            lambda z: np.abs(z[0, :]) < 20.0,
        ]
    else:
        boundaryexpr = [
            lambda z: np.sqrt(z[0, :] ** 2 + z[1, :] ** 2) < 1.0 + 1.0e-6,
            lambda z: z[0, :] > -1.0e-7,
            lambda z: np.abs(z[0, :]) < 20.0,
        ]
    f = facenumbering(p, t, elemtype, boundaryexpr, [])[0]
    return {
        "p": p, "t": t, "f": f, "dgnodes": dgnodes,
        "udg": [], "vdg": [], "wdg": [], "tprd": [],
        "boundaryexpr": boundaryexpr, "boundarycondition": np.array([3, 2, 1]),
        "periodicexpr": [], "curvedboundary": np.array([], dtype=np.int64),
        "curvedboundaryexpr": [], "periodicboundary": np.array([], dtype=np.int64),
    }


def wall_distance(mesh, porder, elemtype):
    perm = np.asarray(masternodes(porder, 2, elemtype)[4], dtype=np.int64) - 1
    lf, elem = np.nonzero(mesh["f"] == 1)
    wall = np.concatenate([mesh["dgnodes"][perm[:, face], :, e] for face, e in zip(lf, elem)])
    nodes = mesh["dgnodes"]
    distance = np.empty((nodes.shape[0], 1, nodes.shape[2]), order="F")
    for e in range(nodes.shape[2]):
        delta = nodes[:, :, e, None] - wall.T[None, :, :]
        distance[:, 0, e] = np.sqrt(np.sum(delta * delta, axis=1)).min(axis=1)
    return distance



# Artificial-viscosity continuation [c, k] for av = c*tanh(k*d), as in pdeapp1.m.
AV_CONTINUATION = ((0.06,30.0),(0.04,30.0),(0.025,30.0),(0.025,5.0))


def _paths(pde, run_directory):
    pde["datapath"] = run_directory
    pde["builddir"] = os.path.join(run_directory, ".exasim")
    pde["buildpath"] = pde["builddir"]


def _fluid(case, run_directory):
    pde, _ = exasim.initializeexasim(); _paths(pde, run_directory)
    pde.update({"model":"ModelD","modelfile":"pdemodel1","platform":"cpu",
                "mpiprocs":2,"hybrid":1,"porder":4,"pgauss":8,"gencode":1,
                "Cxxpreprocessing":0,"GMRESrestart":100,"linearsolvertol":1e-8,
                "linearsolveriter":100,"RBdim":0,"ppdegree":20,"NLtol":1e-6,
                "NLiter":30,"matvectol":1e-6})
    gam, re, pr, mach, tref, twall = 1.4, 1.835e5, 0.71, 8.03, 124.49, 294.44
    pinf = 1.0/(gam*mach**2); tinf = pinf/(gam-1.0)
    base = [gam,re,pr,mach,1.0,1.0,0.0,0.5+pinf/(gam-1.0),tinf,tref,twall]
    # AV continuation av = c*tanh(k*d): one sweep case per [c, k], each warm-started
    pde["physicsparamsweep"] = np.array([base + list(ck) for ck in AV_CONTINUATION])
    pde["physicsparamwarmstart"] = 1
    pde["physicsparam"] = pde["physicsparamsweep"][0].copy()
    pde["tau"] = np.array([1.0])
    ny = 21 if case == "matching" else 28
    mesh = make_cylinder_mesh(4,31,ny,1,solid=False)
    mesh["boundarycondition"] = np.array([3,2,1])
    distance = wall_distance(mesh,4,1)
    mesh["vdg"] = np.zeros((distance.shape[0],2,distance.shape[2]),order="F")
    mesh["vdg"][:,0,:] = distance[:,0,:]
    npe,_,ne = mesh["dgnodes"].shape
    udg=np.zeros((npe,4,ne),order="F"); udg[:,0,:]=1.0
    udg[:,1,:]=np.tanh(10.0*distance[:,0,:])
    tnear=tinf*(twall/tref-1.0)*np.exp(-10.0*distance[:,0,:])+tinf
    udg[:,3,:]=tnear+0.5*udg[:,1,:]**2; mesh["udg"]=udg
    return pde,mesh,distance


def _continue_fluid(pde,mesh,distance):
    solcases,pde,mesh = exasim.exasim(pde,mesh)[0:3]
    # The coupled solve starts from the last continuation case and keeps its AV.
    sol = solcases[-1]
    if sol.ndim == 4:
        sol = sol[:,:,:,-1]
    pde["physicsparam"] = pde["physicsparamsweep"][-1].copy()
    pde["physicsparamsweep"] = []
    pde["physicsparamwarmstart"] = 0
    mesh["udg"]=sol
    return pde,mesh,sol


def _solid(case, fluid_pde, run_directory):
    pde,_=exasim.initializeexasim(); _paths(pde,run_directory)
    pde.update({"model":"ModelD","modelfile":"pdemodel2","platform":"cpu",
                "mpiprocs":2,"hybrid":1,"porder":4,"pgauss":8,"gencode":1,
                "Cxxpreprocessing":0,"GMRESrestart":100,"linearsolvertol":1e-8,
                "linearsolveriter":100,"RBdim":0,"ppdegree":20,"NLtol":1e-6,
                "NLiter":30,"matvectol":1e-6})
    tinf=fluid_pde["physicsparam"][8]; solid_temperature=294.44/124.49*tinf
    pde["physicsparam"]=np.array([0.2,solid_temperature]); pde["tau"]=np.array([1.0])
    elemtype,ny=(0,11) if case=="quad_tri" else (1,21)
    mesh=make_cylinder_mesh(4,5,ny,elemtype,solid=True)
    mesh["udg"]=np.zeros((mesh["dgnodes"].shape[0],3,mesh["dgnodes"].shape[2]),order="F")
    mesh["udg"][:,0,:]=solid_temperature
    return pde,mesh


def run_case(case, case_directory, run_directory=None):
    if case not in ("matching","nonmatching","quad_tri"):
        raise ValueError("unknown cylinder heat-transfer case")
    run_directory=run_directory or os.path.join(case_directory,"python_run")
    os.makedirs(run_directory,exist_ok=True)
    fluid,fluid_mesh,distance=_fluid(case,os.path.join(run_directory,"precursor"))
    fluid,fluid_mesh,_=_continue_fluid(fluid,fluid_mesh,distance)
    solid,solid_mesh=_solid(case,fluid,os.path.join(run_directory,"coupled"))
    # Both models must use one build/data root for the combined executable.
    _paths(fluid,os.path.join(run_directory,"coupled"))
    for model,pde in enumerate((fluid,solid)):
        pde["modelnumber"]=model
        pde["interfacetype"]=0 if case=="matching" else 1
    fluid_mesh["boundarycondition"]=np.array([5,2,1]); fluid_mesh["interfacecondition"]=np.array([1,0,0]); fluid["interfacefluxmap"]=np.array([4])
    solid_mesh["boundarycondition"]=np.array([3,2,1]); solid_mesh["interfacecondition"]=np.array([1,0,0]); solid["interfacefluxmap"]=np.array([1])
    result=exasim.exasim([fluid,solid],[fluid_mesh,solid_mesh])
    return result


def main():
    return run_case(CASE, HERE)


if __name__ == "__main__":
    main()
