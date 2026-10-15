"""Shared Python driver for the three Mach-8 conjugate-heat-transfer cases."""

from __future__ import annotations
import os
import numpy as np
import exasim
from cylinder_mesh import make_cylinder_mesh, wall_distance


# Artificial-viscosity continuation [c, k] for av = c*tanh(k*d), as in pdeapp1.m.
AV_CONTINUATION = ((0.06,30.0),(0.04,30.0),(0.025,30.0),(0.025,5.0))


def _paths(pde, run_directory):
    pde["datapath"] = run_directory
    pde["builddir"] = os.path.join(run_directory, ".exasim")
    pde["buildpath"] = pde["builddir"]


def _fluid(case, run_directory):
    pde, _ = exasim.initializeexasim(); _paths(pde, run_directory)
    pde.update({"model":"ModelD","modelfile":"cylinder_fluid_model","platform":"cpu",
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
    pde.update({"model":"ModelD","modelfile":"solid_heat_model","platform":"cpu",
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
