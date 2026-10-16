"""Shared Python orchestration for the two ISOQ conjugate-heat-transfer cases."""

import os
import numpy as np
import exasim
from exasim.Postprocessing.getsolution import getsolution
from isoq_solid_mesh import make_isoq_solid


def run_isoq_case(build_fluid, matching, case_directory, run_directory=None):
    run_directory=run_directory or os.path.join(case_directory,"python_run")
    precursor=os.path.join(run_directory,"precursor")
    fluid,fluid_mesh=build_fluid(precursor)
    fluid_solution,fluid,fluid_mesh,fluid_master,fluid_dmd=exasim.exasim(fluid,fluid_mesh)[0:5]
    fluid_mesh["udg"]=fluid_solution
    vdg_path=os.path.join(precursor,"dataout","outvdg")
    if os.path.isfile(vdg_path+"_np0.bin"):
        fluid_mesh["vdg"]=getsolution(vdg_path,fluid_dmd,fluid_master["npe"])

    coupled=os.path.join(run_directory,"coupled"); os.makedirs(coupled,exist_ok=True)
    fluid["datapath"]=coupled; fluid["builddir"]=os.path.join(coupled,".exasim"); fluid["buildpath"]=fluid["builddir"]
    # The fluid uses quadrilaterals and the solid uses triangles, so both the
    # coincident and independently meshed interfaces require the type-1 path.
    fluid["modelnumber"]=0; fluid["interfacetype"]=1
    fluid_mesh["boundarycondition"]=np.array([4,2,1,7]); fluid_mesh["interfacecondition"]=np.array([0,0,0,1])
    fluid["interfacefluxmap"]=np.array([4]); fluid["ibs"]=7; fluid["distanceboundaryconditions"]=np.array([7])
    fluid["AVdistfunction"]=0; fluid["AVsmoothingMethod"]=0; fluid["AV"]=0
    params=np.asarray(fluid["physicsparam"]).reshape(-1); params[-2:]=[0.0,1.5*fluid["AVcoeffEnd"]]; fluid["physicsparam"]=params
    fluid["NLiter"]=20

    solid,_=exasim.initializeexasim(); solid.update({"model":"ModelD","modelfile":"isoq_solid_model","platform":fluid["platform"],
        "mpiprocs":2,"hybrid":1,"debugmode":fluid["debugmode"],"porder":fluid["porder"],"pgauss":fluid["pgauss"],
        "gencode":1,"Cxxpreprocessing":0,"GMRESrestart":fluid["GMRESrestart"],"GMRESortho":fluid["GMRESortho"],
        "linearsolvertol":fluid["linearsolvertol"],"linearsolveriter":fluid["linearsolveriter"],"preconditioner":fluid["preconditioner"],
        "RBdim":fluid["RBdim"],"ppdegree":fluid["ppdegree"],"NLtol":fluid["NLtol"],"NLiter":20,"matvectol":fluid["matvectol"],
        "modelnumber":1,"interfacetype":1,"interfacefluxmap":np.array([1]),"datapath":coupled,
        "builddir":os.path.join(coupled,".exasim"),"buildpath":os.path.join(coupled,".exasim")})
    gam,mach,tref,twall=params[0],params[3],params[9],params[10]; tinf=1.0/(gam*(gam-1.0)*mach**2)
    solid_temperature=twall/tref*tinf; solid["physicsparam"]=np.array([0.01,solid_temperature]); solid["tau"]=np.array([1.0])
    solid_mesh=make_isoq_solid(solid,fluid_mesh,matching,0.0028)
    solid_mesh["udg"]=np.zeros((solid_mesh["dgnodes"].shape[0],3,solid_mesh["dgnodes"].shape[2]),order="F"); solid_mesh["udg"][:,0,:]=solid_temperature
    return exasim.exasim([fluid,solid],[fluid_mesh,solid_mesh])
