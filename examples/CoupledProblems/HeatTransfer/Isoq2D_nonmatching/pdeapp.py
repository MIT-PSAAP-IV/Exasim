"""Python version of the axisymmetric ideal-gas ISOQ mesh-adaptivity case."""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPOSITORY = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
SOURCE_INSTALL = os.path.join(os.path.dirname(REPOSITORY), "exasim_install")
CANONICAL_KERNELS = os.path.join(HERE, "canonical_kernels")
if "EXASIM_PREFIX" not in os.environ and os.path.isfile(
    os.path.join(SOURCE_INSTALL, "lib", "cmake", "Exasim", "ExasimConfig.cmake")
):
    os.environ["EXASIM_PREFIX"] = SOURCE_INSTALL
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(REPOSITORY, "frontends", "Python"))

import exasim  # noqa: E402
from exasim.Postprocessing.getsolution import getsolution  # noqa: E402
from isoq_mesh import load_canonical_mesh  # noqa: E402


def build_case(run_directory=None):
    pde, _ = exasim.initializeexasim()
    pde["model"] = "ModelD"
    # All frontends compile the same checked-in generated model sources.
    pde["codegenerator"] = "prebuilt"
    pde["kernelsource"] = os.path.join(CANONICAL_KERNELS, "fluid")
    pde["canonicalmaster"] = os.path.join(HERE, "canonical_mesh", "fluid_master.bin")
    pde["modelfile"] = "pdemodel"
    pde["platform"] = "cpu"
    pde["mpiprocs"] = 4
    pde["porder"] = 2
    pde["pgauss"] = 2 * pde["porder"]
    pde["hybrid"] = 1
    pde["debugmode"] = 0
    pde["saveParaview"] = 1
    pde["saveSolBouFreq"] = 1
    pde["ibs"] = 3
    pde["saveSolBouLoc"] = 1

    run_directory = run_directory or os.path.join(HERE, "python_run")
    os.makedirs(run_directory, exist_ok=True)
    pde["datapath"] = run_directory
    pde["builddir"] = os.path.join(run_directory, ".exasim")
    pde["buildpath"] = pde["builddir"]

    gam, reynolds, prandtl, mach_inf = 1.4, 1.56e5, 0.71, 7.6
    reference_temperature, wall_temperature = 266.5, 300.0
    pressure_inf = 1.0 / (gam * mach_inf**2)
    temperature_inf = pressure_inf / (gam - 1.0)
    density_inf = 1.0
    momentum_x_inf, momentum_y_inf = 1.0, 0.0
    energy_inf = 0.5 + pressure_inf / (gam - 1.0)

    pde["AV"] = 2
    pde["AVcontinuationIter"] = 8
    pde["AVcontinuationLogScale"] = 1.5
    pde["AVcoeffStart"] = 0.002
    pde["AVcoeffEnd"] = 0.00002
    # Canonical continuation schedule shared by MATLAB, Python, and Julia.
    pde["avparam1"] = np.array([
        0.002, 0.0015034374019939735, 0.0011026529166442065,
        0.0007791726457939881, 0.000518085978384667,
        0.0003073582956121101, 0.00013727624306105775, 0.0,
    ])
    pde["avparam2"] = np.array([
        0.0, 1.3727624306105766e-06, 3.0735829561211013e-06,
        5.180859783846669e-06, 7.791726457939882e-06,
        1.1026529166442065e-05, 1.5034374019939733e-05, 2.0e-05,
    ])
    pde["AVdistfunction"] = 1
    pde["distanceboundaryconditions"] = np.array([3], dtype=np.int64)
    pde["AVsmoothingMethod"] = 1
    pde["AVHelmholtzCoeff"] = 0.0001
    av_max_divergence, av_distance_coefficient = 20.0, 100.0

    pde["meshadaptenabled"] = 0
    pde["meshadaptfield"] = 2
    pde["meshadaptavcomponent"] = 1
    pde["meshadaptalpha"] = 0.5
    pde["meshadaptqmin"] = 0.2
    pde["meshadaptqmax"] = 0.8
    pde["meshadaptHelmholtzCoeff"] = 0.001
    pde["meshadaptforcescale"] = 0.25
    pde["meshadaptsmoothingpasses"] = 30
    pde["meshadaptboundaryconditions"] = np.array([3, 3, 3, 2], dtype=np.int64)

    pde["physicsparam"] = np.array(
        [
            gam,
            reynolds,
            prandtl,
            mach_inf,
            density_inf,
            momentum_x_inf,
            momentum_y_inf,
            energy_inf,
            temperature_inf,
            reference_temperature,
            wall_temperature,
            av_max_divergence,
            av_distance_coefficient,
            pde["AVcoeffStart"],
            pde["AVcoeffEnd"],
        ]
    )
    pde["tau"] = np.array([4.0])
    pde["GMRESrestart"] = 500
    pde["GMRESortho"] = 1
    pde["linearsolvertol"] = 1.0e-6
    pde["linearsolveriter"] = 1000
    pde["preconditioner"] = 1
    pde["RBdim"] = 0
    pde["ppdegree"] = 0
    pde["NLtol"] = 1.0e-6
    pde["NLiter"] = 10
    pde["matvectol"] = 1.0e-6

    # Keep the optional serialized extension blocks identical to MATLAB.
    for name in ("flag", "problem", "factor", "solversparam", "stgdata",
                 "stgparam", "stgib", "dae_dt"):
        pde[name] = np.array([], dtype=np.float64)

    mesh = load_canonical_mesh("fluid")
    mesh["boundarycondition"] = np.array([4, 2, 1, 3], dtype=np.int64)
    return pde, mesh




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
    fluid["AVcontinuationIter"]=0; fluid["avparam1"]=np.array([]); fluid["avparam2"]=np.array([])
    params=np.asarray(fluid["physicsparam"]).reshape(-1); params[-2:]=[0.0,fluid["AVcoeffEnd"]]; fluid["physicsparam"]=params
    # The monolithic solve inherits the precursor's nonlinear stopping rule.

    solid,_=exasim.initializeexasim(); solid.update({"model":"ModelD","codegenerator":"prebuilt",
        "kernelsource":os.path.join(CANONICAL_KERNELS,"solid"),
        "canonicalmaster":os.path.join(HERE,"canonical_mesh","solid_master.bin"),
        "modelfile":"pdemodel_solid","platform":fluid["platform"],
        "mpiprocs":2,"hybrid":1,"debugmode":fluid["debugmode"],"porder":fluid["porder"],"pgauss":fluid["pgauss"],
        "gencode":1,"Cxxpreprocessing":1,"GMRESrestart":fluid["GMRESrestart"],"GMRESortho":fluid["GMRESortho"],
        "linearsolvertol":fluid["linearsolvertol"],"linearsolveriter":fluid["linearsolveriter"],"preconditioner":fluid["preconditioner"],
        "RBdim":fluid["RBdim"],"ppdegree":fluid["ppdegree"],"NLtol":fluid["NLtol"],"NLiter":fluid["NLiter"],"matvectol":fluid["matvectol"],
        "modelnumber":1,"interfacetype":1,"interfacefluxmap":np.array([1]),"datapath":coupled,
        "builddir":os.path.join(coupled,".exasim"),"buildpath":os.path.join(coupled,".exasim")})
    for name in ("flag", "problem", "factor", "solversparam", "stgdata",
                 "stgparam", "stgib", "dae_dt"):
        solid[name] = np.array([], dtype=np.float64)
    gam,mach,tref,twall=params[0],params[3],params[9],params[10]; tinf=1.0/(gam*(gam-1.0)*mach**2)
    solid_temperature=twall/tref*tinf; solid["physicsparam"]=np.array([0.01,solid_temperature]); solid["tau"]=np.array([1.0])
    solid_mesh=load_canonical_mesh("solid")
    solid_mesh["boundarycondition"]=np.array([3,2,1],dtype=np.int64)
    return exasim.exasim([fluid,solid],[fluid_mesh,solid_mesh])


def main():
    return run_isoq_case(build_case, False, HERE)


if __name__ == "__main__":
    main()
