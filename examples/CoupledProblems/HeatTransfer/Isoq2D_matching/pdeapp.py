"""Python version of the axisymmetric ideal-gas ISOQ mesh-adaptivity case."""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPOSITORY = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
SOURCE_INSTALL = os.path.join(os.path.dirname(REPOSITORY), "exasim_install")
if "EXASIM_PREFIX" not in os.environ and os.path.isfile(
    os.path.join(SOURCE_INSTALL, "lib", "cmake", "Exasim", "ExasimConfig.cmake")
):
    os.environ["EXASIM_PREFIX"] = SOURCE_INSTALL
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(REPOSITORY, "frontends", "Python"))

import exasim  # noqa: E402
from exasim.Postprocessing.getsolution import getsolution  # noqa: E402
from isoq_mesh import make_isoq_mesh, make_isoq_solid, wall_distance  # noqa: E402


def build_case(run_directory=None):
    pde, _ = exasim.initializeexasim()
    pde["model"] = "ModelD"
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

    mesh = make_isoq_mesh(pde["porder"], 5.0e-4)
    mesh["boundarycondition"] = np.array([4, 2, 1, 3], dtype=np.int64)
    distance = wall_distance(mesh, pde["porder"])
    # ODG layout: wall distance, filtered AV sensor, mesh-adaptation pressure.
    mesh["vdg"] = np.zeros((distance.shape[0], 3, distance.shape[2]), order="F")
    mesh["vdg"][:, 0:1, :] = distance
    npe, _, ne = mesh["dgnodes"].shape
    # Match MATLAB's HDG initialization exactly: conservative variables are
    # followed by the two zero gradient blocks in the 12-component UDG array.
    mesh["udg"] = np.zeros((npe, 12, ne), dtype=np.float64, order="F")
    mesh["udg"][:, 0, :] = density_inf
    mesh["udg"][:, 1, :] = momentum_x_inf * np.tanh(
        av_distance_coefficient * distance[:, 0, :]
    )
    mesh["udg"][:, 2, :] = momentum_y_inf
    near_wall_temperature = temperature_inf * (
        wall_temperature / reference_temperature - 1.0
    ) * np.exp(-av_distance_coefficient * distance[:, 0, :]) + temperature_inf
    mesh["udg"][:, 3, :] = near_wall_temperature + 0.5 * (
        mesh["udg"][:, 1, :] ** 2 + mesh["udg"][:, 2, :] ** 2
    )
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
    params=np.asarray(fluid["physicsparam"]).reshape(-1); params[-2:]=[0.0,1.5*fluid["AVcoeffEnd"]]; fluid["physicsparam"]=params
    fluid["NLiter"]=20

    solid,_=exasim.initializeexasim(); solid.update({"model":"ModelD","modelfile":"pdemodel_solid","platform":fluid["platform"],
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


def main():
    return run_isoq_case(build_case, True, HERE)


if __name__ == "__main__":
    main()
