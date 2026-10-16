include(joinpath(@__DIR__,"isoq_solid_model.jl"))
include(joinpath(@__DIR__,"isoq_solid_mesh.jl"))
using .IsoqSolidMesh

function run_isoq_case(build_fluid,matching,case_directory;run_directory=joinpath(case_directory,"julia_run"))
    precursor=joinpath(run_directory,"precursor"); fluid,fluid_mesh=build_fluid(precursor)
    fluid_solution,fluid,fluid_mesh,fluid_master,fluid_dmd=Exasim.exasim(fluid,fluid_mesh)[1:5]
    ndims(fluid_solution) == 4 && (fluid_solution = fluid_solution[:,:,:,end])
    fluid_mesh.udg=fluid_solution; vdgbase=joinpath(precursor,"dataout","outvdg")
    if isfile(vdgbase*"_np0.bin"); fluid_mesh.odg=Exasim.Postprocessing.getsolution(vdgbase,fluid_dmd,fluid_master.npe); end
    coupled=joinpath(run_directory,"coupled"); mkpath(coupled); setpaths(p,path)=(p.datapath=path;p.builddir=joinpath(path,".exasim");p.buildpath=p.builddir)
    # The fluid uses quadrilaterals and the solid uses triangles, so both the
    # coincident and independently meshed interfaces require the type-1 path.
    setpaths(fluid,coupled); fluid.modelnumber=0; fluid.interfacetype=1
    fluid_mesh.boundarycondition=reshape([4,2,1,7],:,1); fluid_mesh.interfacecondition=reshape([0,0,0,1],:,1)
    fluid.interfacefluxmap=[4]; fluid.ibs=7; fluid.distanceboundaryconditions=[7]
    fluid.AVdistfunction=0; fluid.AVsmoothingMethod=0; fluid.AV=0
    params=vec(fluid.physicsparam); params[end-1:end]=[0.0,1.5*fluid.AVcoeffEnd]; fluid.physicsparam=reshape(params,1,:); fluid.NLiter=20
    solid,_=Exasim.initializeexasim(); setpaths(solid,coupled)
    solid.model="ModelD"; solid.modelfile="IsoqSolidModel"; solid.platform=fluid.platform; solid.mpiprocs=2; solid.hybrid=1
    solid.debugmode=fluid.debugmode; solid.porder=fluid.porder; solid.pgauss=fluid.pgauss; solid.Cxxpreprocessing=0
    solid.GMRESrestart=fluid.GMRESrestart; solid.GMRESortho=fluid.GMRESortho; solid.linearsolvertol=fluid.linearsolvertol
    solid.linearsolveriter=fluid.linearsolveriter; solid.preconditioner=fluid.preconditioner; solid.RBdim=fluid.RBdim; solid.ppdegree=fluid.ppdegree
    solid.NLtol=fluid.NLtol; solid.NLiter=20; solid.matvectol=fluid.matvectol; solid.modelnumber=1; solid.interfacetype=1; solid.interfacefluxmap=[1]
    gam,mach,tref,twall=params[1],params[4],params[10],params[11]; tinf=1.0/(gam*(gam-1.0)*mach^2); solid_temperature=twall/tref*tinf
    solid.physicsparam=reshape([0.01,solid_temperature],1,:); solid.tau=[1.0]
    solid_mesh=IsoqSolidMesh.make_isoq_solid(solid,fluid_mesh,matching;mesh_size=0.0028)
    solid_mesh.udg=zeros(size(solid_mesh.dgnodes,1),3,size(solid_mesh.dgnodes,3)); solid_mesh.udg[:,1,:].=solid_temperature
    return Exasim.exasim([fluid,solid],[fluid_mesh,solid_mesh])
end
