include(joinpath(@__DIR__,"cylinder_fluid_model.jl"))
include(joinpath(@__DIR__,"solid_heat_model.jl"))
include(joinpath(@__DIR__,"cylinder_mesh.jl"))
using DelimitedFiles
using .CylinderMesh

# Artificial-viscosity continuation (c, k) for av = c*tanh(k*d), as in pdeapp1.m.
const AV_CONTINUATION = ((0.06,30.0),(0.04,30.0),(0.025,30.0),(0.025,5.0))

function setpaths!(pde,path)
    mkpath(path); pde.datapath=path; pde.builddir=joinpath(path,".exasim"); pde.buildpath=pde.builddir
end

function cylinder_fluid(case,path)
    pde,_=Exasim.initializeexasim(); setpaths!(pde,path)
    pde.model="ModelD"; pde.modelfile="CylinderFluidModel"; pde.platform="cpu"; pde.mpiprocs=2
    pde.hybrid=1; pde.porder=4; pde.pgauss=8; pde.Cxxpreprocessing=0
    pde.GMRESrestart=100; pde.linearsolvertol=1e-8; pde.linearsolveriter=100
    pde.RBdim=0; pde.ppdegree=20; pde.NLtol=1e-6; pde.NLiter=30; pde.matvectol=1e-6
    gam,re,pr,mach,tref,twall=1.4,1.835e5,0.71,8.03,124.49,294.44
    pinf=1.0/(gam*mach^2); tinf=pinf/(gam-1.0)
    base=[gam,re,pr,mach,1.0,1.0,0.0,0.5+pinf/(gam-1.0),tinf,tref,twall]
    # AV continuation av = c*tanh(k*d): one sweep case per [c, k], each warm-started
    pde.physicsparamsweep=reduce(vcat,[reshape([base; c; k],1,:) for (c,k) in AV_CONTINUATION])
    pde.physicsparamwarmstart=1
    pde.physicsparam=pde.physicsparamsweep[1:1,:]
    pde.tau=[1.0]; ny=case==:matching ? 21 : 28
    mesh=CylinderMesh.make_cylinder_mesh(4,31,ny,1;solid=false); mesh.boundarycondition=reshape([3,2,1],:,1)
    distance=CylinderMesh.wall_distance(mesh,4,1); mesh.odg=zeros(size(distance,1),2,size(distance,3))
    mesh.odg[:,1,:].=distance[:,1,:]
    # state and gradient components (gradients start at zero), as in pdeapp1.m; the
    # sweep warm start rewrites all 12 components
    mesh.udg=zeros(size(mesh.dgnodes,1),12,size(mesh.dgnodes,3)); mesh.udg[:,1,:].=1.0
    mesh.udg[:,2,:].=tanh.(10.0.*distance[:,1,:])
    tnear=tinf.*(twall/tref-1.0).*exp.(-10.0.*distance[:,1,:]).+tinf
    mesh.udg[:,4,:].=tnear.+0.5.*mesh.udg[:,2,:].^2
    return pde,mesh,distance
end

function continue_fluid(pde,mesh,distance)
    solcases,pde,mesh=Exasim.exasim(pde,mesh)[1:3]
    # The coupled solve starts from the last continuation case and keeps its AV.
    sol=solcases[end]
    ndims(sol) == 4 && (sol = sol[:,:,:,end])
    pde.physicsparam=pde.physicsparamsweep[end:end,:]
    pde.physicsparamsweep=[]; pde.physicsparamwarmstart=0
    mesh.udg=sol; return pde,mesh,sol
end

function cylinder_solid(case,fluid,path)
    pde,_=Exasim.initializeexasim(); setpaths!(pde,path)
    pde.model="ModelD"; pde.modelfile="SolidHeatModel"; pde.platform="cpu"; pde.mpiprocs=2
    pde.hybrid=1; pde.porder=4; pde.pgauss=8; pde.Cxxpreprocessing=0
    pde.GMRESrestart=100; pde.linearsolvertol=1e-8; pde.linearsolveriter=100
    pde.RBdim=0; pde.ppdegree=20; pde.NLtol=1e-6; pde.NLiter=30; pde.matvectol=1e-6
    solid_temperature=294.44/124.49*fluid.physicsparam[9]
    pde.physicsparam=reshape([0.2,solid_temperature],1,:); pde.tau=[1.0]
    elemtype,ny=case==:quad_tri ? (0,11) : (1,21)
    mesh=CylinderMesh.make_cylinder_mesh(4,5,ny,elemtype;solid=true)
    mesh.udg=zeros(size(mesh.dgnodes,1),3,size(mesh.dgnodes,3)); mesh.udg[:,1,:].=solid_temperature
    return pde,mesh
end

function run_cylinder_case(case,case_directory;run_directory=joinpath(case_directory,"julia_run"))
    fluid,fluid_mesh,distance=cylinder_fluid(case,joinpath(run_directory,"precursor"))
    fluid,fluid_mesh,_=continue_fluid(fluid,fluid_mesh,distance)
    solid,solid_mesh=cylinder_solid(case,fluid,joinpath(run_directory,"coupled")); setpaths!(fluid,joinpath(run_directory,"coupled"))
    for (m,pde) in enumerate((fluid,solid)); pde.modelnumber=m-1; pde.interfacetype=case==:matching ? 0 : 1; end
    fluid_mesh.boundarycondition=reshape([5,2,1],:,1); fluid_mesh.interfacecondition=reshape([1,0,0],:,1); fluid.interfacefluxmap=[4]
    solid_mesh.boundarycondition=reshape([3,2,1],:,1); solid_mesh.interfacecondition=reshape([1,0,0],:,1); solid.interfacefluxmap=[1]
    return Exasim.exasim([fluid,solid],[fluid_mesh,solid_mesh])
end
