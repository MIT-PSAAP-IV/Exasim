# Julia coupled Mach-8 cylinder heat-transfer example (non-matching meshes).
# Model 1 is the fluid (pdemodel1.jl), model 2 the solid (pdemodel2.jl). A fluid
# precursor with artificial-viscosity continuation is solved first; the coupled
# fluid/solid solve then starts from it.
# Run with `julia --project=<repo>/frontends/Julia/Exasim pdeapp.jl`; data and the
# solver build go to ./julia_run.

repository = dirname(dirname(dirname(dirname(@__DIR__))))
# Use EXASIM_PREFIX if set, otherwise a sibling <repo>/../exasim_install.
source_install = joinpath(dirname(repository), "exasim_install")
if !haskey(ENV, "EXASIM_PREFIX") &&
   isfile(joinpath(source_install, "lib", "cmake", "Exasim", "ExasimConfig.cmake"))
    ENV["EXASIM_PREFIX"] = source_install
end
pushfirst!(LOAD_PATH, joinpath(repository, "frontends", "Julia", "Exasim"))
using Exasim
include(joinpath(@__DIR__,"pdemodel1.jl"))
include(joinpath(@__DIR__,"pdemodel2.jl"))
using DelimitedFiles

module CylinderMesh
using Exasim

function stretch(values, alpha, increasing)
    lo, hi = extrema(values); s = (values .- lo) ./ (hi - lo)
    mapped = increasing ? (exp.(alpha .* s) .- 1.0) ./ (exp(alpha) - 1.0) :
                          (1.0 .- exp.(-alpha .* s)) ./ (1.0 - exp(-alpha))
    return lo .+ (hi - lo) .* mapped
end

function logical_mesh(nx, ny, elemtype)
    x = repeat(collect(range(0.0,1.0,length=nx+1)), ny+1)
    y = repeat(collect(range(0.0,1.0,length=ny+1)), inner=nx+1)
    p = vcat(reshape(x,1,:), reshape(y,1,:)); cells = Vector{Vector{Int}}()
    for iy=0:ny-1, ix=0:nx-1
        ll = ix + iy*(nx+1) + 1
        elemtype == 1 ? push!(cells,[ll,ll+1,ll+nx+2,ll+nx+1]) :
                        # Match mkmesh_square(..., parity=1): NW-SE diagonal.
                        append!(cells,[[ll,ll+1,ll+nx+1],[ll+1,ll+nx+2,ll+nx+1]])
    end
    return p, reduce(hcat,cells)
end

function half_circle_points(points,a,b,c)
    theta = 1.5pi .- pi .* points[2,:]
    outer = -b .* cos.(theta) .+ c .* (1.0 .+ cos.(theta))
    radius = outer .+ points[1,:] .* (a .- outer)
    return vcat(reshape(radius.*cos.(theta),1,:),reshape(radius.*sin.(theta),1,:))
end

function half_circle_nodes(nodes,a,b,c)
    theta = 1.5pi .- pi .* nodes[:,2,:]
    outer = -b .* cos.(theta) .+ c .* (1.0 .+ cos.(theta))
    radius = outer .+ nodes[:,1,:] .* (a .- outer)
    out = similar(nodes); out[:,1,:] .= radius.*cos.(theta); out[:,2,:] .= radius.*sin.(theta)
    return out
end

function make_cylinder_mesh(porder,nx,ny,elemtype; solid=false)
    p,t = logical_mesh(nx,ny,elemtype); nfe = elemtype == 1 ? 4 : 3
    dgnodes = Exasim.Preprocessing.createdgnodes(p,t,zeros(Int,nfe,size(t,2)),[],[],porder)
    p[1,:] .= stretch(p[1,:],solid ? 2.0 : 5.0,solid)
    dgnodes[:,1,:] .= stretch(dgnodes[:,1,:],solid ? 2.0 : 5.0,solid)
    args = solid ? (0.85,1.0,1.0) : (1.0,3.0,4.7)
    p = half_circle_points(p,args...); dgnodes = half_circle_nodes(dgnodes,args...)
    boundaryexpr = solid ? [z->sqrt.(z[1,:].^2+z[2,:].^2).>1.0-1e-6,z->z[1,:].>-1e-7,z->abs.(z[1,:]).<20.0] :
                           [z->sqrt.(z[1,:].^2+z[2,:].^2).<1.0+1e-6,z->z[1,:].>-1e-7,z->abs.(z[1,:]).<20.0]
    f,_,_ = Exasim.Preprocessing.facenumbering(p,t,elemtype,boundaryexpr,[])
    _,mesh = Exasim.initializeexasim(); mesh.p=p; mesh.t=t; mesh.f=f; mesh.dgnodes=dgnodes
    mesh.boundaryexpr=boundaryexpr; mesh.boundarycondition=reshape([3,2,1],:,1)
    mesh.periodicexpr=[]; mesh.curvedboundary=zeros(Int,0,0); mesh.curvedboundaryexpr=[]; mesh.periodicboundary=zeros(Int,0,0)
    return mesh
end

function wall_distance(mesh,porder,elemtype)
    _,_,_,_,perm = Exasim.Preprocessing.Master.masternodes(porder,2,elemtype)
    faces = findall(mesh.f .== 1)
    wall = reduce(vcat,[mesh.dgnodes[perm[:,i[1]],:,i[2]] for i in faces])
    distance=zeros(size(mesh.dgnodes,1),1,size(mesh.dgnodes,3))
    for e=1:size(mesh.dgnodes,3), i=1:size(mesh.dgnodes,1)
        delta=wall.-reshape(mesh.dgnodes[i,:,e],1,2); distance[i,1,e]=minimum(sqrt.(sum(delta.^2,dims=2)))
    end
    return distance
end

end

using .CylinderMesh

# Artificial-viscosity continuation (c, k) for av = c*tanh(k*d), as in pdeapp1.m.
const AV_CONTINUATION = ((0.06,30.0),(0.04,30.0),(0.025,30.0),(0.025,5.0))

function setpaths!(pde,path)
    mkpath(path); pde.datapath=path; pde.builddir=joinpath(path,".exasim"); pde.buildpath=pde.builddir
end

function cylinder_fluid(case,path)
    pde,_=Exasim.initializeexasim(); setpaths!(pde,path)
    pde.model="ModelD"; pde.modelfile="pdemodel1"; pde.platform="cpu"; pde.mpiprocs=2
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
    pde.model="ModelD"; pde.modelfile="pdemodel2"; pde.platform="cpu"; pde.mpiprocs=2
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

main()=run_cylinder_case(:nonmatching,@__DIR__)
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
