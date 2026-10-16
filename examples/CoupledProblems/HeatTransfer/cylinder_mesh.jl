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
