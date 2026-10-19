"""Curved half-cylinder mesh helpers for the reacting-flow example."""

logdec_values(x,a)=minimum(x).+(maximum(x)-minimum(x)).*(1.0.-exp.(-a.*(x.-minimum(x))./(maximum(x)-minimum(x))))./(1.0-exp(-a))
function quad_mesh(nx,ny)
    x=repeat(collect(range(0.,1.,length=nx+1)),ny+1); y=repeat(collect(range(0.,1.,length=ny+1)),inner=nx+1)
    p=vcat(reshape(x,1,:),reshape(y,1,:)); cells=zeros(Int,4,nx*ny); e=1
    for j in 0:ny-1, i in 0:nx-1
        n=i+j*(nx+1)+1; cells[:,e].=[n,n+1,n+nx+2,n+nx+1]; e+=1
    end
    p,cells
end
function map_points(p)
    theta=1.5pi.+p[2,:].*(0.5pi-1.5pi); outer=-1.25.*cos.(theta).+2.25.*(1.0.+cos.(theta)); radius=outer.+p[1,:].*(1.0.-outer)
    vcat(reshape(radius.*cos.(theta),1,:),reshape(radius.*sin.(theta),1,:))
end
function map_nodes(nodes)
    theta=1.5pi.+nodes[:,2,:].*(0.5pi-1.5pi); outer=-1.25.*cos.(theta).+2.25.*(1.0.+cos.(theta)); radius=outer.+nodes[:,1,:].*(1.0.-outer)
    mapped=similar(nodes); mapped[:,1,:].=radius.*cos.(theta); mapped[:,2,:].=radius.*sin.(theta); mapped
end
function make_mesh(porder)
    p,t=quad_mesh(80,80); nodes=Exasim.Preprocessing.createdgnodes(p,t,zeros(Int,4,size(t,2)),[],[],porder)
    p[1,:].=logdec_values(p[1,:],0.5); nodes[:,1,:].=logdec_values(nodes[:,1,:],0.5); p=map_points(p); nodes=map_nodes(nodes)
    expressions = [
        x -> sqrt.(x[1, :] .^ 2 .+ x[2, :] .^ 2) .< 1.0 + 1.0e-6,
        x -> x[1, :] .> -1.0e-7,
        x -> abs.(x[1, :]) .< 20.0,
    ]
    f,_,_=Exasim.Preprocessing.facenumbering(p,t,1,expressions,[]); _,mesh=Exasim.initializeexasim()
    mesh.p,mesh.t,mesh.f,mesh.dgnodes=p,t,f,nodes; mesh.boundaryexpr=expressions; mesh.boundarycondition=reshape([3,6,5],:,1); mesh
end
function wall_distance(mesh,porder)
    _,_,_,_,perm=Exasim.Preprocessing.Master.masternodes(porder,2,1); indices=findall(mesh.f.==1)
    wall=reduce(vcat,[mesh.dgnodes[perm[:,i[1]],:,i[2]] for i in indices]); npe,_,ne=size(mesh.dgnodes); distance=zeros(npe,1,ne)
    for e in 1:ne, a in 1:npe
        delta=wall.-reshape(mesh.dgnodes[a,:,e],1,2); distance[a,1,e]=minimum(sqrt.(sum(delta.*delta,dims=2)))
    end
    distance
end
function element_scales(mesh)
    npe,_,ne=size(mesh.dgnodes); scales=zeros(npe,ne)
    for e in 1:ne
        vertices=mesh.p[:,mesh.t[:,e]]; dx=vertices[:,2]-vertices[:,1]; dy=vertices[:,4]-vertices[:,1]
        scales[:,e].=sqrt(abs(dx[1]*dy[2]-dx[2]*dy[1]))
    end
    scales
end
