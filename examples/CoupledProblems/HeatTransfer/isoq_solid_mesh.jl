module IsoqSolidMesh
using Exasim, LinearAlgebra, Printf

function ordered_wall(fluid,boundary=4)
    face=Exasim.Preprocessing.getelemface(2,1); indices=findall(fluid.f.==boundary)
    edges=[fluid.t[face[:,i[1]],i[2]] for i in indices]; adjacency=Dict{Int,Vector{Int}}()
    for (i,edge) in enumerate(edges), node in edge; push!(get!(adjacency,node,Int[]),i); end
    ends=[node for (node,links) in adjacency if length(links)==1]
    current=sort(ends,by=n->(fluid.p[2,n],fluid.p[1,n]))[1]; chain=[current]; used=falses(length(edges)); elements=Int[]; faces=Int[]
    for _=1:length(edges)
        candidates=[i for i in adjacency[current] if !used[i]]; length(candidates)==1 || error("fluid wall is not one open edge chain")
        i=candidates[1]; used[i]=true; push!(elements,indices[i][2]); push!(faces,indices[i][1])
        current=edges[i][1]==current ? edges[i][2] : edges[i][1]; push!(chain,current)
    end
    return fluid.p[:,chain]',elements,faces
end

function write_geo(path,wall,ymin,xmax,h,matching)
    open(path,"w") do io
        for i=1:size(wall,1); @printf(io,"Point(%d) = {%.17g, %.17g, 0, %.17g};\n",i,wall[i,1],wall[i,2],h); end
        corner=size(wall,1)+1; @printf(io,"Point(%d) = {%.17g, %.17g, 0, %.17g};\n",corner,xmax,ymin,h)
        if matching
            for i=1:size(wall,1)-1; println(io,"Line($i) = {$i, $(i+1)};"); end
            println(io,"Line($(size(wall,1))) = {$(size(wall,1)), $corner};\nLine($(size(wall,1)+1)) = {$corner, 1};")
            println(io,"Curve Loop(1) = {",join(1:size(wall,1)+1,","),"};\nPlane Surface(1) = {1};")
            println(io,"Transfinite Curve {",join(1:size(wall,1)-1,","),"} = 2;")
            @printf(io,"Mesh.MeshSizeMin = %.17g;\nMesh.MeshSizeMax = %.17g;\n",0.25h,h)
        else
            println(io,"Spline(1) = {",join(1:size(wall,1),","),"};")
            println(io,"Line(2) = {$(size(wall,1)), $corner};\nLine(3) = {$corner, 1};\nCurve Loop(1) = {1,2,3};\nPlane Surface(1) = {1};")
            @printf(io,"Mesh.MeshSizeMin = %.17g;\nMesh.MeshSizeMax = %.17g;\n",0.95h,1.05h)
            println(io,"Mesh.MeshSizeFromPoints = 1;\nMesh.MeshSizeFromCurvature = 0;\nMesh.MeshSizeExtendFromBoundary = 0;\nMesh.Smoothing = 20;\nMesh.Optimize = 1;\nMesh.OptimizeNetgen = 1;")
        end
        println(io,"Mesh.Algorithm = 6;\nMesh.ElementOrder = 1;\nMesh.RecombineAll = 0;")
    end
end

function golden_minimum(fun)
    a,b=0.0,1.0; ratio=(sqrt(5.0)-1.0)/2.0; c=b-ratio*(b-a); d=a+ratio*(b-a); fc,fd=fun(c),fun(d)
    for _=1:70
        if fc<fd; b,d,fd=d,c,fc; c=b-ratio*(b-a); fc=fun(c)
        else; a,c,fc=c,d,fd; d=a+ratio*(b-a); fd=fun(d); end
    end
    x=0.5*(a+b); return x,fun(x)
end

function project_points(points,fluid,wall,elements,faces,porder)
    plocfc=Exasim.Preprocessing.Master.masternodes(porder,1,1)[1]
    perm=Exasim.Preprocessing.Master.masternodes(porder,2,1)[5]
    face_nodes=Matrix{Float64}[]
    for i=1:length(elements)
        nodes=fluid.dgnodes[perm[:,faces[i]],:,elements[i]]
        if norm(nodes[1,:]-wall[i,:])+norm(nodes[end,:]-wall[i+1,:]) > norm(nodes[1,:]-wall[i+1,:])+norm(nodes[end,:]-wall[i,:]); nodes=nodes[end:-1:1,:]; end
        push!(face_nodes,nodes)
    end
    result=similar(points)
    for j=1:size(points,1)
        best_distance=Inf; best=copy(points[j,:])
        for nodes in face_nodes
            evaluate=xi->(Exasim.Preprocessing.mkshape(porder,plocfc,reshape([xi],1,1),1)[:,:,1]'*nodes)[1,:]
            objective=xi->sum((evaluate(xi).-points[j,:]).^2); xi,_=golden_minimum(objective)
            for candidate in (0.0,xi,1.0); mapped=evaluate(candidate); distance=sum((mapped.-points[j,:]).^2); if distance<best_distance; best_distance=distance; best=mapped; end; end
        end
        result[j,:]=best
    end
    return result
end

function make_isoq_solid(pde,fluid,matching;mesh_size=0.0028)
    wall,elements,faces=ordered_wall(fluid); ymin=minimum(fluid.p[2,:]); xmax=maximum(fluid.p[1,:]); scale=max(1.0,maximum(maximum(fluid.p,dims=2)-minimum(fluid.p,dims=2))); tol=1e-10scale
    p,t=mktempdir() do directory
        base=joinpath(directory,"solid"); write_geo(base*".geo",wall,ymin,xmax,mesh_size,matching); Exasim.Mesh.gmshcall(pde,base,2,0)
    end
    area=(p[1,t[2,:]]-p[1,t[1,:]]).*(p[2,t[3,:]]-p[2,t[1,:]]).-(p[2,t[2,:]]-p[2,t[1,:]]).*(p[1,t[3,:]]-p[1,t[1,:]])
    for e in findall(area.<0); t[2,e],t[3,e]=t[3,e],t[2,e]; end
    startx,endy=wall[1,1],wall[end,2]; wall_vertices=Int[]
    if matching
        for point in eachrow(wall); distance=[norm(p[:,i]-point) for i=1:size(p,2)]; node=argmin(distance); distance[node]<=tol || error("Gmsh did not preserve a fluid wall vertex"); p[:,node]=point; push!(wall_vertices,node); end
        onwall=z->[minimum([norm(z[:,j]-wall[i,:]) for i=1:size(wall,1)])<=tol for j=1:size(z,2)]
    else
        onwall=z->.!(((abs.(z[2,:].-ymin).<tol).&(z[1,:].>startx+tol)).|((abs.(z[1,:].-xmax).<tol).&(z[2,:].<endy-tol)))
    end
    boundaryexpr=[onwall,z->abs.(z[2,:].-ymin).<tol,z->abs.(z[1,:].-xmax).<tol]
    f,_,_=Exasim.Preprocessing.facenumbering(p,t,0,boundaryexpr,[]); edge=Exasim.Preprocessing.getelemface(2,0)
    if !matching
        vertices=unique(vcat([t[edge[:,i[1]],i[2]] for i in findall(f.==1)]...)); p[:,vertices]=project_points(p[:,vertices]',fluid,wall,elements,faces,pde.porder)'
        f,_,_=Exasim.Preprocessing.facenumbering(p,t,0,boundaryexpr,[])
    end
    dgnodes=Exasim.Preprocessing.createdgnodes(p,t,f,[],[],pde.porder); perm=Exasim.Preprocessing.Master.masternodes(pde.porder,2,0)[5]; fluidperm=Exasim.Preprocessing.Master.masternodes(pde.porder,2,1)[5]
    for index in findall(f.==1); lf,e=index[1],index[2]
        if matching
            vertices=t[edge[:,lf],e]; segment=findfirst(i->Set(vertices)==Set(wall_vertices[i:i+1]),1:length(wall_vertices)-1)
            target=fluid.dgnodes[fluidperm[:,faces[segment]],:,elements[segment]]; vertices[1]==wall_vertices[segment+1] && (target=target[end:-1:1,:]); dgnodes[perm[:,lf],:,e]=target
        else
            dgnodes[perm[:,lf],:,e]=project_points(dgnodes[perm[:,lf],:,e],fluid,wall,elements,faces,pde.porder)
        end
    end
    _,mesh=Exasim.initializeexasim(); mesh.p=p; mesh.t=t; mesh.f=f; mesh.dgnodes=dgnodes; mesh.boundaryexpr=boundaryexpr
    mesh.boundarycondition=reshape([3,2,1],:,1); mesh.interfacecondition=reshape([1,0,0],:,1); mesh.periodicexpr=[]; mesh.curvedboundary=zeros(Int,0,0); mesh.curvedboundaryexpr=[]; mesh.periodicboundary=zeros(Int,0,0)
    return mesh
end

end
