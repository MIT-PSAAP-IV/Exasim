"""Two-block ISOQ mesh shared by the Julia verification driver."""

using LinearAlgebra

function read_canonical_array(name)
    path = joinpath(@__DIR__, "canonical_mesh", name * ".bin")
    open(path, "r") do io
        dims = filter(>(0), Int.(read!(io, Vector{Float64}(undef, 3))))
        length(dims) >= 2 || error("invalid canonical mesh header in $path")
        values = read!(io, Vector{Float64}(undef, prod(dims)))
        eof(io) || error("unexpected trailing data in $path")
        reshape(values, Tuple(dims))
    end
end

function load_canonical_mesh(domain)
    points = read_canonical_array(domain * "_p")
    elements = Int.(read_canonical_array(domain * "_t")) .+ 1
    faces = Int.(read_canonical_array(domain * "_f"))
    dgnodes = read_canonical_array(domain * "_dgnodes")
    ymin, xmax = minimum(points[2, :]), maximum(points[1, :])
    _, mesh = Exasim.initializeexasim()
    mesh.p, mesh.t, mesh.f, mesh.dgnodes = points, elements, faces, dgnodes
    if domain == "fluid"
        mesh.udg = read_canonical_array("fluid_udg")
        mesh.odg = read_canonical_array("fluid_vdg")
        mesh.boundaryexpr = [
            p -> abs.(p[2, :] .- ymin) .< 1.0e-6,
            p -> p[1, :] .> xmax - 1.0e-4,
            p -> (p[1, :] .< -1.0e-3) .| (p[2, :] .> 0.1),
            p -> abs.(p[1, :]) .< 20.0 + 1.0e-6,
        ]
    else
        scale = max(1.0, maximum(maximum(points, dims=2) - minimum(points, dims=2)))
        tolerance = 1.0e-10 * scale
        edge = Exasim.Preprocessing.getelemface(2, 0)
        indices = findall(faces .== 1)
        vertices = unique(vcat([elements[edge[:, index[1]], index[2]] for index in indices]...))
        startx, endy = minimum(points[1, vertices]), maximum(points[2, vertices])
        mesh.udg = read_canonical_array("solid_udg")
        mesh.boundaryexpr = [
            p -> .!(((abs.(p[2, :] .- ymin) .< tolerance) .& (p[1, :] .> startx + tolerance)) .|
                    ((abs.(p[1, :] .- xmax) .< tolerance) .& (p[2, :] .< endy - tolerance))),
            p -> abs.(p[2, :] .- ymin) .< tolerance,
            p -> abs.(p[1, :] .- xmax) .< tolerance,
        ]
        mesh.interfacecondition = reshape([1, 0, 0], :, 1)
    end
    mesh.periodicexpr = []
    mesh.curvedboundary = zeros(Int, 0, 0)
    mesh.curvedboundaryexpr = []
    mesh.periodicboundary = zeros(Int, 0, 0)
    mesh
end

function loginc_values(values, alpha)
    lower, upper = extrema(values)
    upper == lower && return copy(values)
    lower .+ (upper - lower) .* (
        exp.(alpha .* (values .- lower) ./ (upper - lower)) .- 1.0
    ) ./ (exp(alpha) - 1.0)
end

function logdec_values(values, alpha)
    lower, upper = extrema(values)
    upper == lower && return copy(values)
    lower .+ (upper - lower) .* (
        1.0 .- exp.(-alpha .* (values .- lower) ./ (upper - lower))
    ) ./ (1.0 - exp(-alpha))
end

function rectangular_quad_mesh(nx, ny, porder)
    x = repeat(collect(range(0.0, 1.0, length=nx + 1)), ny + 1)
    y = repeat(collect(range(0.0, 1.0, length=ny + 1)), inner=nx + 1)
    points = vcat(reshape(x, 1, :), reshape(y, 1, :))
    elements = zeros(Int, 4, nx * ny)
    element = 1
    for iy in 0:(ny-1), ix in 0:(nx-1)
        lower_left = ix + iy * (nx + 1) + 1
        elements[:, element] .= [
            lower_left, lower_left + 1, lower_left + nx + 2, lower_left + nx + 1
        ]
        element += 1
    end
    dgnodes = Exasim.Preprocessing.createdgnodes(
        points, elements, zeros(Int, 4, size(elements, 2)), [], [], porder
    )
    return points, elements, dgnodes
end

function inverse_bilinear(point, coefficients, initial)
    xi, eta = initial
    for _ in 1:100
        residual = [1.0, xi, eta, xi * eta]' * coefficients - point'
        jacobian = [
            coefficients[2, 1] + coefficients[4, 1] * eta coefficients[3, 1] + coefficients[4, 1] * xi
            coefficients[2, 2] + coefficients[4, 2] * eta coefficients[3, 2] + coefficients[4, 2] * xi
        ]
        update = jacobian \ (-vec(residual))
        xi += update[1]
        eta += update[2]
        norm(update) < 1.0e-10 && return [xi, eta]
    end
    error("inverse bilinear map did not converge")
end

function polynomial_fit(x, y, degree)
    vandermonde = hcat([x .^ power for power in degree:-1:0]...)
    vandermonde \ y
end

function polynomial_values(coefficients, x)
    value = zero(x)
    for coefficient in coefficients
        value .= value .* x .+ coefficient
    end
    value
end

function forward_bilinear(reference_points, coefficients)
    xi, eta = reference_points[:, 1], reference_points[:, 2]
    hcat(ones(length(xi)), xi, eta, xi .* eta) * coefficients
end

function surfmesh2d(lower_curve, upper_curve, nx, ny, porder;
                    streamwise_scaling=(2.0, 1.5), normal_scaling=(3.0, 1.0e-8))
    matrix = [
        1.0 0.0 0.0 0.0
        1.0 1.0 0.0 0.0
        1.0 1.0 1.0 1.0
        1.0 0.0 1.0 0.0
    ]
    corners = vcat(
        lower_curve[1:1, :], lower_curve[end:end, :],
        upper_curve[end:end, :], upper_curve[1:1, :]
    )
    coefficients = matrix \ corners
    lower_reference = reduce(
        vcat,
        [reshape(inverse_bilinear(lower_curve[i, :], coefficients,
                                  ((i - 1) / (size(lower_curve, 1) - 1), 0.0)), 1, 2)
         for i in axes(lower_curve, 1)],
    )
    upper_reference = reduce(
        vcat,
        [reshape(inverse_bilinear(upper_curve[i, :], coefficients,
                                  ((i - 1) / (size(upper_curve, 1) - 1), 1.0)), 1, 2)
         for i in axes(upper_curve, 1)],
    )
    lower_fit = polynomial_fit(
        lower_reference[:, 1], lower_reference[:, 2], min(size(lower_curve, 1) - 1, 12)
    )
    upper_fit = polynomial_fit(
        upper_reference[:, 1], upper_reference[:, 2], min(size(upper_curve, 1) - 1, 12)
    )

    points, elements, dgnodes = rectangular_quad_mesh(nx, ny, porder)
    vertex_reference = copy(points')
    node_reference = hcat(vec(dgnodes[:, 1, :]), vec(dgnodes[:, 2, :]))
    sides = (
        vertex_left=abs.(vertex_reference[:, 1]) .< 1.0e-6,
        vertex_right=abs.(vertex_reference[:, 1] .- 1.0) .< 1.0e-6,
        node_left=abs.(node_reference[:, 1]) .< 1.0e-6,
        node_right=abs.(node_reference[:, 1] .- 1.0) .< 1.0e-6,
    )
    for reference in (vertex_reference, node_reference)
        reference[:, 1] .= logdec_values(
            loginc_values(reference[:, 1], streamwise_scaling[1]),
            max(streamwise_scaling[2], 1.0e-8),
        )
        reference[:, 2] .= logdec_values(
            loginc_values(reference[:, 2], normal_scaling[1]),
            max(normal_scaling[2], 1.0e-8),
        )
        lower = polynomial_values(lower_fit, reference[:, 1])
        upper = polynomial_values(upper_fit, reference[:, 1])
        endpoints = (abs.(reference[:, 1]) .< 1.0e-6) .|
                    (abs.(reference[:, 1] .- 1.0) .< 1.0e-6)
        lower[endpoints] .= 0.0
        upper[endpoints] .= 1.0
        reference[:, 2] .= lower .+ (upper .- lower) .* reference[:, 2]
    end
    physical_points = forward_bilinear(vertex_reference, coefficients)
    physical_nodes = forward_bilinear(node_reference, coefficients)
    dgnodes[:, 1, :] .= reshape(physical_nodes[:, 1], size(dgnodes, 1), size(dgnodes, 3))
    dgnodes[:, 2, :] .= reshape(physical_nodes[:, 2], size(dgnodes, 1), size(dgnodes, 3))
    return physical_points', elements, dgnodes, sides
end

function isoq_curves()
    nose_radius = 0.102
    shoulder_radius = nose_radius / 16.0
    end_x = 0.06
    nose_angle = 27.82pi / 180.0
    theta = collect(range(0.0, nose_angle, length=481))
    nose = hcat(nose_radius .* (1.0 .- cos.(theta)), nose_radius .* sin.(theta))
    initial = pi - nose_angle
    center = nose[end, :] - shoulder_radius .* [cos(initial), sin(initial)]
    theta = collect(range(initial, pi / 2.0, length=49))
    shoulder = reshape(center, 1, 2) .+
               shoulder_radius .* hcat(cos.(theta), sin.(theta))
    afterbody = hcat(
        collect(range(shoulder[end, 1], end_x, length=240)),
        fill(shoulder[end, 2], 240),
    )
    lower = vcat(nose[1:end-1, :], shoulder[1:end-1, :], afterbody)
    theta = collect(range(pi, pi / 2.0, length=2000))
    upper = hcat(
        3.7end_x .+ 4.4end_x .* cos.(theta), 3.6end_x .* sin.(theta)
    )
    upper = upper[upper[:, 1] .<= end_x, :]
    upper[end, 1] = end_x
    return lower, upper
end

function connect_blocks(first, second; tolerance=1.0e-5)
    p1, t1, dgnodes1, side1 = first
    p2, t2, dgnodes2, side2 = second
    average_vertices = 0.5 .* (
        p1[:, side1.vertex_right]' .+ p2[:, side2.vertex_left]'
    )
    p1[:, side1.vertex_right] .= average_vertices'
    p2[:, side2.vertex_left] .= average_vertices'
    flat1 = hcat(vec(dgnodes1[:, 1, :]), vec(dgnodes1[:, 2, :]))
    flat2 = hcat(vec(dgnodes2[:, 1, :]), vec(dgnodes2[:, 2, :]))
    average_nodes = 0.5 .* (flat1[side1.node_right, :] .+ flat2[side2.node_left, :])
    flat1[side1.node_right, :] .= average_nodes
    flat2[side2.node_left, :] .= average_nodes
    dgnodes1[:, 1, :] .= reshape(flat1[:, 1], size(dgnodes1, 1), size(dgnodes1, 3))
    dgnodes1[:, 2, :] .= reshape(flat1[:, 2], size(dgnodes1, 1), size(dgnodes1, 3))
    dgnodes2[:, 1, :] .= reshape(flat2[:, 1], size(dgnodes2, 1), size(dgnodes2, 3))
    dgnodes2[:, 2, :] .= reshape(flat2[:, 2], size(dgnodes2, 1), size(dgnodes2, 3))

    mapping = zeros(Int, size(p2, 2))
    keep = trues(size(p2, 2))
    next_index = size(p1, 2) + 1
    for index in axes(p2, 2)
        squared_distance = vec(sum((p1 .- p2[:, index]).^2, dims=1))
        nearest = argmin(squared_distance)
        if squared_distance[nearest] < tolerance^2
            mapping[index] = nearest
            keep[index] = false
        else
            mapping[index] = next_index
            next_index += 1
        end
    end
    points = hcat(p1, p2[:, keep])
    elements = hcat(t1, mapping[t2])
    dgnodes = cat(dgnodes1, dgnodes2, dims=3)
    return points, elements, dgnodes
end

function make_isoq_mesh(porder; radial_shift=5.0e-4)
    lower, upper = isoq_curves()
    upper_first = upper[upper[:, 1] .<= -0.02, :]
    upper_second = vcat(upper_first[end:end, :], upper[upper[:, 1] .> -0.02, :])
    lower_first = lower[lower[:, 1] .<= 0.013, :]
    lower_second = vcat(lower_first[end:end, :], lower[lower[:, 1] .> 0.013, :])
    points, elements, dgnodes = connect_blocks(
        surfmesh2d(lower_first, upper_first, 30, 60, porder),
        surfmesh2d(lower_second, upper_second, 25, 60, porder),
    )
    points[2, :] .+= radial_shift
    dgnodes[:, 2, :] .+= radial_shift
    lower_y, right_x = minimum(points[2, :]), maximum(points[1, :])
    boundaryexpr = [
        p -> abs.(p[2, :] .- lower_y) .< 1.0e-6,
        p -> p[1, :] .> right_x - 1.0e-4,
        p -> (p[1, :] .< -1.0e-3) .| (p[2, :] .> 0.1),
        p -> abs.(p[1, :]) .< 20.0 + 1.0e-6,
    ]
    faces, _, _ = Exasim.Preprocessing.facenumbering(points, elements, 1, boundaryexpr, [])
    _, mesh = Exasim.initializeexasim()
    mesh.p = points
    mesh.t = elements
    mesh.f = faces
    mesh.dgnodes = dgnodes
    mesh.boundaryexpr = boundaryexpr
    mesh.boundarycondition = reshape([5, 2, 1, 3], :, 1)
    mesh.periodicexpr = []
    mesh.curvedboundary = zeros(Int, 0, 0)
    mesh.curvedboundaryexpr = []
    mesh.periodicboundary = zeros(Int, 0, 0)
    return mesh
end

function wall_distance(mesh, porder)
    _, _, _, _, perm = Exasim.Preprocessing.Master.masternodes(porder, 2, 1)
    boundary_faces = findall(mesh.f .== 4)
    wall_nodes = reduce(
        vcat,
        [mesh.dgnodes[perm[:, index[1]], :, index[2]] for index in boundary_faces],
    )
    npe, _, ne = size(mesh.dgnodes)
    distance = zeros(Float64, npe, 1, ne)
    for element in 1:ne, node in 1:npe
        delta = wall_nodes .- reshape(mesh.dgnodes[node, :, element], 1, 2)
        distance[node, 1, element] = minimum(sqrt.(sum(delta .* delta, dims=2)))
    end
    distance
end

# Gmsh triangular solid meshes for the matching and non-matching ISOQ cases.
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
