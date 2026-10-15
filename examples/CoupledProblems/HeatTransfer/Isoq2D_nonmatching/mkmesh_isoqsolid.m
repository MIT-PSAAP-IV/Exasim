function [mesh, info] = mkmesh_isoqsolid(pde, fluid, meshSize)
%MKMESH_ISOQSOLID Generate an independent triangular solid mesh with Gmsh.
% The fluid wall vertices define only the spline geometry. Gmsh chooses the
% solid interface vertices and the interior triangulation independently.

if nargin < 3 || isempty(meshSize)
    meshSize = 0.0028;
end
if size(fluid.t,1) ~= 4
    error('mkmesh_isoqsolid expects a quadrilateral fluid mesh.');
end
if meshSize <= 0
    error('The solid mesh size must be positive.');
end

[wall,fluidElement,fluidLocalFace] = ordered_fluid_wall(fluid,4);
ymin = min(fluid.p(2,:));
xmax = max(fluid.p(1,:));
scale = max(1.0,max(max(fluid.p,[],2)-min(fluid.p,[],2)));
tol = 1.0e-10*scale;

if abs(wall(1,2)-ymin) > tol || abs(wall(end,1)-xmax) > tol
    error('The fluid wall must run from the minimum-y point to the maximum-x boundary.');
end

basename = tempname;
cleanup = onCleanup(@() cleanup_gmsh_files(basename));
write_solid_geo([basename '.geo'],wall,ymin,xmax,meshSize);
[p,t] = gmshcall(pde,basename,2,0);

% Gmsh chooses the solid interface nodes independently. Project only their
% coordinates onto the fluid's high-order wall so the interface is
% aligned at the interpolation nodes without sharing nodes or face partitions.
interfaceVertex = interface_boundary_vertices( ...
    p,t,ymin,xmax,wall(1,1),wall(end,2),tol);
originalInterface = p(:,interfaceVertex).';
p(:,interfaceVertex) = project_to_fluid_wall( ...
    originalInterface,fluid,wall,fluidElement,fluidLocalFace).';
maxGridProjection = max(vecnorm(p(:,interfaceVertex).'-originalInterface,2,2));

area2 = signed_double_area(p,t);
if any(abs(area2) <= eps(scale^2))
    error('Gmsh generated a degenerate solid triangle.');
end
flip = area2 < 0;
t([2 3],flip) = t([3 2],flip);
area2 = abs(signed_double_area(p,t));

mesh = mkmesh(p.',t.',pde.porder,{'true'},0,1);
mesh.p = mesh.p.';
mesh.t = mesh.t.';
wallStartX = wall(1,1);
wallEndY = wall(end,2);
mesh.boundaryexpr = { ...
    @(q) on_interface(q,ymin,xmax,wallStartX,wallEndY,tol), ...
    @(q) abs(q(2,:)-ymin) < tol, ...
    @(q) abs(q(1,:)-xmax) < tol};
mesh.boundarycondition = [3;2;1];
mesh.interfacecondition = [1;0;0];
mesh.periodicboundary = [];
mesh.periodicexpr = {};
mesh.f = facenumbering(mesh.p,mesh.t,0,mesh.boundaryexpr,[]);

boundaryMask = boundary_face_mask(mesh.t);
if any(mesh.f(boundaryMask) == 0)
    error('At least one Gmsh solid boundary face was not classified.');
end
if abs(min(mesh.p(2,:))-ymin) > tol || abs(max(mesh.p(1,:))-xmax) > tol
    error('The solid mesh does not preserve the fluid ymin/xmax extents.');
end

[solidLocalFace,solidElement] = find(mesh.f == 1);
maxDGProjection = 0.0;
for k = 1:numel(solidElement)
    index = mesh.perm(:,solidLocalFace(k));
    points = mesh.dgnodes(index,:,solidElement(k));
    projected = project_to_fluid_wall( ...
        points,fluid,wall,fluidElement,fluidLocalFace);
    maxDGProjection = max(maxDGProjection,max(vecnorm(projected-points,2,2)));
    mesh.dgnodes(index,:,solidElement(k)) = projected;
end

edge12 = mesh.p(:,mesh.t(2,:))-mesh.p(:,mesh.t(1,:));
edge23 = mesh.p(:,mesh.t(3,:))-mesh.p(:,mesh.t(2,:));
edge31 = mesh.p(:,mesh.t(1,:))-mesh.p(:,mesh.t(3,:));
edge2sum = sum(edge12.^2,1)+sum(edge23.^2,1)+sum(edge31.^2,1);
quality = 2*sqrt(3)*area2./edge2sum;
triangleArea = 0.5*area2;

info.interfaceFaces = nnz(mesh.f == 1);
info.fluidInterfaceFaces = size(wall,1)-1;
info.triangles = size(mesh.t,2);
info.ymin = min(mesh.p(2,:));
info.xmax = max(mesh.p(1,:));
info.minQuality = min(quality);
info.meanQuality = mean(quality);
info.areaRatio = max(triangleArea)/min(triangleArea);
info.areaCoefficientOfVariation = std(triangleArea)/mean(triangleArea);
info.maxGridProjection = maxGridProjection;
info.maxDGProjection = maxDGProjection;

fprintf(['Independent Gmsh solid mesh: %d triangles, %d fluid / %d solid interface faces; ' ...
    'quality min/mean %.3f/%.3f, area ratio %.3f, area CV %.3f; ' ...
    'projection grid/DG %.3e/%.3e; ymin %.16g, xmax %.16g\n'], ...
    info.triangles,info.fluidInterfaceFaces,info.interfaceFaces, ...
    info.minQuality,info.meanQuality,info.areaRatio, ...
    info.areaCoefficientOfVariation,info.maxGridProjection, ...
    info.maxDGProjection,info.ymin,info.xmax);
end

function [wall,orderedElement,orderedFace] = ordered_fluid_wall(mesh,boundary)
face = getelemface(2,1);
[localFace,element] = find(mesh.f == boundary);
if isempty(element)
    error('The fluid mesh has no faces on wall boundary %d.',boundary);
end

edges = zeros(numel(element),2);
for i = 1:numel(element)
    edges(i,:) = mesh.t(face(:,localFace(i)),element(i));
end

nodes = unique(edges(:));
degree = zeros(max(nodes),1);
for i = 1:size(edges,1)
    degree(edges(i,:)) = degree(edges(i,:)) + 1;
end
ends = nodes(degree(nodes)==1);
if numel(ends) ~= 2 || any(degree(nodes)>2)
    error('The fluid wall is not one open, non-branching edge chain.');
end

[~,k] = min(mesh.p(2,ends) + eps*mesh.p(1,ends));
current = ends(k);
used = false(size(edges,1),1);
chain = current;
orderedElement = zeros(size(element));
orderedFace = zeros(size(localFace));
for i = 1:size(edges,1)
    candidate = find(~used & (edges(:,1)==current | edges(:,2)==current));
    if numel(candidate) ~= 1
        error('Unable to order the fluid wall edge chain at segment %d.',i);
    end
    edge = candidate(1);
    used(edge) = true;
    orderedElement(i) = element(edge);
    orderedFace(i) = localFace(edge);
    if edges(edge,1) == current
        current = edges(edge,2);
    else
        current = edges(edge,1);
    end
    chain(end+1) = current; %#ok<AGROW>
end
wall = mesh.p(:,chain).';
end

function vertex = interface_boundary_vertices( ...
        p,t,ymin,xmax,wallStartX,wallEndY,tol)
face = getelemface(2,0);
mask = boundary_face_mask(t);
[localFace,element] = find(mask);
vertex = [];
for k = 1:numel(element)
    nodes = t(face(:,localFace(k)),element(k));
    if all(on_interface(p(:,nodes),ymin,xmax,wallStartX,wallEndY,tol))
        vertex = [vertex; nodes(:)]; %#ok<AGROW>
    end
end
vertex = unique(vertex);
end

function projected = project_to_fluid_wall( ...
        points,fluid,wall,fluidElement,fluidLocalFace)
nface = numel(fluidElement);
faceNodes = cell(nface,1);
for f = 1:nface
    nodes = fluid.dgnodes( ...
        fluid.perm(:,fluidLocalFace(f)),:,fluidElement(f));
    faceNodes{f} = orient_face_nodes(nodes,wall(f,:),wall(f+1,:));
end

projected = zeros(size(points));
options = optimset('Display','off','TolX',1.0e-13);
for j = 1:size(points,1)
    bestDistance = realmax;
    bestPoint = points(j,:);
    for f = 1:nface
        objective = @(xi) sum((evaluate_face(faceNodes{f},fluid,xi)-points(j,:)).^2);
        [xi,distance] = fminbnd(objective,0.0,1.0,options);
        candidates = [0.0; xi; 1.0];
        for c = 1:numel(candidates)
            candidate = evaluate_face(faceNodes{f},fluid,candidates(c));
            candidateDistance = sum((candidate-points(j,:)).^2);
            if candidateDistance < bestDistance
                bestDistance = candidateDistance;
                bestPoint = candidate;
            end
        end
    end
    projected(j,:) = bestPoint;
end
end

function point = evaluate_face(nodes,fluid,xi)
shape = mkshape(fluid.porder,fluid.plocfc,xi,1);
point = shape(:,:,1).'*nodes;
end

function nodes = orient_face_nodes(nodes,first,last)
forward = norm(nodes(1,:)-first) + norm(nodes(end,:)-last);
reverse = norm(nodes(1,:)-last) + norm(nodes(end,:)-first);
if reverse < forward
    nodes = flipud(nodes);
end
end

function write_solid_geo(filename,wall,ymin,xmax,meshSize)
fid = fopen(filename,'w');
if fid < 0
    error('Unable to create temporary Gmsh geometry %s.',filename);
end
cleanup = onCleanup(@() fclose(fid));

nwall = size(wall,1);
for i = 1:nwall
    fprintf(fid,'Point(%d) = {%.17g, %.17g, 0, %.17g};\n', ...
        i,wall(i,1),wall(i,2),meshSize);
end
bottomRight = nwall + 1;
fprintf(fid,'Point(%d) = {%.17g, %.17g, 0, %.17g};\n', ...
    bottomRight,xmax,ymin,meshSize);

fprintf(fid,'Spline(1) = {1');
fprintf(fid,',%d',2:nwall);
fprintf(fid,'};\n');
fprintf(fid,'Line(2) = {%d, %d};\n',nwall,bottomRight);
fprintf(fid,'Line(3) = {%d, 1};\n',bottomRight);
fprintf(fid,'Curve Loop(1) = {1, 2, 3};\n');
fprintf(fid,'Plane Surface(1) = {1};\n');
fprintf(fid,'Mesh.Algorithm = 6;\n');
fprintf(fid,'Mesh.ElementOrder = 1;\n');
fprintf(fid,'Mesh.RecombineAll = 0;\n');
fprintf(fid,'Mesh.MeshSizeMin = %.17g;\n',0.95*meshSize);
fprintf(fid,'Mesh.MeshSizeMax = %.17g;\n',1.05*meshSize);
fprintf(fid,'Mesh.MeshSizeFromPoints = 1;\n');
fprintf(fid,'Mesh.MeshSizeFromCurvature = 0;\n');
fprintf(fid,'Mesh.MeshSizeExtendFromBoundary = 0;\n');
fprintf(fid,'Mesh.Smoothing = 20;\n');
fprintf(fid,'Mesh.Optimize = 1;\n');
fprintf(fid,'Mesh.OptimizeNetgen = 1;\n');
end

function tf = on_interface(q,ymin,xmax,wallStartX,wallEndY,tol)
onBottomInterior = abs(q(2,:)-ymin) < tol & q(1,:) > wallStartX+tol;
onRightInterior = abs(q(1,:)-xmax) < tol & q(2,:) < wallEndY-tol;
tf = ~(onBottomInterior | onRightInterior);
end

function area2 = signed_double_area(p,t)
area2 = (p(1,t(2,:))-p(1,t(1,:))).*(p(2,t(3,:))-p(2,t(1,:))) ...
      - (p(2,t(2,:))-p(2,t(1,:))).*(p(1,t(3,:))-p(1,t(1,:)));
end

function mask = boundary_face_mask(t)
face = getelemface(2,0);
allEdges = sort(reshape(t(face,:),2,[]),1).';
[~,~,ic] = unique(allEdges,'rows');
count = accumarray(ic,1);
mask = reshape(count(ic)==1,size(face,2),size(t,2));
end

function cleanup_gmsh_files(basename)
files = {[basename '.geo'],[basename '.msh']};
for i = 1:numel(files)
    if isfile(files{i})
        delete(files{i});
    end
end
end
