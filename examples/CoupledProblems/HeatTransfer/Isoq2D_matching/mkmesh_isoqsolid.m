function [mesh, info] = mkmesh_isoqsolid(pde, fluid, meshSize)
%MKMESH_ISOQSOLID Generate the triangular solid below the fluid wall.
% Gmsh creates the linear triangle topology. The interface vertices and DG
% nodes are then copied from the quadrilateral fluid mesh exactly.

if nargin < 3 || isempty(meshSize)
    meshSize = 0.002;
end
if size(fluid.t,1) ~= 4
    error('mkmesh_isoqsolid expects a quadrilateral fluid mesh.');
end

[wall, fluidElement, fluidLocalFace] = ordered_fluid_wall(fluid, 4);
ymin = min(fluid.p(2,:));
xmax = max(fluid.p(1,:));
scale = max(1.0, max(max(fluid.p,[],2)-min(fluid.p,[],2)));
tol = 1.0e-10*scale;

if abs(wall(1,2)-ymin) > tol || abs(wall(end,1)-xmax) > tol
    error('The fluid wall must run from the minimum-y point to the maximum-x boundary.');
end
if meshSize <= 0
    error('The solid mesh size must be positive.');
end

basename = tempname;
cleanup = onCleanup(@() cleanup_gmsh_files(basename));
write_solid_geo([basename '.geo'], wall, ymin, xmax, meshSize);
[p, t] = gmshcall(pde, basename, 2, 0);

% Restore the exact fluid coordinates at all Gmsh interface vertices.
wallVertex = zeros(size(wall,1),1);
for i = 1:size(wall,1)
    [distance, node] = min(vecnorm(p-wall(i,:).',2,1));
    if distance > tol
        error('Gmsh did not preserve fluid wall vertex %d (distance %.3e).', i, distance);
    end
    p(:,node) = wall(i,:).';
    wallVertex(i) = node;
end
if numel(unique(wallVertex)) ~= numel(wallVertex)
    error('Multiple fluid wall vertices map to one Gmsh node.');
end

% Enforce counter-clockwise triangle orientation before constructing the mesh.
area2 = (p(1,t(2,:))-p(1,t(1,:))).*(p(2,t(3,:))-p(2,t(1,:))) ...
      - (p(2,t(2,:))-p(2,t(1,:))).*(p(1,t(3,:))-p(1,t(1,:)));
if any(abs(area2) <= eps(scale^2))
    error('Gmsh generated a degenerate solid triangle.');
end
flip = area2 < 0;
t([2 3],flip) = t([3 2],flip);

mesh = mkmesh(p.',t.',pde.porder,{'true'},0,1);
mesh.p = mesh.p.';
mesh.t = mesh.t.';
mesh.boundaryexpr = { ...
    @(q) on_wall_vertex(q,wall,tol), ...
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

[mesh, maxGridMismatch, maxDGMismatch] = copy_interface_nodes( ...
    mesh, fluid, wall, wallVertex, fluidElement, fluidLocalFace, tol);

interfaceFaces = nnz(mesh.f == 1);
if interfaceFaces ~= size(wall,1)-1
    error('Expected %d solid interface faces, but Gmsh produced %d.', ...
        size(wall,1)-1, interfaceFaces);
end
if abs(min(mesh.p(2,:))-ymin) > tol || abs(max(mesh.p(1,:))-xmax) > tol
    error('The solid mesh does not preserve the fluid ymin/xmax extents.');
end

info.interfaceFaces = interfaceFaces;
info.triangles = size(mesh.t,2);
info.ymin = min(mesh.p(2,:));
info.xmax = max(mesh.p(1,:));
info.maxGridMismatch = maxGridMismatch;
info.maxDGMismatch = maxDGMismatch;

fprintf(['Gmsh solid mesh: %d triangles, %d interface faces; ' ...
    'grid mismatch %.3e, DG mismatch %.3e; ymin %.16g, xmax %.16g\n'], ...
    info.triangles, info.interfaceFaces, maxGridMismatch, maxDGMismatch, ...
    info.ymin, info.xmax);
end

function [wall, element, localFace] = ordered_fluid_wall(mesh, boundary)
face = getelemface(2,1);
[localFace, element] = find(mesh.f == boundary);
if isempty(element)
    error('The fluid mesh has no faces on wall boundary %d.', boundary);
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
        error('Unable to order the fluid wall edge chain at segment %d.', i);
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
element = orderedElement;
localFace = orderedFace;
end

function write_solid_geo(filename, wall, ymin, xmax, meshSize)
fid = fopen(filename,'w');
if fid < 0
    error('Unable to create temporary Gmsh geometry %s.', filename);
end
cleanup = onCleanup(@() fclose(fid));

nwall = size(wall,1);
for i = 1:nwall
    fprintf(fid,'Point(%d) = {%.17g, %.17g, 0, %.17g};\n', ...
        i, wall(i,1), wall(i,2), meshSize);
end
bottomRight = nwall + 1;
fprintf(fid,'Point(%d) = {%.17g, %.17g, 0, %.17g};\n', ...
    bottomRight, xmax, ymin, meshSize);

for i = 1:nwall-1
    fprintf(fid,'Line(%d) = {%d, %d};\n',i,i,i+1);
end
rightLine = nwall;
bottomLine = nwall + 1;
fprintf(fid,'Line(%d) = {%d, %d};\n',rightLine,nwall,bottomRight);
fprintf(fid,'Line(%d) = {%d, %d};\n',bottomLine,bottomRight,1);

fprintf(fid,'Curve Loop(1) = {');
fprintf(fid,'%d,',1:nwall);
fprintf(fid,'%d};\n',bottomLine);
fprintf(fid,'Plane Surface(1) = {1};\n');
fprintf(fid,'Transfinite Curve {');
fprintf(fid,'%d',1);
fprintf(fid,',%d',2:nwall-1);
fprintf(fid,'} = 2;\n');
fprintf(fid,'Mesh.Algorithm = 6;\n');
fprintf(fid,'Mesh.ElementOrder = 1;\n');
fprintf(fid,'Mesh.RecombineAll = 0;\n');
fprintf(fid,'Mesh.MeshSizeMin = %.17g;\n',0.25*meshSize);
fprintf(fid,'Mesh.MeshSizeMax = %.17g;\n',meshSize);
end

function tf = on_wall_vertex(q, wall, tol)
tf = false(1,size(q,2));
for j = 1:size(q,2)
    tf(j) = min(vecnorm(wall-q(:,j).',2,2)) <= tol;
end
end

function mask = boundary_face_mask(t)
face = getelemface(2,0);
allEdges = sort(reshape(t(face,:),2,[]),1).';
[~,~,ic] = unique(allEdges,'rows');
count = accumarray(ic,1);
mask = reshape(count(ic)==1,size(face,2),size(t,2));
end

function [mesh, maxGridMismatch, maxDGMismatch] = copy_interface_nodes( ...
        mesh, fluid, wall, wallVertex, fluidElement, fluidLocalFace, tol)
face = getelemface(2,0);
nsegment = size(wall,1)-1;
seen = false(nsegment,1);
maxGridMismatch = 0.0;
maxDGMismatch = 0.0;

[solidLocalFace, solidElement] = find(mesh.f == 1);
for k = 1:numel(solidElement)
    e = solidElement(k);
    lf = solidLocalFace(k);
    vertices = mesh.t(face(:,lf),e);
    segment = find_segment(vertices,wallVertex);
    if segment < 1 || seen(segment)
        error('A Gmsh interface edge does not match exactly one fluid wall segment.');
    end
    seen(segment) = true;

    target = fluid.dgnodes(fluid.perm(:,fluidLocalFace(segment)),:,fluidElement(segment));
    target = orient_face_nodes(target,wall(segment,:),wall(segment+1,:),tol);
    if vertices(1) == wallVertex(segment+1)
        target = flipud(target);
    elseif vertices(1) ~= wallVertex(segment)
        error('Unexpected Gmsh interface-edge orientation.');
    end

    mesh.dgnodes(mesh.perm(:,lf),:,e) = target;
    grid = mesh.p(:,vertices).';
    exact = [wall(segment,:); wall(segment+1,:)];
    if vertices(1) == wallVertex(segment+1)
        exact = flipud(exact);
    end
    maxGridMismatch = max(maxGridMismatch,max(vecnorm(grid-exact,2,2)));
    copied = mesh.dgnodes(mesh.perm(:,lf),:,e);
    maxDGMismatch = max(maxDGMismatch,max(vecnorm(copied-target,2,2)));
end

if ~all(seen)
    error('Gmsh omitted %d fluid wall segments from the solid interface.',nnz(~seen));
end
end

function segment = find_segment(vertices, wallVertex)
segment = 0;
for i = 1:numel(wallVertex)-1
    if isequal(sort(vertices(:)),sort(wallVertex(i:i+1)))
        segment = i;
        return;
    end
end
end

function nodes = orient_face_nodes(nodes, first, last, tol)
if norm(nodes(1,:)-first) <= tol && norm(nodes(end,:)-last) <= tol
    return;
end
if norm(nodes(1,:)-last) <= tol && norm(nodes(end,:)-first) <= tol
    nodes = flipud(nodes);
    return;
end
error('Fluid face DG-node endpoints do not match the wall segment.');
end

function cleanup_gmsh_files(basename)
files = {[basename '.geo'],[basename '.msh']};
for i = 1:numel(files)
    if isfile(files{i})
        delete(files{i});
    end
end
end
