function [p, t] = gmsh_circlewithholes(pc, ph, elemtype, dmin, dmax, alpha, beta, gamma, Gmsh)
%GMSH_CIRCLEWITHHOLES Mesh a circular domain with polygonal holes.
%
%   [p,t] = gmsh_circlewithholes(pc, ph, elemtype, dmin, dmax, alpha, beta, gamma, Gmsh)
%
% pc is an ordered outer circular boundary. ph is a cell array of ordered
% polygonal hole boundaries.  p is returned as 2-by-np and t as nve-by-ne,
% matching the Exasim Matlab mesh utilities.

[pc, ph, elemtype, dmin, dmax, alpha, beta, gamma, Gmsh] = ...
    local_validate_inputs(pc, ph, elemtype, dmin, dmax, alpha, beta, gamma, Gmsh);

hc = local_boundary_point_sizes(pc);
hc = beta*hc;
hh = cellfun(@local_boundary_point_sizes, ph, 'UniformOutput', false);
for j = 1:length(hh)
  hh{j} = hh{j}*gamma;
end
hmax = max([hc; vertcat(hh{:})]);
hglobalmin = min([hc; vertcat(hh{:})]);

base = pwd() + "/gmsh";
geofile = base + ".geo";

local_write_geo(geofile, pc, hc, ph, hh, elemtype, dmin, dmax, ...
    alpha, hglobalmin, hmax);

pde.gmsh = Gmsh;
pde.version = ""; 
[p,t] = gmshcall(pde, base, 2, elemtype);

% cmd = sprintf('"%s" "%s" -2 -format msh2 -o "%s"', ...
%     char(Gmsh), geofile, mshfile);
% [status, output] = system(cmd);
% if status ~= 0
%     error('Gmsh failed with status %d.\nCommand: %s\nOutput:\n%s', ...
%         status, cmd, output);
% end
% if ~exist(mshfile, 'file')
%     error('Gmsh completed but did not create expected mesh file: %s', mshfile);
% end
% 
% [p, t, counts] = local_read_msh2(mshfile, elemtype);
% if isempty(t)
%     error('Gmsh did not generate any requested 2-D elements.');
% end
% if elemtype == 1 && counts.triangles > 0
%     error(['Gmsh generated a mixed mesh for elemtype=1: %d triangles and ' ...
%         '%d quadrilaterals. A purely quadrilateral mesh could not be generated.'], ...
%         counts.triangles, counts.quadrilaterals);
% end

% [p, t] = local_remove_unused_nodes(p, t);
% local_check_connectivity(p, t);
end

function [pc, ph, elemtype, dmin, dmax, alpha, beta, gamma, Gmsh] = ...
    local_validate_inputs(pc, ph, elemtype, dmin, dmax, alpha, beta, gamma, Gmsh)
if ~(isnumeric(pc) && isreal(pc) && size(pc,2) == 2 && size(pc,1) >= 3)
    error('pc must be a real numeric npc-by-2 array with at least three points.');
end
pc = local_remove_closure_point(pc, 'pc');
if size(pc,1) < 3
    error('pc must contain at least three distinct boundary points.');
end
if local_polygon_area(pc) < 0
    pc = flipud(pc);
end

if ~iscell(ph)
    error('ph must be a cell array of hole-boundary arrays.');
end
for k = 1:numel(ph)
    q = ph{k};
    if ~(isnumeric(q) && isreal(q) && size(q,2) == 2 && size(q,1) >= 3)
        error('ph{%d} must be a real numeric nh-by-2 array with at least three points.', k);
    end
    q = local_remove_closure_point(q, sprintf('ph{%d}', k));
    if size(q,1) < 3
        error('ph{%d} must contain at least three distinct boundary points.', k);
    end
    if local_polygon_area(q) > 0
        q = flipud(q);
    end
    ph{k} = q;
end
if ~(isscalar(elemtype) && isnumeric(elemtype) && any(elemtype == [0 1]))
    error('elemtype must be 0 for triangles or 1 for quadrilaterals.');
end
elemtype = double(elemtype);
if ~(isnumeric(dmin) && isscalar(dmin) && isfinite(dmin) && dmin > 0)
    error('dmin must be a positive finite scalar.');
end
if ~(isnumeric(dmax) && isscalar(dmax) && isfinite(dmax) && dmax > dmin)
    error('dmax must be a positive finite scalar greater than dmin.');
end
if ~(isnumeric(alpha) && isscalar(alpha) && isfinite(alpha) && alpha > 0)
    error('alpha must be a positive finite scalar.');
end
if ~(isnumeric(beta) && isscalar(beta) && isfinite(beta) && beta > 0)
    error('beta must be a positive finite scalar.');
end
if ~(isnumeric(gamma) && isscalar(gamma) && isfinite(gamma) && gamma > 0)
    error('gamma must be a positive finite scalar.');
end
if ~((ischar(Gmsh) || isstring(Gmsh)) && strlength(string(Gmsh)) > 0)
    error('Gmsh must be a nonempty character vector or string.');
end
Gmsh = string(Gmsh);
end

function p = local_remove_closure_point(p, name)
tol = 100*eps(max(1, max(abs(p(:)))));
if size(p,1) > 1 && norm(p(1,:) - p(end,:)) <= tol
    p = p(1:end-1,:);
end
edge = sqrt(sum((p([2:end 1],:) - p).^2, 2));
if any(edge <= tol)
    error('%s contains duplicate or nearly duplicate consecutive points.', name);
end
end

function h = local_boundary_point_sizes(p)
prev = p - p([end 1:end-1], :);
next = p([2:end 1], :) - p;
h = 0.5*(sqrt(sum(prev.^2, 2)) + sqrt(sum(next.^2, 2)));
end

function local_write_geo(fname, pc, hc, ph, hh, elemtype, dmin, dmax, ...
    alpha, hglobalmin, hmax)
fid = fopen(fname, 'w');
if fid < 0
    error('Could not open temporary Gmsh geometry file: %s', fname);
end
closer = onCleanup(@() fclose(fid));

fprintf(fid, 'SetFactory("Built-in");\n');
fprintf(fid, 'Mesh.MeshSizeMin = %.16g;\n', hglobalmin);
fprintf(fid, 'Mesh.MeshSizeMax = %.16g;\n', hmax);
fprintf(fid, 'Mesh.MeshSizeFromPoints = 1;\n');
fprintf(fid, 'Mesh.MeshSizeFromCurvature = 0;\n');
fprintf(fid, 'Mesh.MeshSizeExtendFromBoundary = 0;\n\n');

pid = 0;
lid = 0;
[pid, lid, outerLoop, ~] = local_write_loop(fid, pid, lid, pc, hc, false);
loopIds = outerLoop;
holePointIds = [];
holeSizes = [];
for k = 1:numel(ph)
    [pid, lid, holeLoop, ~, pointIds] = ...
        local_write_loop(fid, pid, lid, ph{k}, hh{k}, true);
    loopIds(end+1) = holeLoop; %#ok<AGROW>
    holePointIds = [holePointIds; pointIds(:)]; %#ok<AGROW>
    holeSizes = [holeSizes; hh{k}(:)]; %#ok<AGROW>
end

fprintf(fid, 'Plane Surface(1) = {%s};\n', local_id_list(loopIds));

if ~isempty(holePointIds)
    local_write_pointwise_fields(fid, holePointIds, holeSizes, hmax, ...
        dmin, dmax, alpha);
end

switch elemtype
    case 0
        fprintf(fid, 'Mesh.Algorithm = 8;\n');
    case 1
        fprintf(fid, 'Mesh.Algorithm = 8;\n');
        fprintf(fid, 'Recombine Surface {1};\n');
        fprintf(fid, 'Mesh.RecombineAll = 1;\n');
        %fprintf(fid, 'Mesh.RecombinationAlgorithm = 2;\n');
        fprintf(fid, 'Mesh.Smoothing = 20; \n');
        fprintf(fid, 'Mesh.Optimize = 1;\n');
end
end

function [pid, lid, loopId, lineIds, pointIds] = local_write_loop(fid, pid, lid, p, h, lockSegments)
n = size(p, 1);
pointIds = pid + (1:n);
for i = 1:n
    fprintf(fid, 'Point(%d) = {%.16g, %.16g, 0, %.16g};\n', ...
        pointIds(i), p(i,1), p(i,2), h(i));
end
fprintf(fid, '\n');

lineIds = lid + (1:n);
for i = 1:n
    fprintf(fid, 'Line(%d) = {%d, %d};\n', lineIds(i), ...
        pointIds(i), pointIds(mod(i,n)+1));
end
if lockSegments
    % Force one mesh edge per polygon segment.  The two transfinite nodes
    % are the supplied segment endpoints, so Gmsh cannot insert extra nodes
    % on hole boundaries during 1-D meshing.
    fprintf(fid, 'Transfinite Curve {%s} = 2;\n', local_id_list(lineIds));
end
loopId = max(lineIds) + 100000;
fprintf(fid, 'Line Loop(%d) = {%s};\n\n', loopId, local_id_list(lineIds));
pid = pointIds(end);
lid = lineIds(end);
end

function local_write_pointwise_fields(fid, pointIds, hmin, hmax, dmin, dmax, alpha)
fprintf(fid, '// Pointwise hole-distance mesh-size fields\n');
smoothFields = zeros(numel(pointIds), 1);
fieldId = 0;
for i = 1:numel(pointIds)
    fieldId = fieldId + 1;
    distanceField = fieldId;
    fprintf(fid, 'Field[%d] = Distance;\n', distanceField);
    fprintf(fid, 'Field[%d].PointsList = {%d};\n', distanceField, pointIds(i));

    fieldId = fieldId + 1;
    smoothFields(i) = fieldId;
    expr = local_hfield_expression(sprintf('F%d', distanceField), ...
        hmin(i), hmax, dmin, dmax, alpha);
    fprintf(fid, 'Field[%d] = MathEval;\n', fieldId);
    fprintf(fid, 'Field[%d].F = "%s";\n', fieldId, expr);
end
fieldId = fieldId + 1;
fprintf(fid, 'Field[%d] = Min;\n', fieldId);
fprintf(fid, 'Field[%d].FieldsList = {%s};\n', fieldId, local_id_list(smoothFields));
fprintf(fid, 'Background Field = %d;\n\n', fieldId);
end

function expr = local_hfield_expression(dexpr, hmin, hmax, dmin, dmax, alpha)
slope = (hmax - hmin)/(dmax - dmin);
piValue = sprintf('%.16g', pi);
offset = -atan(alpha)/pi + 0.5;
x = sprintf('(%.16g*(%s-%.16g))', slope, dexpr, dmin);
lmaxx = sprintf('((%s)*(atan(%.16g*(%s))/%s+0.5)+%.16g)', ...
    x, alpha, x, piValue, offset);
h0 = sprintf('(%.16g+%s)', hmin, lmaxx);
y = sprintf('(%s-%.16g)', h0, hmax);
lmaxy = sprintf('((%s)*(atan(%.16g*(%s))/%s+0.5)+%.16g)', ...
    y, alpha, y, piValue, offset);
expr = sprintf('(%s-%s)', h0, lmaxy);
end

function [p, t, counts] = local_read_msh2(fname, elemtype)
fid = fopen(fname, 'r');
if fid < 0
    error('Cannot open Gmsh mesh file: %s', fname);
end
cleanup = onCleanup(@() fclose(fid));

local_readuntil(fid, '$Nodes');
np = fscanf(fid, '%d', 1);
nodes = zeros(3, np);
nodeTags = zeros(np, 1);
for i = 1:np
    nodeTags(i) = fscanf(fid, '%d', 1);
    nodes(:,i) = fscanf(fid, '%f', 3);
    fgetl(fid);
end

local_readuntil(fid, '$Elements');
ne = fscanf(fid, '%d', 1);
fgetl(fid);
tri = zeros(3, ne);
quad = zeros(4, ne);
counts.triangles = 0;
counts.quadrilaterals = 0;
for i = 1:ne
    row = sscanf(fgetl(fid), '%d');
    if numel(row) < 4
        continue;
    end
    etype = row(2);
    ntags = row(3);
    conn = row((4+ntags):end);
    if etype == 2 && numel(conn) == 3
        counts.triangles = counts.triangles + 1;
        tri(:,counts.triangles) = conn(:);
    elseif etype == 3 && numel(conn) == 4
        counts.quadrilaterals = counts.quadrilaterals + 1;
        quad(:,counts.quadrilaterals) = conn(:);
    end
end

if elemtype == 0
    ttags = tri(:,1:counts.triangles);
else
    ttags = quad(:,1:counts.quadrilaterals);
end
[tf, loc] = ismember(ttags, nodeTags);
if any(~tf(:))
    error('Mesh connectivity references node tags not present in $Nodes.');
end
p = nodes(1:2,:);
t = loc;
end

function [p, t] = local_remove_unused_nodes(p, t)
used = unique(t(:));
map = zeros(1, size(p,2));
map(used) = 1:numel(used);
p = p(:, used);
t = map(t);
end

function local_check_connectivity(p, t)
if any(t(:) < 1) || any(t(:) > size(p,2))
    error('Invalid mesh connectivity after node renumbering.');
end
if size(unique(t', 'rows'), 1) ~= size(t, 2)
    error('Mesh contains duplicate elements.');
end
if size(t,1) == 3
    area = local_triangle_area(p, t);
elseif size(t,1) == 4
    area = local_quad_area(p, t);
else
    error('Unsupported element size %d.', size(t,1));
end
if any(~isfinite(area)) || any(abs(area) <= 100*eps(max(1, max(abs(p(:))))))
    error('Mesh contains zero-area or invalid elements.');
end
end

function a = local_triangle_area(p, t)
x1 = p(:,t(1,:));
x2 = p(:,t(2,:));
x3 = p(:,t(3,:));
a = 0.5*((x2(1,:)-x1(1,:)).*(x3(2,:)-x1(2,:)) - ...
    (x2(2,:)-x1(2,:)).*(x3(1,:)-x1(1,:)));
end

function a = local_quad_area(p, t)
a = local_triangle_area(p, t([1 2 3],:)) + ...
    local_triangle_area(p, t([1 3 4],:));
end

function local_readuntil(fid, marker)
while ~feof(fid)
    line = strtrim(fgetl(fid));
    if strcmp(line, marker)
        return;
    end
end
error('Could not find section %s in Gmsh mesh file.', marker);
end

function s = local_id_list(ids)
s = sprintf('%d,', ids);
s = s(1:end-1);
end

function area = local_polygon_area(p)
area = 0.5*sum(p(:,1).*p([2:end 1],2) - p(:,2).*p([2:end 1],1));
end

function local_cleanup(files)
for i = 1:numel(files)
    if exist(files{i}, 'file')
        delete(files{i});
    end
end
end
