function [geofile, gmshInfo] = windturbine2d_write_gmsh_geometry(filename, farLoop, bladeLoops, opts)
%WINDTURBINE2D_WRITE_GMSH_GEOMETRY Write a circle-with-holes Gmsh geometry.

opts = windturbine2d_background_mesh_options(opts);

[folder,~,ext] = fileparts(filename);
if ~isempty(folder) && ~exist(folder, 'dir')
    mkdir(folder);
end
if ext == ""
    geofile = filename + ".geo";
else
    geofile = string(filename);
end

fid = fopen(geofile, 'w');
if fid < 0
    error('Could not open %s for writing.', geofile);
end

pid = 0;
lid = 0;
loopIds = [];
bladeLineIds = cell(numel(bladeLoops), 1);

[pid, lid, loopId, ~] = local_write_loop(fid, pid, lid, farLoop, ...
    opts.backgroundMeshSizeFar, true);
loopIds(end+1) = loopId; %#ok<AGROW>
for i = 1:numel(bladeLoops)
    [pid, lid, loopId, bladeLineIds{i}] = local_write_loop(fid, pid, lid, ...
        bladeLoops{i}.vertices, opts.backgroundMeshSizeNearBlade, false);
    loopIds(end+1) = loopId; %#ok<AGROW>
end

fprintf(fid, 'Plane Surface(1) = {%s};\n', local_id_list(loopIds));
if opts.preserveInterfaceSegments
    local_write_fixed_interface_curves(fid, bladeLineIds);
end
local_write_mesh_size_controls(fid, bladeLineIds, opts);
local_write_element_type_controls(fid, opts);
fclose(fid);

gmshInfo.mode = opts.backgroundMeshMode;
gmshInfo.elementType = opts.backgroundElementType;
gmshInfo.elemtype = opts.backgroundElemtype;
gmshInfo.bladeLineIds = bladeLineIds;
gmshInfo.farSize = opts.backgroundMeshSizeFar;
gmshInfo.nearBladeSize = opts.backgroundMeshSizeNearBlade;
gmshInfo.interfaceSize = opts.backgroundMeshSizeInterface;
gmshInfo.distMin = opts.backgroundMeshDistMin;
gmshInfo.distMax = opts.backgroundMeshSizeTransition;
gmshInfo.uniformSize = opts.backgroundMeshSize;
gmshInfo.preserveInterfaceSegments = opts.preserveInterfaceSegments;
end

function [pid, lid, loopId, lineIds] = local_write_loop(fid, pid, lid, loop, h, ccw)
if local_polygon_area(loop) < 0 && ccw
    loop = flipud(loop);
elseif local_polygon_area(loop) > 0 && ~ccw
    loop = flipud(loop);
end

n = size(loop, 1);
ids = pid + (1:n);
for i = 1:n
    fprintf(fid, 'Point(%d) = {%.16g, %.16g, 0, %.16g};\n', ids(i), loop(i,1), loop(i,2), h);
end
fprintf(fid, '\n');

lineIds = lid + (1:n);
for i = 1:n
    fprintf(fid, 'Line(%d) = {%d, %d};\n', lineIds(i), ids(i), ids(mod(i,n)+1));
end
fprintf(fid, '\n');

loopId = max(lineIds) + 100000;
fprintf(fid, 'Line Loop(%d) = {%s};\n\n', loopId, local_id_list(lineIds));

pid = ids(end);
lid = lineIds(end);
end

function local_write_fixed_interface_curves(fid, bladeLineIds)
% Preserve blade-loop segmentation exactly: one Gmsh edge per blade edge.
ids = [];
for i = 1:numel(bladeLineIds)
    ids = [ids, bladeLineIds{i}]; %#ok<AGROW>
end
fprintf(fid, '\n// Preserve structured blade-interface segmentation\n');
fprintf(fid, 'Transfinite Curve {%s} = 2;\n\n', local_id_list(ids));
end

function local_write_mesh_size_controls(fid, bladeLineIds, opts)
fprintf(fid, '\n// Background mesh-size controls\n');
fprintf(fid, 'Mesh.MeshSizeMin = %.16g;\n', opts.backgroundMeshMinSize);
fprintf(fid, 'Mesh.MeshSizeMax = %.16g;\n', opts.backgroundMeshMaxSize);
fprintf(fid, 'Mesh.MeshSizeFromPoints = 0;\n');
fprintf(fid, 'Mesh.MeshSizeFromCurvature = 0;\n');
fprintf(fid, 'Mesh.MeshSizeExtendFromBoundary = 0;\n');

switch opts.backgroundMeshMode
    case "uniform"
        fprintf(fid, 'Mesh.MeshSizeMin = %.16g;\n', opts.backgroundMeshSize);
        fprintf(fid, 'Mesh.MeshSizeMax = %.16g;\n', opts.backgroundMeshSize);
    case "graded"
        ids = [];
        for i = 1:numel(bladeLineIds)
            ids = [ids, bladeLineIds{i}]; %#ok<AGROW>
        end
        fprintf(fid, 'Field[1] = Distance;\n');
        fprintf(fid, 'Field[1].CurvesList = {%s};\n', local_id_list(ids));
        fprintf(fid, 'Field[1].Sampling = 200;\n');
        fprintf(fid, 'Field[2] = Threshold;\n');
        fprintf(fid, 'Field[2].InField = 1;\n');
        fprintf(fid, 'Field[2].SizeMin = %.16g;\n', opts.backgroundMeshSizeNearBlade);
        fprintf(fid, 'Field[2].SizeMax = %.16g;\n', opts.backgroundMeshSizeFar);
        fprintf(fid, 'Field[2].DistMin = %.16g;\n', opts.backgroundMeshDistMin);
        fprintf(fid, 'Field[2].DistMax = %.16g;\n', opts.backgroundMeshSizeTransition);
        fprintf(fid, 'Background Field = 2;\n');
end
fprintf(fid, '\n');
end

function local_write_element_type_controls(fid, opts)
fprintf(fid, '\n// Background element controls\n');
switch opts.backgroundElementType
    case "tri"
        fprintf(fid, 'Mesh.Algorithm = 6;\n'); % Frontal-Delaunay triangles.
    case "quad"
        fprintf(fid, 'Mesh.Algorithm = 6;\n'); % Frontal-Delaunay, then blossom recombination.
        fprintf(fid, 'Recombine Surface {1};\n');
        fprintf(fid, 'Mesh.RecombineAll = 1;\n');
        fprintf(fid, 'Mesh.RecombinationAlgorithm = 2;\n');
        fprintf(fid, 'Mesh.Smoothing = 20;\n');
        fprintf(fid, 'Mesh.Optimize = 1;\n');
end
fprintf(fid, '\n');
end

function s = local_id_list(ids)
s = sprintf('%d,', ids);
s = s(1:end-1);
end

function area = local_polygon_area(p)
area = 0.5*sum(p(:,1).*p([2:end 1],2) - p(:,2).*p([2:end 1],1));
end
