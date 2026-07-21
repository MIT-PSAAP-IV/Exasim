function [mesh, p, t, geofile, info, farLoop] = windturbine2d_background_mesh(bladeLoops, opts)
%WINDTURBINE2D_BACKGROUND_MESH Generate the unstructured far-field mesh.

opts = windturbine2d_background_mesh_options(opts);
[opts, interfaceInfo] = windturbine2d_interface_sizing(bladeLoops, opts);
[farLoop, farInfo] = windturbine2d_farfield_loop(opts);
if opts.plotFarfieldBoundary
    local_plot_farfield_boundary(farLoop, farInfo);
end

if ~exist(opts.workdir, 'dir')
    mkdir(opts.workdir);
end
base = fullfile(opts.workdir, 'windturbine2d_background');
[geofile, gmshInfo] = windturbine2d_write_gmsh_geometry(base, farLoop, bladeLoops, opts);

info.gmsh = gmshInfo;
info.farfield = farInfo;
info.interface = interfaceInfo;

if ~opts.runGmsh
    mesh = [];
    p = [];
    t = [];
    info.vertices = 0;
    info.elements = 0;
    info.minElementSize = NaN;
    info.maxElementSize = NaN;
    return;
end

pde.gmsh = opts.gmsh;
pde.version = opts.gmshVersion;
[pg, tg] = gmshcall(pde, base, 2, opts.backgroundElemtype);
info.gmshElementCounts = local_gmsh_element_counts([char(base) '.msh']);
p = pg';
t = tg';

bndexpr = {sprintf('all(abs(sqrt(sum(p.^2,2)) - %.16g) < %.16g)', ...
    opts.Rfar, 10*opts.backgroundMeshSizeFar), 'true'};
mesh = mkmesh(p, t, opts.porder, bndexpr, opts.backgroundElemtype, opts.nodetype);
info.vertices = size(p, 1);
info.elements = size(t, 1);
info.interfaceConformity = windturbine2d_interface_conformity(p, t, bladeLoops, opts);
[info.minElementSize, info.maxElementSize] = local_element_size_range(p, t);
[info.minElementQuality, info.avgElementQuality, info.maxElementQuality, ...
    info.elementQualityStats] = ...
    local_element_quality(p, t, opts.backgroundElemtype);
if opts.backgroundElemtype == 1 && info.gmshElementCounts.triangles > 0
    warning('Requested quadrilateral background mesh but Gmsh wrote %d triangles.', ...
        info.gmshElementCounts.triangles);
end
end

function local_plot_farfield_boundary(farLoop, info)
figure;
plot(farLoop([1:end 1],1), farLoop([1:end 1],2), '-o', ...
    'LineWidth', 1, 'MarkerSize', 4);
axis equal;
title(sprintf('Far-field boundary: %d points, mean h = %.4g', ...
    info.pointCount, info.meanEdgeLength));
xlabel('x');
ylabel('y');
end

function [hmin, hmax] = local_element_size_range(p, t)
if size(t, 2) == 3
    edges = [t(:,[1 2]); t(:,[2 3]); t(:,[3 1])];
elseif size(t, 2) == 4
    edges = [t(:,[1 2]); t(:,[2 3]); t(:,[3 4]); t(:,[4 1])];
else
    error('Unsupported background element with %d vertices.', size(t, 2));
end
d = p(edges(:,1),:) - p(edges(:,2),:);
h = sqrt(sum(d.^2, 2));
hmin = min(h);
hmax = max(h);
end

function counts = local_gmsh_element_counts(filename)
counts.triangles = 0;
counts.quadrilaterals = 0;
fid = fopen(filename, 'r');
if fid < 0
    return;
end
cleanup = onCleanup(@() fclose(fid));
while ~feof(fid)
    line = strtrim(fgetl(fid));
    if strcmp(line, '$Elements')
        ne = fscanf(fid, '%d', 1);
        fgetl(fid);
        for i = 1:ne
            row = sscanf(fgetl(fid), '%d');
            if numel(row) >= 2
                if row(2) == 2
                    counts.triangles = counts.triangles + 1;
                elseif row(2) == 3
                    counts.quadrilaterals = counts.quadrilaterals + 1;
                end
            end
        end
        return;
    end
end
end

function [qmin, qavg, qmax, stats] = local_element_quality(p, t, elemtype)
q = zeros(size(t, 1), 1);
if elemtype == 0
    for i = 1:size(t, 1)
        x = p(t(i,:), :);
        e12 = norm(x(2,:) - x(1,:));
        e23 = norm(x(3,:) - x(2,:));
        e31 = norm(x(1,:) - x(3,:));
        area = 0.5*abs(det([x(2,:) - x(1,:); x(3,:) - x(1,:)]));
        q(i) = 4*sqrt(3)*area/(e12^2 + e23^2 + e31^2);
    end
elseif elemtype == 1
    for i = 1:size(t, 1)
        x = p(t(i,:), :);
        q(i) = local_quad_quality(x);
    end
else
    error('Unsupported background elemtype %d.', elemtype);
end
qmin = min(q);
qavg = mean(q);
qmax = max(q);
stats.p01 = prctile(q, 1);
stats.p05 = prctile(q, 5);
stats.p10 = prctile(q, 10);
stats.below01 = nnz(q < 0.1);
stats.below02 = nnz(q < 0.2);
[stats.worstQuality, stats.worstElement] = min(q);
end

function q = local_quad_quality(x)
edges = x([2 3 4 1], :) - x;
lengths = sqrt(sum(edges.^2, 2));
scaledJacobian = zeros(4, 1);
for j = 1:4
    a = x(j,:) - x(mod(j-2,4)+1,:);
    b = x(mod(j,4)+1,:) - x(j,:);
    scaledJacobian(j) = det([a; b])/(norm(a)*norm(b));
end
q = min(abs(scaledJacobian));
if any(lengths <= 0)
    q = 0;
end
end
