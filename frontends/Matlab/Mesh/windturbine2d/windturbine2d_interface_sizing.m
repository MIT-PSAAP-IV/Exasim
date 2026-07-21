function [opts, info] = windturbine2d_interface_sizing(bladeLoops, opts)
%WINDTURBINE2D_INTERFACE_SIZING Derive background sizing from blade loops.
%
% The blade loops are hard geometry constraints for Gmsh.  The background
% mesh size next to those loops must therefore be comparable to the loop
% edge spacing, not to the much coarser far-field size.

opts = windturbine2d_background_mesh_options(opts);

info = repmat(struct('points', [], 'minEdge', [], 'meanEdge', [], ...
    'medianEdge', [], 'maxEdge', [], 'maxToMin', [], 'duplicates', [], ...
    'signedArea', [], 'orientation', "", 'minTurningAngle', []), ...
    numel(bladeLoops), 1);

allEdges = [];
tol = max(1e-12, 100*eps(max(1, opts.Rfar)));
for i = 1:numel(bladeLoops)
    loop = bladeLoops{i}.vertices;
    if size(loop, 1) < 4
        error('Blade loop %d has too few points.', i);
    end

    edge = sqrt(sum(diff(loop([1:end 1], :)).^2, 2));
    if any(edge <= tol)
        error('Blade loop %d contains zero-length or nearly zero-length edges.', i);
    end

    info(i).points = size(loop, 1);
    info(i).minEdge = min(edge);
    info(i).meanEdge = mean(edge);
    info(i).medianEdge = median(edge);
    info(i).maxEdge = max(edge);
    info(i).maxToMin = info(i).maxEdge/info(i).minEdge;
    info(i).duplicates = local_duplicate_count(loop, tol);
    info(i).signedArea = local_polygon_area(loop);
    if info(i).signedArea >= 0
        info(i).orientation = "ccw";
    else
        info(i).orientation = "cw";
    end
    info(i).minTurningAngle = local_min_turning_angle(loop);

    allEdges = [allEdges; edge(:)]; %#ok<AGROW>
end

robustInterfaceSize = median(allEdges);
if isempty(opts.backgroundMeshSizeInterface)
    opts.backgroundMeshSizeInterface = robustInterfaceSize;
else
    opts.backgroundMeshSizeInterface = min(opts.backgroundMeshSizeInterface, ...
        opts.backgroundMeshSizeNearBlade);
end

% Keep the legacy near-blade option as an upper bound, but use the actual
% interface spacing for the finest Gmsh size next to constrained blade loops.
opts.backgroundMeshSizeNearBlade = min(opts.backgroundMeshSizeNearBlade, ...
    opts.backgroundMeshSizeInterface);
opts.backgroundMeshMinSize = min(opts.backgroundMeshMinSize, ...
    opts.backgroundMeshSizeNearBlade);

if isempty(opts.backgroundMeshDistMin)
    opts.backgroundMeshDistMin = max(2*opts.backgroundMeshSizeNearBlade, ...
        min(0.25, 5*robustInterfaceSize));
end
if opts.backgroundMeshDistMin >= opts.backgroundMeshSizeTransition
    opts.backgroundMeshSizeTransition = 4*opts.backgroundMeshDistMin;
end

fprintf('Blade-interface boundary diagnostics\n');
for i = 1:numel(info)
    fprintf(['  Blade %d: n=%d, hmin/hmean/hmedian/hmax = ' ...
        '%.6g / %.6g / %.6g / %.6g, hmax/hmin = %.6g, ' ...
        'duplicates = %d, area = %.6g (%s), min turn = %.6g deg\n'], ...
        i, info(i).points, info(i).minEdge, info(i).meanEdge, ...
        info(i).medianEdge, info(i).maxEdge, info(i).maxToMin, ...
        info(i).duplicates, info(i).signedArea, info(i).orientation, ...
        info(i).minTurningAngle*180/pi);
end
fprintf('  backgroundMeshSizeInterface = %.6g\n', opts.backgroundMeshSizeInterface);
fprintf('  backgroundMeshSizeNearBlade = %.6g\n', opts.backgroundMeshSizeNearBlade);
fprintf('  backgroundMeshDistMin       = %.6g\n', opts.backgroundMeshDistMin);
fprintf('  backgroundMeshSizeTransition = %.6g\n', opts.backgroundMeshSizeTransition);
end

function ndup = local_duplicate_count(p, tol)
q = round(p/tol)*tol;
[~, ia] = unique(q, 'rows', 'stable');
ndup = size(p, 1) - numel(ia);
end

function area = local_polygon_area(p)
area = 0.5*sum(p(:,1).*p([2:end 1],2) - p(:,2).*p([2:end 1],1));
end

function angle = local_min_turning_angle(p)
prev = p - p([end 1:end-1], :);
next = p([2:end 1], :) - p;
den = sqrt(sum(prev.^2, 2)).*sqrt(sum(next.^2, 2));
c = sum(prev.*next, 2)./den;
c = max(-1, min(1, c));
angle = min(acos(c));
end
