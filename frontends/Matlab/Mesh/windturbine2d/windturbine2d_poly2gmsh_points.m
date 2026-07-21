function pv = windturbine2d_poly2gmsh_points(farLoop, bladeLoops)
%WINDTURBINE2D_POLY2GMSH_POINTS Convert turbine loops to poly2gmsh input.
%
% The returned array uses NaN separator rows:
%   [outer loop; NaN NaN; hole 1; NaN NaN; hole 2; ...]
% Updated poly2gmsh routines interpret these rows as independent Gmsh loops.

if nargin < 2
    error('Expected farLoop and bladeLoops.');
end
if size(farLoop, 2) ~= 2
    error('farLoop must be an n-by-2 array.');
end

farLoop = local_remove_closing_duplicate(farLoop);
if local_polygon_area(farLoop) < 0
    farLoop = flipud(farLoop);
end

pv = farLoop;
for ib = 1:numel(bladeLoops)
    loop = bladeLoops{ib}.vertices;
    loop = local_remove_closing_duplicate(loop);
    if size(loop, 2) ~= 2
        error('bladeLoops{%d}.vertices must be an n-by-2 array.', ib);
    end
    % Gmsh convention: outer loop counterclockwise, hole loops clockwise.
    if local_polygon_area(loop) > 0
        loop = flipud(loop);
    end
    pv = [pv; NaN NaN; loop]; %#ok<AGROW>
end
end

function p = local_remove_closing_duplicate(p)
if size(p, 1) > 1 && norm(p(1,:) - p(end,:)) <= 100*eps(max(1, norm(p(1,:))))
    p = p(1:end-1,:);
end
end

function area = local_polygon_area(p)
area = 0.5*sum(p(:,1).*p([2:end 1],2) - p(:,2).*p([2:end 1],1));
end
