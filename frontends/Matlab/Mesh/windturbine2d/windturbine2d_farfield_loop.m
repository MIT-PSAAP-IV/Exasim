function [farLoop, info] = windturbine2d_farfield_loop(opts)
%WINDTURBINE2D_FARFIELD_LOOP Build the polygonal far-field circle.
%
% The segment count is chosen from the polygon chord length, not from arc
% length, so the far-field boundary does not over-resolve the background mesh.

opts = windturbine2d_background_mesh_options(opts);
Rfar = opts.Rfar;
h = opts.farfieldBoundarySpacing;

if ~(isnumeric(Rfar) && isscalar(Rfar) && isfinite(Rfar) && Rfar > 0)
    error('Rfar must be a positive finite scalar.');
end
if ~(isnumeric(h) && isscalar(h) && isfinite(h) && h > 0)
    error('farfieldBoundarySpacing must be a positive finite scalar.');
end

minPts = opts.minFarfieldPoints;
maxPts = opts.maxFarfieldPoints;
if ~(isnumeric(minPts) && isscalar(minPts) && isfinite(minPts) && minPts >= 8)
    error('minFarfieldPoints must be a finite scalar >= 8.');
end
if ~(isnumeric(maxPts) && isscalar(maxPts) && isfinite(maxPts) && maxPts >= minPts)
    error('maxFarfieldPoints must be a finite scalar >= minFarfieldPoints.');
end
minPts = ceil(minPts);
maxPts = floor(maxPts);

ratio = min(1, h/(2*Rfar));
if ratio >= 1
    requestedPts = minPts;
else
    requestedPts = ceil(pi/asin(ratio));
end
Nfar = max(minPts, requestedPts);
Nfar = min(maxPts, Nfar);

theta = linspace(0, 2*pi, Nfar + 1)';
theta(end) = [];
farLoop = Rfar * [cos(theta), sin(theta)];

edges = farLoop - farLoop([2:end 1], :);
edgeLengths = sqrt(sum(edges.^2, 2));

info.radius = Rfar;
info.requestedSpacing = h;
info.requestedPointCount = requestedPts;
info.pointCount = Nfar;
info.minPointCount = minPts;
info.maxPointCount = maxPts;
info.minEdgeLength = min(edgeLengths);
info.maxEdgeLength = max(edgeLengths);
info.meanEdgeLength = mean(edgeLengths);
info.meanToRequestedRatio = info.meanEdgeLength / h;

if requestedPts > maxPts && info.maxEdgeLength > 1.2*h
    warning(['maxFarfieldPoints capped the far-field circle at %d points; ' ...
        'max chord %.6g exceeds requested spacing %.6g.'], Nfar, ...
        info.maxEdgeLength, h);
end
if info.meanToRequestedRatio < 0.5
    warning(['Far-field boundary mean chord %.6g is much smaller than ' ...
        'requested spacing %.6g. Consider lowering minFarfieldPoints.'], ...
        info.meanEdgeLength, h);
end

fprintf('Far-field boundary discretization\n');
fprintf('  Rfar                  = %.6g\n', info.radius);
fprintf('  requested spacing     = %.6g\n', info.requestedSpacing);
fprintf('  selected points       = %d\n', info.pointCount);
fprintf('  edge length min/mean/max = %.6g / %.6g / %.6g\n', ...
    info.minEdgeLength, info.meanEdgeLength, info.maxEdgeLength);
fprintf('  mean edge / hfar      = %.6g\n', info.meanToRequestedRatio);
end
