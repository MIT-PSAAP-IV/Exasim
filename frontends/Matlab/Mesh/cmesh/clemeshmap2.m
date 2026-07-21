function [p,t,xdg,wakeinfo] = clemeshmap2(xf, yf, p, t, lw, ll, porder, wakeopts)
%CLEMESHMAP2 Map a six-block C mesh and smoothly expand the downstream wake.
%
%   [p,t,xdg] = clemeshmap2(xf, yf, p, t, lw, ll, porder)
%   [p,t,xdg,wakeinfo] = clemeshmap2(..., wakeopts)
%
% This function preserves the clemeshmap interface and first expands the
% parametric wake blocks smoothly away from the trailing-edge interfaces.
% It then applies the baseline Trefftz airfoil mapping.  The parametric
% expansion is
%
%   y_new = S(s) * y,
%   S(s) = 1 + (R - 1) * (3*s^2 - 2*s^3),
%
% where s=0 on the trailing-edge block interface and s=1 at the downstream
% wake end.  The smoothstep factor has zero slope at both ends, so the wake
% expansion starts without a kink.  Since this deformation is applied before
% clemeshmap constructs the high-order geometry, vertex nodes and DG nodes
% remain consistent.
%
% wakeopts fields:
%   expansionRatio - far-wake parametric radial scaling R, default 2.0.
%   xStart         - expansion start x0, default max(xf).
%   xEnd           - full-expansion x1, default max mapped x-coordinate.
%   centerY        - wake centerline yc, default yf(1).
%   nstations      - diagnostic station count, default 9.
%   verbose        - print diagnostics, default true.

if nargin < 7
    porder = 1;
end
if nargin < 8
    wakeopts = struct();
end

opts = local_wake_options(wakeopts, xf, yf, p);
p = local_expand_parametric_wake(p, opts);

[p,t,xdg] = clemeshmap(xf, yf, p, t, lw, ll, porder);

opts = local_update_physical_options(opts, p);
wakeinfo = local_wake_diagnostics(p, t, opts);

if opts.verbose
    fprintf('clemeshmap2: wake expansion ratio = %.6g\n', opts.expansionRatio);
    fprintf('clemeshmap2: wake x-range = [%.16e, %.16e]\n', opts.xStart, opts.xEnd);
    fprintf('clemeshmap2: wake center y = %.16e\n', opts.centerY);
    fprintf('clemeshmap2: min signed area = %.16e\n', wakeinfo.minSignedArea);
    fprintf('clemeshmap2: min absolute area = %.16e\n', wakeinfo.minAbsArea);
    fprintf('clemeshmap2: negative signed-area elements = %d\n', wakeinfo.negativeSignedAreaElements);
    fprintf('clemeshmap2: near-zero area elements = %d\n', wakeinfo.nearZeroAreaElements);
    fprintf('clemeshmap2: wake width monotone = %d\n', wakeinfo.widthMonotone);
    for i = 1:numel(wakeinfo.stations)
        fprintf('clemeshmap2: wake width x = %.16e, width = %.16e\n', ...
            wakeinfo.stations(i), wakeinfo.width(i));
    end
end

end

function opts = local_wake_options(wakeopts, xf, yf, p)
if isnumeric(wakeopts)
    wakeopts = struct('expansionRatio', wakeopts);
end
if ~isstruct(wakeopts)
    error('clemeshmap2: wakeopts must be a struct or numeric expansion ratio.');
end

opts.expansionRatio = local_getfield(wakeopts, 'expansionRatio', 2.0);
opts.xStart = local_getfield(wakeopts, 'xStart', max(xf(:)));
opts.centerY = local_getfield(wakeopts, 'centerY', yf(1));
opts.nstations = local_getfield(wakeopts, 'nstations', 9);
opts.verbose = local_getfield(wakeopts, 'verbose', true);
opts.xEnd = local_getfield(wakeopts, 'xEnd', []);

if opts.expansionRatio < 1
    error('clemeshmap2: expansionRatio must be >= 1.');
end
if opts.nstations < 2
    error('clemeshmap2: nstations must be at least 2.');
end
end

function opts = local_update_physical_options(opts, p)
if isempty(opts.xEnd)
    allp = vertcat(p{:});
    opts.xEnd = max(allp(:,1));
end
if opts.xEnd <= opts.xStart
    error('clemeshmap2: xEnd must be greater than xStart.');
end
end

function value = local_getfield(s, name, defaultValue)
if isfield(s, name) && ~isempty(s.(name))
    value = s.(name);
else
    value = defaultValue;
end
end

function p = local_expand_parametric_wake(p, opts)
p{1} = local_expand_parametric_block(p{1}, true, opts.expansionRatio);
p{6} = local_expand_parametric_block(p{6}, false, opts.expansionRatio);
end

function q = local_expand_parametric_block(q, reversed, expansionRatio)
if reversed
    x0 = max(q(:,1));
    x1 = min(q(:,1));
    s = (x0 - q(:,1)) / (x0 - x1);
else
    x0 = min(q(:,1));
    x1 = max(q(:,1));
    s = (q(:,1) - x0) / (x1 - x0);
end
s = min(max(s, 0), 1);
%blend = s.^2 .* (3 - 2*s);
blend = s;
scale = 1 + (expansionRatio - 1) * blend;
q(:,2) = scale .* q(:,2);
end

function info = local_wake_diagnostics(p, t, opts)
area = [];
for i = 1:numel(p)
    area = [area; local_signed_area(p{i}, t{i})]; %#ok<AGROW>
end

[stations, width] = local_wake_width(p, opts);
tol = 1e-12 * max(1, max(abs(width)));

info.minSignedArea = min(area);
info.minAbsArea = min(abs(area));
info.negativeSignedAreaElements = nnz(area < 0);
info.nearZeroAreaElements = nnz(abs(area) <= 1e-14);
info.stations = stations;
info.width = width;
info.widthMonotone = all(isfinite(width)) && all(diff(width) >= -tol);
end

function area = local_signed_area(p, t)
x = reshape(p(t',1), size(t,2), [])';
y = reshape(p(t',2), size(t,2), [])';
area = 0.5*sum(x.*y(:,[2:end 1]) - y.*x(:,[2:end 1]), 2);
end

function [stations, width] = local_wake_width(p, opts)
stations = linspace(opts.xStart, opts.xEnd, opts.nstations);
[xu, yu] = local_boundary_envelope(p{6}, 1, opts.centerY);
[xl, yl] = local_boundary_envelope(p{1}, -1, opts.centerY);
yu = interp1(xu, yu, stations, 'linear', 'extrap');
yl = interp1(xl, yl, stations, 'linear', 'extrap');
width = yu - yl;
end

function [xo, yo] = local_boundary_envelope(p, side, centerY)
% side = 1 extracts the upper wake envelope; side = -1 extracts the lower.
if side > 0
    q = p(p(:,2) >= centerY, :);
else
    q = p(p(:,2) <= centerY, :);
end
if isempty(q)
    xo = NaN;
    yo = NaN;
    return;
end

xrounded = round(q(:,1) * 1e12) / 1e12;
[xo, ~, ic] = unique(xrounded);
if side > 0
    yo = accumarray(ic, q(:,2), [], @max);
else
    yo = accumarray(ic, q(:,2), [], @min);
end
end
