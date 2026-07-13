function [xf1, yf1] = airfoil(xf, yf, n, alpha)

%AIRFOIL Resample airfoil coordinates by approximate arc length.
%
%   [xf1, yf1] = airfoil(xf, yf, n)
%   [xf1, yf1] = airfoil(xf, yf, n, alpha)
%
% xf, yf are the original airfoil coordinates. xf1, yf1 are resampled to n
% points. If the input airfoil is closed, meaning the first and last points
% coincide to geometry-scale tolerance, the output is also closed and contains
% the same trailing-edge point as both the first and last entries.
%
% alpha controls leading-edge clustering. alpha = 0 gives uniform arc-length
% spacing. Larger positive alpha places more points near the leading edge than
% near the trailing edge on both airfoil sides.

if nargin < 4
    alpha = 0;
end

xf = xf(:);
yf = yf(:);

if length(xf) ~= length(yf)
    error('xf and yf must have the same number of entries.');
end

if length(xf) < 2
    error('At least two airfoil points are required.');
end

if n < 2 || n ~= round(n)
    error('n must be an integer greater than or equal to 2.');
end

if ~isfinite(alpha) || alpha < 0
    error('alpha must be a finite nonnegative scalar.');
end

if any(~isfinite(xf)) || any(~isfinite(yf))
    error('Airfoil coordinates must be finite.');
end

ds = sqrt(diff(xf).^2 + diff(yf).^2);
scale = max([max(xf)-min(xf), max(yf)-min(yf), 1]);
tol = 1.0e-12*scale;

keep = [true; ds > tol];
xf = xf(keep);
yf = yf(keep);

if length(xf) < 2
    error('Airfoil coordinates collapse to fewer than two distinct points.');
end

closed = hypot(xf(1)-xf(end), yf(1)-yf(end)) <= 1.0e-10*scale;

s = [0; cumsum(sqrt(diff(xf).^2 + diff(yf).^2))];
if s(end) <= tol
    error('Airfoil arc length is too small.');
end

si = local_airfoil_distribution(s, xf, n, alpha);
xf1 = interp1(s, xf, si, 'pchip');
yf1 = interp1(s, yf, si, 'pchip');

if closed
    xf1(1) = xf(1);
    yf1(1) = yf(1);
    xf1(end) = xf(1);
    yf1(end) = yf(1);
end

end

function si = local_airfoil_distribution(s, xf, n, alpha)

if alpha == 0
    si = linspace(0, s(end), n)';
    return;
end

[~, ile] = min(xf);
sLE = s(ile);

if sLE <= s(1) || sLE >= s(end)
    si = linspace(0, s(end), n)';
    return;
end

nleft = max(2, round(n*sLE/s(end)) + 1);
nright = n - nleft + 1;
if nright < 2
    nright = 2;
    nleft = n - nright + 1;
end

% First side: TE -> LE. logdec clusters toward the upper end of the interval,
% which is the leading edge.
s1 = logdec(linspace(s(1), sLE, nleft)', alpha);

% Second side: LE -> TE. loginc clusters toward the lower end of the interval,
% which is the leading edge.
s2 = loginc(linspace(sLE, s(end), nright)', alpha);

si = [s1; s2(2:end)];

end
