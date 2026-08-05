function [blt, dpt, mot, H, ve, v] = BLthicknesses(y, vort)
%BLTHICKNESSES Estimate boundary-layer thicknesses from vorticity.
%
%   [blt, dpt, mot, ye, ve] = BLthicknesses(y, vort)
%
% y is an n-point distance-from-wall array and vort is the signed vorticity
% at those points. A pseudo velocity is computed as
%
%   v(y) = integral_0^y vort(s) ds
%
% using the trapezoidal rule. The boundary-layer edge is the first point
% satisfying the supplied vorticity and vorticity-gradient criteria.

[y, vort] = validateInputs(y, vort);

v = cumtrapz(y, vort);
dvort = finiteDifference(y, vort);

edge = numel(y);
for i = 1:numel(y)
    if (abs(vort(i))*y(i) < 0.01*abs(v(i))) && ...
       (abs(dvort(i))*y(i)*y(i) < abs(v(i))) && ...
       (abs(v(i)) > 0.5)
        edge = i;
        break;
    end
end

ye = y(edge);
ve = v(edge);
blt = ye;

if ve == 0
    error('Detected edge pseudo velocity is zero; cannot compute thicknesses.');
end

eta = v(1:edge)/ve;
h = diff(y(1:edge));

d = 1 - eta;
dpt = sum(0.5*(d(1:end-1) + d(2:end)).*h);

momIntegrand = (1 - eta).*eta;
mot = sum(0.5*(momIntegrand(1:end-1) + momIntegrand(2:end)).*h);

H = dpt/mot;

end

function [y, vort] = validateInputs(y, vort)
if ~isnumeric(y) || ~isnumeric(vort)
    error('y and vort must be numeric arrays.');
end
if ~isvector(y) || ~isvector(vort)
    error('y and vort must be vectors.');
end
if numel(y) ~= numel(vort)
    error('y and vort must have the same number of entries.');
end
if numel(y) < 2
    error('At least two points are required.');
end
if any(~isfinite(y(:))) || any(~isfinite(vort(:)))
    error('y and vort must contain only finite values.');
end

y = y(:);
vort = vort(:);

if any(diff(y) <= 0)
    error('y must be strictly increasing.');
end
if y(1) ~= 0
    warning('BLthicknesses:NonzeroWallDistance', ...
            'The first y value is not zero; integration starts from y(1).');
end
end

function dvort = finiteDifference(y, vort)
n = numel(y);
dvort = zeros(n, 1);
dvort(1) = (vort(2) - vort(1))/(y(2) - y(1));
dvort(n) = (vort(n) - vort(n-1))/(y(n) - y(n-1));
for i = 2:n-1
    dvort(i) = (vort(i+1) - vort(i-1))/(y(i+1) - y(i-1));
end
end
