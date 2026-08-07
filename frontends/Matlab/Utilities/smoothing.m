function usmooth = smoothing(u, k)
%SMOOTHING Smooth a one-dimensional array with repeated three-point passes.
%
%   usmooth = smoothing(u, k)
%
% u is a one-dimensional array and k is the number of smoothing passes.
% Each pass applies the filter
%
%   u_i <- 0.25*u_{i-1} + 0.5*u_i + 0.25*u_{i+1}
%
% to interior entries. Endpoints are kept fixed. The output has the same
% size and orientation as u.

if ~isvector(u)
    error('u must be a one-dimensional array.');
end
if ~isscalar(k) || ~isnumeric(k) || ~isfinite(k) || k < 0 || fix(k) ~= k
    error('k must be a nonnegative integer.');
end

usmooth = u;
if numel(u) <= 2 || k == 0
    return;
end

for iter = 1:k
    unew = usmooth;
    unew(2:end-1) = 0.25*usmooth(1:end-2) + ...
                    0.50*usmooth(2:end-1) + ...
                    0.25*usmooth(3:end);
    usmooth = unew;
end

end
