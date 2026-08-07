function [ugn, xgn, nlg] = surfacefieldalongnormal(mesh, master, udg, wid, delta, m, alpha, ind1)
%SURFACEFIELDALONGNORMAL Sample a solution along inward surface normals.
%
%   [ugn, xgn, nlg] = surfacefieldalongnormal(mesh, master, udg, wid, delta, m)
%   [ugn, xgn, nlg] = surfacefieldalongnormal(mesh, master, udg, wid, delta, m, alpha)
%   [ugn, xgn, nlg] = surfacefieldalongnormal(mesh, master, udg, wid, delta, m, alpha, ind1)
%
% udg has size npe x nc x ne. wid may contain one or more selected boundary
% markers. delta is the sampling thickness measured inward from the selected
% boundary, m is the number of points along each normal line, and alpha
% controls the logarithmic point distribution. ind1 optionally selects k
% indices in the first output dimension.
%
% nlg has size k x nd x nf. For m > 1, xgn has size k x nd x nf x m and
% ugn has size k x nc x nf x m, with
%
%   s = loginc(linspace(0,1,m), alpha);
%   xgn(:,:,:,i) = xg - delta*s(i)*nlg.
%
% For m == 1, this function returns surfacefield(mesh, master, udg, wid).

if nargin < 7 || isempty(alpha)
    alpha = 1.0;
end
if nargin < 8
    ind1 = min(2, master.ngf);
end

validateInputs(udg, delta, m, alpha);

if m == 1
    [ugn, xgn, nlg] = surfacefield(mesh, master, udg, wid);
    ind1 = validateIndices(ind1, size(xgn, 1));
    ugn = ugn(ind1,:,:);
    xgn = xgn(ind1,:,:);
    nlg = nlg(ind1,:,:);
    return;
end

[ug, xg, nlg] = surfacefield(mesh, master, udg, wid);

ind1 = validateIndices(ind1, size(xg, 1));
xg = xg(ind1,:,:);
nlg = nlg(ind1,:,:);

[ngf, nd, nf] = size(xg);
nc = size(ug, 2);

xgn = zeros(ngf, nd, nf, m, 'like', xg);
if alpha <= 1e-2
  s = reshape(linspace(0, 1, m), [1 1 1 m]);
else
  s = reshape(loginc(linspace(0, 1, m), alpha), [1 1 1 m]);
end

for i = 1:m
    xgn(:,:,:,i) = xg - (delta*s(i)).*nlg;
end

samplePoints = reshape(permute(xgn, [1 4 2 3]), [ngf*m, nd, nf]);
sampleValues = fieldatdgnodes(mesh, master, udg, samplePoints);

ugn = permute(reshape(sampleValues, [ngf, m, nc, nf]), [1 3 4 2]);

end

function validateInputs(udg, delta, m, alpha)
if ndims(udg) ~= 3
    error('udg must have size npe x nc x ne.');
end
if ~isscalar(delta) || ~isnumeric(delta) || ~isfinite(delta)
    error('delta must be a finite scalar.');
end
if delta < 0
    error('delta must be nonnegative.');
end
if ~isscalar(m) || ~isnumeric(m) || ~isfinite(m) || m < 1 || fix(m) ~= m
    error('m must be a positive integer.');
end
if ~isscalar(alpha) || ~isnumeric(alpha) || ~isfinite(alpha)
    error('alpha must be a finite scalar.');
end
end

function ind1 = validateIndices(ind1, ngf)
if isempty(ind1)
    ind1 = 1:ngf;
    return;
end
if ~isnumeric(ind1) || ~isvector(ind1) || any(~isfinite(ind1(:)))
    error('ind1 must be a vector of finite positive integer indices.');
end
if any(ind1(:) < 1) || any(fix(ind1(:)) ~= ind1(:)) || any(ind1(:) > ngf)
    error('ind1 entries must be integers between 1 and ngf.');
end
ind1 = ind1(:).';
end
