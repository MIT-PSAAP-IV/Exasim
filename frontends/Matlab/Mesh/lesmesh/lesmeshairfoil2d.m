function [p,t] = lesmeshairfoil2d(xf, yf, dlay, dwall, nx, ny, xref, yref)

%LESMESHAIRFOIL2D LES/DNS-style 2D airfoil mesh with cusped trailing-edge support.
%
%   [p,t] = lesmeshairfoil2d(xf, yf, dlay, dwall, nx, ny, xref, yref)
%
% This function follows the lesmesh2d interface and mesh-generation pipeline,
% but adds robust handling for closed airfoil coordinate lists whose first and
% last points coincide at a zero-thickness trailing edge.
%
% lesmesh2d maps the two ends of a rectangular computational grid to the two
% trailing-edge points of the airfoil curve. If the airfoil is supplied as a
% closed cusped curve, those two points are identical and the trailing-edge
% normal is not uniquely defined. This routine keeps the true cusped wall
% geometry and uses one-sided trailing-edge tangents to offset the first and
% last off-wall grid lines.
%
% No airfoil coordinates are dropped. The two logical wall nodes at the cusp
% are intentionally kept distinct even though they have the same physical
% coordinate; merging them would connect the two sides through a folded cap
% element. Non-cusped inputs are delegated directly to lesmesh2d.

if nargin ~= 8
    error('lesmeshairfoil2d expects 8 inputs: xf, yf, dlay, dwall, nx, ny, xref, yref.');
end

[xf, yf] = local_column_vectors(xf, yf);
[zeroTE, teInfo] = local_detect_zero_trailing_edge(xf, yf);

if zeroTE
    fprintf('lesmeshairfoil2d: zero-thickness trailing edge detected.\n');
    fprintf('lesmeshairfoil2d: TE gap = %.6e, tolerance = %.6e.\n', ...
        teInfo.gap, teInfo.tolerance);
end

if zeroTE
    [p,t] = local_lesmesh_grid(xf, yf, dlay, dwall, nx, ny, xref, yref);
    figure(3);clf;simpplot(p,t);axis on;
    p = local_map2cusped_airfoil(p, xf, yf);
else
    [p,t] = lesmesh2d(xf, yf, dlay, dwall, nx, ny, xref, yref);
end

if zeroTE
    %[p,t] = local_orient_elements(p,t);
    local_check_elements(p, t);
end

end

function [xf, yf] = local_column_vectors(xf, yf)

xf = xf(:);
yf = yf(:);

if length(xf) ~= length(yf)
    error('xf and yf must have the same number of entries.');
end

if length(xf) < 4
    error('At least four airfoil coordinate points are required.');
end

if any(~isfinite(xf)) || any(~isfinite(yf))
    error('Airfoil coordinates must be finite.');
end

end

function [zeroTE, info] = local_detect_zero_trailing_edge(xf, yf)

info = struct('scale', 0, 'gap', 0, 'tolerance', 0);
zeroTE = false;

coords = [xf yf];
bb = max(coords, [], 1) - min(coords, [], 1);
scale = max(norm(bb), eps);
info.scale = scale;

teGap = norm(coords(1,:) - coords(end,:));
tol = 1.0e-10*scale;
info.gap = teGap;
info.tolerance = tol;

zeroTE = teGap <= tol;

end

function [p,t] = local_lesmesh_grid(xf, yf, dlay, dwall, nx, ny, xref, yref)

if size(xref,2)~=2
    error('xref must have dimension Nx times 2');
end

% calculate the mesh ratio
c = 1 - dlay/dwall;
rat = fsolve(@(x) scalingfun(x,ny,c),[1;3]);
rat = rat(1);

% scaling distribution over the normal direction
yv = zeros(ny+1,1);
yv(2) = dwall;
for i = 1:(ny-1)
    yv(i+2) = yv(i+1) + dwall*(rat^i);
end

if abs(yv(end)-dlay)>1e-8
    error("Something wrong with the input parameters (dlay, dwall, ny)");
end

% Uniform distribution over foil
ns = length(xf);
tt = 0:ns-1;
spx = spline(tt,xf);
spy = spline(tt,yf);
ttp = distribute(nx,spx,spy,ns,50);
xv = zeros(nx+1,1);
xv(1:nx+1,1) = 2*ttp/(ns-1);

% make the computational grid from points
[p,t] = quadgrid(xv,yv);
figure(1);clf;simpplot(p,t);axis on;

% refine according to xref
for i = 1:size(xref,1)
    ind1 = (xv < xref(i,1));
    ind2 = (xv >= xref(i,1)) & (xv <= xref(i,2));
    ind3 = (xv > xref(i,2));
    x1 = xv(ind1);
    x2 = xv(ind2);
    x3 = xv(ind3);
    xw = 0.5*(x2(1:end-1)+x2(2:end));
    x2 = sort([x2; xw]);
    xv = unique([x1; x2; x3]);
end

% make the computational grid from points
[p,t] = quadgrid(xv,yv);

figure(2);clf;simpplot(p,t);axis on;

% refine according to yref
n = length(yref);
if n>0
    yref = sort(yref,'descend');
    for i = 1:n
        [p,t] = refineaty(p,t,yref(i));
    end
    [p,t] = fixmesh(p,t);
end

%[p,t] = removeelemement(p, t, ['y>' num2str(dlay/1.75)]);

end

function p = local_map2cusped_airfoil(pp, xf, yf)

% Map the computational strip to a closed cusped airfoil. The wall coordinate
% is the original closed airfoil. Near the duplicated trailing-edge point the
% normal direction blends from a one-sided trailing-edge tangent to the regular
% spline tangent. The smooth blend avoids the folded cap elements produced by a
% hard tangent switch at the cusp.
p = 0*pp;
nr = 0*pp;

coords = [xf(:), yf(:)];
nn = size(coords,1)-1;
tt = 0:nn;
spf = [spline(tt,coords(:,1)), spline(tt,coords(:,2))];
spfder = [fnder(spf(1)), fnder(spf(2))];

s = nn*pp(:,1)/2;
p(:,1:2) = [fnval(spf(1),s), fnval(spf(2),s)];
nr(:,1:2) = [fnval(spfder(1),s), fnval(spfder(2),s)];

scale = max(norm(max(coords,[],1)-min(coords,[],1)), eps);
lowerTangent = coords(2,:) - coords(1,:);
upperTangent = coords(end,:) - coords(end-1,:);
if norm(lowerTangent) <= 100*eps*scale || norm(upperTangent) <= 100*eps*scale
    error('Trailing-edge adjacent airfoil segments are too small to define one-sided tangents.');
end

% Blend over a small number of airfoil parameter intervals. In the first and
% last intervals the one-sided TE tangent dominates; away from the cusp the
% mapping smoothly recovers the original map2foil spline tangent.
blendWidth = min([8, max(3, 0.05*nn), 0.25*nn]);
blendWidth = 2;
left = s < blendWidth;
right = s > (nn - blendWidth);
if any(left)
    a = s(left)/blendWidth;
    w = local_smoothstep(a);
    w2 = repmat(w,1,2);
    nr(left,:) = (1-w2).*repmat(lowerTangent, sum(left), 1) + w2.*nr(left,:);
end
if any(right)
    a = (nn - s(right))/blendWidth;
    w = local_smoothstep(a);
    w2 = repmat(w,1,2);
    nr(right,:) = (1-w2).*repmat(upperTangent, sum(right), 1) + w2.*nr(right,:);
end

nr(:,1:2) = [-nr(:,2), nr(:,1)];
nrs = sqrt(nr(:,1).^2 + nr(:,2).^2);
bad = nrs <= 100*eps*scale;
if any(bad)
    error('Could not compute valid normals for the cusped airfoil mapping.');
end
nr(:,1:2) = nr(:,1:2)./nrs;
p(:,1:2) = p(:,1:2) + nr(:,1:2).*pp(:,2);

end

function w = local_smoothstep(a)

a = max(0, min(1, a(:)));
w = a.*a.*(3 - 2*a);

end

function [p, t] = local_orient_elements(p, t)

area2 = local_signed_area_measure(p, t);
tol = 100*eps*max(1, max(abs(p(:)))^2);
flip = area2 < -tol;

if any(flip)
    nv = size(t,2);
    t(flip,:) = t(flip,[1 nv:-1:2]);
    fprintf('lesmeshairfoil2d: reoriented %d negatively oriented elements.\n', sum(flip));
end

end

function local_check_elements(p, t)

if size(t,2) < 3
    return;
end

area2 = local_signed_area_measure(p, t);
tol = 100*eps*max(1, max(abs(p(:)))^2);
if any(abs(area2) <= tol)
    warning('lesmeshairfoil2d:DegenerateElements', ...
        'Generated mesh contains elements with near-zero signed area.');
end
if any(area2 < -tol) && any(area2 > tol)
    warning('lesmeshairfoil2d:MixedOrientation', ...
        'Generated mesh contains mixed element orientations.');
end

end

function area2 = local_signed_area_measure(p, t)

nv = size(t,2);
area2 = zeros(size(t,1),1);
for i = 1:nv
    j = i + 1;
    if j > nv
        j = 1;
    end
    xi = p(t(:,i),1);
    yi = p(t(:,i),2);
    xj = p(t(:,j),1);
    yj = p(t(:,j),2);
    area2 = area2 + xi.*yj - xj.*yi;
end

end
