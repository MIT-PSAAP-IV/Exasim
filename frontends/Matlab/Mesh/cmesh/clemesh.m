function mesh = clemesh(p, t, xdg, porder)
%CLEMESH Connect six CLE/C-grid component meshes into one Exasim mesh.
%
%   mesh = clemesh(p, t, xdg, porder)
%
% Inputs are the cell-array outputs of clemeshmap:
%   p{i}   - linear vertices for component i, size np_i-by-2
%   t{i}   - element connectivity for component i, size ne_i-by-nve
%   xdg{i} - mapped high-order geometry, size npe-by-2-by-ne_i
%
% Components are connected in the C-grid order produced by clemeshparam6:
% lower wake, lower rear, lower front, upper front, upper rear, upper wake.
% connectmesh performs the linear vertex merge; xdg is preserved by
% concatenating component DG nodes because connectmesh does not reorder
% elements.

if nargin < 4
    error('clemesh requires p, t, xdg, and porder.');
end
if ~iscell(p) || ~iscell(t) || ~iscell(xdg)
    error('clemesh expects p, t, and xdg to be cell arrays.');
end
if numel(p) ~= 6 || numel(t) ~= 6 || numel(xdg) ~= 6
    error('clemesh expects exactly six component meshes.');
end
if isempty(porder) || any(porder(:) < 1) || any(abs(porder(:) - round(porder(:))) > 0)
    error('clemesh requires a positive integer porder.');
end

nodetype = 1;
elemtype = 1;
tol = 1e-6;
pall = p{1};
tall = t{1};
for i = 2:6
    [pall, tall] = connectmesh(pall, tall, p{i}, t{i}, tol);
end

if min(tall(:)) < 1 || max(tall(:)) > size(pall, 1)
    error('clemesh generated invalid element connectivity.');
end

% Build the standard Exasim mesh structure from the merged linear mesh, then
% replace the interpolated geometry with clemeshmap's high-order geometry.
bndexpr = {'true'};
mesh = mkmesh(pall, tall, porder, bndexpr, elemtype, nodetype);
mesh.dgnodes = cat(3, xdg{:});
mesh.xpe = mesh.plocal;
mesh.telem = mesh.tlocal;

local_check_final_mesh(mesh);

mesh.p = pall';
mesh.t = tall';
mesh.boundaryexpr = {@(p) sqrt((p(1,:)-.5).^2+p(2,:).^2)<3, ...
                     @(p) abs(p(1,:))< 1e6 + 1e-6};
mesh.boundarycondition = [1;2];
% The C-grid mapping already supplies curved high-order DG nodes. Do not
% re-project the Eppler boundary with an approximate implicit distance.
mesh.curvedboundary = [0 0];
mesh.curvedboundaryexpr = {@(p) 0*p(1,:), @(p) 0*p(1,:)};
mesh.periodicexpr = {};
mesh.f = local_facenumbering(mesh.p, mesh.t, mesh.elemtype, mesh.boundaryexpr);

end

function [pi, ti, xdgi, ne] = local_validate_component(pi, ti, xdgi, icomp, nve, npe, nd)
if ~isnumeric(pi) || size(pi, 2) ~= nd
    error('clemesh component %d has invalid p; expected np-by-2 numeric coordinates.', icomp);
end
if ~isnumeric(ti) || size(ti, 2) ~= nve
    error('clemesh component %d has invalid t; expected ne-by-%d connectivity.', icomp, nve);
end
if any(abs(ti(:) - round(ti(:))) > 0) || min(ti(:)) < 1 || max(ti(:)) > size(pi, 1)
    error('clemesh component %d has invalid one-based connectivity indices.', icomp);
end
if ~isnumeric(xdgi) || size(xdgi, 1) ~= npe || size(xdgi, 2) ~= nd
    error('clemesh component %d has invalid xdg; expected %d-by-2-by-ne.', icomp, npe);
end
ne = size(ti, 1);
if size(xdgi, 3) ~= ne
    error('clemesh component %d has inconsistent element counts between t and xdg.', icomp);
end
end

function [t, xdg] = local_fix_orientation(p, t, xdg, elemtype, plocal)
area = local_signed_area(p, t, elemtype);
flip = area < 0;
if ~any(flip)
    return;
end

if elemtype == 0
    t(flip, [1 2]) = t(flip, [2 1]);
    map = local_localnode_map(plocal, [1 - plocal(:,1) - plocal(:,2), plocal(:,2)]);
else
    t(flip, [1 2 3 4]) = t(flip, [4 3 2 1]);
    map = local_localnode_map(plocal, [plocal(:,1), 1 - plocal(:,2)]);
end
xdg(:,:,flip) = xdg(map,:,flip);
end

function map = local_localnode_map(plocal, target)
map = zeros(size(plocal, 1), 1);
for i = 1:size(target, 1)
    d = sum((plocal - target(i,:)).^2, 2);
    [dm, im] = min(d);
    if dm > 1e-24
        error('clemesh could not construct the local-node orientation map.');
    end
    map(i) = im;
end
end

function area = local_signed_area(p, t, elemtype)
if elemtype == 0
    x1 = p(t(:,1),1); y1 = p(t(:,1),2);
    x2 = p(t(:,2),1); y2 = p(t(:,2),2);
    x3 = p(t(:,3),1); y3 = p(t(:,3),2);
    area = 0.5*((x2-x1).*(y3-y1) - (y2-y1).*(x3-x1));
else
    x = reshape(p(t',1), size(t,2), [])';
    y = reshape(p(t',2), size(t,2), [])';
    area = 0.5*sum(x.*y(:,[2:end 1]) - y.*x(:,[2:end 1]), 2);
end
end

function local_check_final_mesh(mesh)
area = local_signed_area(mesh.p, mesh.t, mesh.elemtype);
if any(area <= 0)    
    error('clemesh generated %d inverted or zero-area elements.', nnz(area <= 0));
end
if size(mesh.dgnodes, 3) ~= size(mesh.t, 1)
    error('clemesh generated inconsistent mesh.t and mesh.dgnodes sizes.');
end
end

function f = local_facenumbering(p,t,elemtype,bndexpr)
%LOCAL_FACENUMBERING Path-independent boundary numbering for CLE meshes.
dim = size(p,1);
ne = size(t,2);
if dim ~= 2 || elemtype ~= 1
    error('clemesh local_facenumbering currently expects 2-D quadrilateral meshes.');
end

face = [1 2; 2 3; 3 4; 4 1]';
[nvf,nfe] = size(face);
t2fl = reshape(t(face,:),[nvf nfe*ne]);
pf = reshape(p(:,t2fl),[dim nvf nfe ne]);
f = zeros(nfe,ne);
[f2t, ~] = local_mkf2e(t, elemtype, dim);
boundaryFaces = find(f2t(3,:)==0);

for i = 1:numel(boundaryFaces)
    e = f2t(1,boundaryFaces(i));
    l = f2t(2,boundaryFaces(i));
    for k = 1:numel(bndexpr)
        if bndexpr{k}(pf(:,1,l,e)) && bndexpr{k}(pf(:,2,l,e))
            f(l,e) = k;
            break;
        end
    end
end
end

function [f2e, e2e] = local_mkf2e(t,elemtype,nd)
if nd ~= 2 || elemtype ~= 1
    error('clemesh local_mkf2e currently expects 2-D quadrilateral meshes.');
end

[~,ne] = size(t);
face = [1 2; 2 3; 3 4; 4 1]';
[nvf,nfe] = size(face);
tf = sort(reshape(t(face,:),[nvf nfe*ne]),1);

[tf, jx] = sortrows(tf');
tf = tf';
dx = sum((tf(:,2:end)-tf(:,1:end-1)).^2,1);
in1 = find(dx==0);
in2 = in1+1;
in0 = setdiff((1:length(jx))',unique([in1(:); in2(:)]));

nf = length(in0)+length(in1);
f2e = zeros(4,nf);
e2e = zeros(nfe,ne);

e1 = ceil(jx(in1)/nfe);
l1 = jx(in1) - (e1-1)*nfe;
e2 = ceil(jx(in2)/nfe);
l2 = jx(in2) - (e2-1)*nfe;
g = 1:length(in1);
f2e(1,g) = e1;
f2e(2,g) = l1;
f2e(3,g) = e2;
f2e(4,g) = l2;
for i = 1:length(e1)
    e2e(l1(i),e1(i)) = e2(i);
    e2e(l2(i),e2(i)) = e1(i);
end

e1 = ceil(jx(in0)/nfe);
l1 = jx(in0) - (e1-1)*nfe;
g = (length(in1)+1):nf;
f2e(1,g) = e1;
f2e(2,g) = l1;
end

function nmatch = local_count_matches(p1, p2, tol)
nmatch = 0;
for i = 1:size(p2, 1)
    d = sum((p1 - p2(i,:)).^2, 2);
    if min(d) < tol^2
        nmatch = nmatch + 1;
    end
end
end

function scale = local_mesh_scale(p)
q = vertcat(p{:});
scale = max(max(q, [], 1) - min(q, [], 1));
if scale == 0
    scale = 1;
end
end

function maxMismatch = local_interface_mismatch(mesh, elemComponent, permedge, tol)
maxMismatch = 0;
for iface = 1:size(mesh.f, 1)
    e1 = mesh.f(iface, end-1);
    e2 = mesh.f(iface, end);
    if e2 <= 0 || elemComponent(e1) == elemComponent(e2)
        continue;
    end
    lf1 = find(abs(mesh.t2f(e1,:)) == iface, 1);
    lf2 = find(abs(mesh.t2f(e2,:)) == iface, 1);
    if isempty(lf1) || isempty(lf2)
        error('clemesh could not identify local face numbers for interface face %d.', iface);
    end
    x1 = mesh.dgnodes(permedge(:,lf1),:,e1);
    x2 = mesh.dgnodes(permedge(:,lf2),:,e2);
    dforward = max(sqrt(sum((x1 - x2).^2, 2)));
    dreverse = max(sqrt(sum((x1 - flipud(x2)).^2, 2)));
    mismatch = min(dforward, dreverse);
    maxMismatch = max(maxMismatch, mismatch);
    if mismatch > 10*tol
        error('clemesh high-order interface mismatch %.3e exceeds tolerance %.3e on face %d.', ...
            mismatch, 10*tol, iface);
    end
end
end
