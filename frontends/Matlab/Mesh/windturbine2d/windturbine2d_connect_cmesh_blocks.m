function [mesh, info] = windturbine2d_connect_cmesh_blocks(p, t, xdg, opts)
%WINDTURBINE2D_CONNECT_CMESH_BLOCKS Connect six C-mesh blocks into one mesh.
%
% This helper is intentionally local to the turbine workflow because the
% current production clemesh path fails for the Eppler test case.  It keeps
% p/t row-major to match mkmesh and fixes quadrilateral orientation before
% concatenating high-order DG geometry.

if numel(p) ~= 6 || numel(t) ~= 6 || numel(xdg) ~= 6
    error('Expected six C-mesh blocks.');
end

elemtype = 1;
tol = opts.mergeTolerance;
[plocal,~,~,~,~,~,~] = mkmasternodes(opts.porder, 2, elemtype, opts.nodetype);

areaBefore = [];
for i = 1:6
    [t{i}, xdg{i}] = local_fix_quad_orientation(p{i}, t{i}, xdg{i}, plocal);
    areaBefore = [areaBefore; local_signed_area(p{i}, t{i})]; %#ok<AGROW>
end

pall = p{1};
tall = t{1};
for i = 2:6
    [pall, tall] = connectmesh(pall, tall, p{i}, t{i}, tol);
end

bndexpr = {'true'};
mesh = mkmesh(pall, tall, opts.porder, bndexpr, elemtype, opts.nodetype);
mesh.dgnodes = cat(3, xdg{:});
mesh.xpe = mesh.plocal;
mesh.telem = mesh.tlocal;
mesh.boundarycondition = 1;
mesh.boundaryexpr = bndexpr;

areaAfter = local_signed_area(mesh.p, mesh.t);
info.vertices = size(mesh.p, 1);
info.elements = size(mesh.t, 1);
info.minAbsAreaBefore = min(abs(areaBefore));
info.minSignedAreaAfter = min(areaAfter);
info.minAbsAreaAfter = min(abs(areaAfter));
info.negativeAfter = nnz(areaAfter < 0);
info.nearZeroAfter = nnz(abs(areaAfter) <= 1e-14);

end

function [t, xdg] = local_fix_quad_orientation(p, t, xdg, plocal)
area = local_signed_area(p, t);
flip = area < 0;
if ~any(flip)
    return;
end

t(flip, [1 2 3 4]) = t(flip, [4 3 2 1]);
map = local_localnode_map(plocal, [plocal(:,1), 1 - plocal(:,2)]);
xdg(:,:,flip) = xdg(map,:,flip);
end

function map = local_localnode_map(plocal, target)
map = zeros(size(plocal, 1), 1);
for i = 1:size(target, 1)
    d = sum((plocal - target(i,:)).^2, 2);
    [dm, im] = min(d);
    if dm > 1e-24
        error('Could not construct local-node orientation map.');
    end
    map(i) = im;
end
end

function area = local_signed_area(p, t)
x = reshape(p(t',1), size(t,2), [])';
y = reshape(p(t',2), size(t,2), [])';
area = 0.5*sum(x.*y(:,[2:end 1]) - y.*x(:,[2:end 1]), 2);
end
