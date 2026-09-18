function [mesh, elem2cpu] = uniformrefinemesh(mesh, app, master, elem2cpu)
%UNIFORMREFINEMESH Uniformly refine a MATLAB frontend mesh.
%
% The element ordering is parent-major. If elem2cpu is supplied, every child
% inherits the MPI rank of its parent element.

if nargin < 4
    elem2cpu = [];
end

if ~isfield(app, 'uniformrefinementlevel') || app.uniformrefinementlevel == 0
    return;
end
if app.uniformrefinementlevel < 0 || app.uniformrefinementlevel ~= round(app.uniformrefinementlevel)
    error('uniformrefinementlevel must be a nonnegative integer.');
end
if isfield(mesh, 'uhat')
    error('uniformrefinementlevel > 0 cannot prolongate the face-based mesh.uhat field.');
end

if isfield(mesh, 'dgnodes') == 0
    try
        [f, ~, ~] = facenumbering(mesh.p, mesh.t, app.elemtype, mesh.boundaryexpr, mesh.periodicexpr);
        mesh.dgnodes = createdgnodes(mesh.p, mesh.t, f, mesh.curvedboundary, mesh.curvedboundaryexpr, app.porder);
    catch
        % Fall back to vertex-based refinement when the mesh does not provide
        % enough boundary information to build high-order geometry nodes.
    end
end

T = refinementtemplate(app.nd, app.elemtype, app.nve);
R = refinementprolongation(T, master.xpe, app.porder);

ne0 = size(mesh.t, 2);
np0 = size(mesh.p, 2);
checkfield(mesh, 'dgnodes', size(master.xpe, 1), ne0);
checkfield(mesh, 'udg', size(master.xpe, 1), ne0);
checkfield(mesh, 'vdg', size(master.xpe, 1), ne0);
checkfield(mesh, 'wdg', size(master.xpe, 1), ne0);
if ~isempty(elem2cpu) && numel(elem2cpu) ~= ne0
    error('uniformrefinement: elem2cpu must contain one entry per parent element.');
end

for level = 1:app.uniformrefinementlevel
    [mesh.p, mesh.t] = refineconnectivity(mesh.p, mesh.t, T, R, getfieldifpresent(mesh, 'dgnodes'));
    mesh = prolongmeshfield(mesh, 'dgnodes', R.P);
    mesh = prolongmeshfield(mesh, 'udg', R.P);
    mesh = prolongmeshfield(mesh, 'vdg', R.P);
    mesh = prolongmeshfield(mesh, 'wdg', R.P);
    if ~isempty(elem2cpu)
        elem2cpu = reshape(repmat(elem2cpu(:).', T.nchild, 1), [], 1);
    end
end

fprintf('uniformrefinementlevel = %d: ne %d -> %d, np %d -> %d\n', ...
    app.uniformrefinementlevel, ne0, size(mesh.t, 2), np0, size(mesh.p, 2));

end

function T = refinementtemplate(nd, elemtype, nve)

T.nd = nd;
T.elemtype = elemtype;
T.nve = nve;
T.xlat = [];
T.child = [];
T.nchild = 0;

    function addpt(varargin)
        x = zeros(1, nd);
        for d = 1:nd
            x(d) = varargin{d};
        end
        T.xlat = [T.xlat; x];
    end

    function addchild(v)
        T.child = [T.child v(:)];
        T.nchild = T.nchild + 1;
    end

if nd == 1 || elemtype == 1
    ny = 1 + 2*(nd >= 2);
    nz = 1 + 2*(nd == 3);
    for k = 0:(nz-1)
        for j = 0:(ny-1)
            for i = 0:2
                addpt(0.5*i, 0.5*j, 0.5*k);
            end
        end
    end
    lat = @(i,j,k) 1 + i + 3*(j + 3*k);
    if nd == 1
        addchild([lat(0,0,0); lat(1,0,0)]);
        addchild([lat(1,0,0); lat(2,0,0)]);
    elseif nd == 2
        for cy = 0:1
            for cx = 0:1
                addchild([lat(cx,cy,0); lat(cx+1,cy,0); lat(cx+1,cy+1,0); lat(cx,cy+1,0)]);
            end
        end
    else
        for cz = 0:1
            for cy = 0:1
                for cx = 0:1
                    addchild([lat(cx,cy,cz); lat(cx+1,cy,cz); lat(cx+1,cy+1,cz); lat(cx,cy+1,cz); ...
                        lat(cx,cy,cz+1); lat(cx+1,cy,cz+1); lat(cx+1,cy+1,cz+1); lat(cx,cy+1,cz+1)]);
                end
            end
        end
    end
elseif nd == 2
    addpt(0, 0, 0); addpt(1, 0, 0); addpt(0, 1, 0);
    addpt(0.5, 0, 0); addpt(0.5, 0.5, 0); addpt(0, 0.5, 0);
    addchild([1; 4; 6]);
    addchild([4; 2; 5]);
    addchild([6; 5; 3]);
    addchild([4; 5; 6]);
elseif nd == 3
    addpt(0, 0, 0); addpt(1, 0, 0); addpt(0, 1, 0); addpt(0, 0, 1);
    addpt(0.5, 0, 0); addpt(0, 0.5, 0); addpt(0, 0, 0.5);
    addpt(0.5, 0.5, 0); addpt(0.5, 0, 0.5); addpt(0, 0.5, 0.5);
    addchild([1; 5; 6; 7]);
    addchild([5; 2; 8; 9]);
    addchild([6; 8; 3; 10]);
    addchild([7; 9; 10; 4]);
    addchild([5; 6; 7; 9]);
    addchild([5; 6; 8; 9]);
    addchild([6; 7; 9; 10]);
    addchild([6; 8; 9; 10]);
end

if numel(T.child) ~= nve*T.nchild
    error('uniformrefinement: unsupported element (nd=%d, nve=%d, elemtype=%d).', nd, nve, elemtype);
end

phivl = linearbasis(T.xlat, nd, elemtype);
T.nlat = size(T.xlat, 1);
T.nsup = zeros(T.nlat, 1);
T.sup = zeros(nve, T.nlat);
for l = 1:T.nlat
    s = find(phivl(l, :) > 1e-12);
    T.nsup(l) = numel(s);
    T.sup(1:numel(s), l) = s(:);
end

if elemtype == 0 && nd >= 2
    for c = 1:T.nchild
        cv = T.child(:, c);
        E = T.xlat(cv(2:nd+1), :) - T.xlat(cv(1), :);
        detJ = det(E(:, 1:nd));
        if detJ < 0
            T.child([nd nd+1], c) = T.child([nd+1 nd], c);
        end
    end
end

end

function R = refinementprolongation(T, xpe, porder)

npe = size(xpe, 1);
phielem = linearbasis(xpe, T.nd, T.elemtype);
R.P = zeros(npe, npe, T.nchild);
R.Slat = mkshape(porder, xpe, T.xlat, T.elemtype);
R.Slat = R.Slat(:, :, 1);

for c = 1:T.nchild
    pts = zeros(npe, T.nd);
    for d = 1:T.nd
        for v = 1:T.nve
            pts(:, d) = pts(:, d) + phielem(:, v)*T.xlat(T.child(v, c), d);
        end
    end
    shap = mkshape(porder, xpe, pts, T.elemtype);
    R.P(:, :, c) = shap(:, :, 1).';
    rowsum = sum(R.P(:, :, c), 2);
    if max(abs(rowsum - 1.0)) > 1e-8
        error('uniformrefinement: prolongation is not a partition of unity.');
    end
end

end

function [pref, tref] = refineconnectivity(p, t, T, R, dgnodes)

[nd, np] = size(p);
ne = size(t, 2);
nve = T.nve;
pref = p;
tref = zeros(nve, ne*T.nchild);
index = containers.Map('KeyType', 'char', 'ValueType', 'double');
latid = zeros(T.nlat, 1);

for e = 1:ne
    te = t(:, e);
    for l = 1:T.nlat
        ns = T.nsup(l);
        if ns == 1
            latid(l) = te(T.sup(1, l));
            continue;
        end
        support = sort(te(T.sup(1:ns, l)));
        key = sprintf('%d_', support);
        if isKey(index, key)
            latid(l) = index(key);
        else
            id = np + 1;
            np = id;
            index(key) = id;
            x = zeros(nd, 1);
            if ~isempty(dgnodes)
                for d = 1:nd
                    x(d) = R.Slat(:, l).'*dgnodes(:, d, e);
                end
            else
                x = mean(p(:, support), 2);
            end
            pref(:, id) = x;
            latid(l) = id;
        end
    end
    for c = 1:T.nchild
        tref(:, (e-1)*T.nchild + c) = latid(T.child(:, c));
    end
end

end

function mesh = prolongmeshfield(mesh, name, P)

if ~isfield(mesh, name)
    return;
end
f = mesh.(name);
if isempty(f)
    return;
end
[npe, nc, ne] = size(f);
nchild = size(P, 3);
g = zeros(npe, nc, ne*nchild, 'like', f);
for e = 1:ne
    for c = 1:nchild
        g(:, :, (e-1)*nchild + c) = P(:, :, c)*f(:, :, e);
    end
end
mesh.(name) = g;

end

function checkfield(mesh, name, npe, ne)

if ~isfield(mesh, name) || isempty(mesh.(name))
    return;
end
if size(mesh.(name), 1) ~= npe || size(mesh.(name), 3) ~= ne
    error('uniformrefinement: mesh.%s must have size npe x ncomp x ne.', name);
end

end

function f = getfieldifpresent(s, name)

if isfield(s, name)
    f = s.(name);
else
    f = [];
end

end

function phi = linearbasis(pts, nd, elemtype)

if nd == 1
    xi = pts(:, 1);
    phi = [1-xi, xi];
elseif nd == 2 && elemtype == 0
    xi = pts(:, 1);
    eta = pts(:, 2);
    phi = [1-xi-eta, xi, eta];
elseif nd == 2 && elemtype == 1
    xi = pts(:, 1);
    eta = pts(:, 2);
    phi = [(1-xi).*(1-eta), xi.*(1-eta), xi.*eta, (1-xi).*eta];
elseif nd == 3 && elemtype == 0
    xi = pts(:, 1);
    eta = pts(:, 2);
    zeta = pts(:, 3);
    phi = [1-xi-eta-zeta, xi, eta, zeta];
elseif nd == 3 && elemtype == 1
    xi = pts(:, 1);
    eta = pts(:, 2);
    zeta = pts(:, 3);
    phi = [(1-xi).*(1-eta).*(1-zeta), xi.*(1-eta).*(1-zeta), ...
        xi.*eta.*(1-zeta), (1-xi).*eta.*(1-zeta), ...
        (1-xi).*(1-eta).*zeta, xi.*(1-eta).*zeta, ...
        xi.*eta.*zeta, (1-xi).*eta.*zeta];
else
    error('uniformrefinement: unsupported element.');
end

end
