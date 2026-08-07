function [ug, xg, nlg] = surfacefield(mesh, master, udg, wid)
%SURFACEFIELD Interpolate solution fields on selected boundary faces.
%
%   [ug, xg, nlg] = surfacefield(mesh, master, udg, wid)
%
% udg has size npe x nc x ne. wid may contain one or more boundary markers.
% If mesh.boundarycondition exists, wid may also contain boundary-condition
% values; these are mapped to the corresponding boundary marker ids.
%
% ug has size ngf x nc x nf, while xg and nlg have size ngf x nd x nf. Here
% ngf is the number of face quadrature points, nd is the physical dimension,
% nc is the number of solution components, and nf is the total number of
% selected boundary faces.

validateInputs(mesh, master, udg, wid);

nd = size(mesh.dgnodes, 2);
nc = size(udg, 2);
perm = master.perm(:,:,1);
nfe = size(perm, 2);
ngf = getNumberOfFaceGaussPoints(master);
[shapft, dshapft] = getFaceShapeFunctions(master, nd);
markers = getBoundaryMarkers(mesh, wid);

faceElem = [];
faceLocal = [];
for elem = 1:size(mesh.f, 2)
    for face = 1:nfe
        marker = mesh.f(face, elem);
        if marker ~= 0 && ismember(abs(marker), markers)
            faceElem(end+1, 1) = elem; %#ok<AGROW>
            faceLocal(end+1, 1) = face; %#ok<AGROW>
        end
    end
end

nf = numel(faceElem);
if nf == 0
    error('No boundary faces found for wid = %s.', mat2str(wid));
end

ug = zeros(ngf, nc, nf, 'like', udg);
xg = zeros(ngf, nd, nf, 'like', mesh.dgnodes);
nlg = zeros(ngf, nd, nf, 'like', mesh.dgnodes);

for k = 1:nf
    elem = faceElem(k);
    face = faceLocal(k);
    faceNodes = perm(:, face);

    xdg = mesh.dgnodes(:, 1:nd, elem);
    xface = xdg(faceNodes, :);
    xgk = shapft*xface;

    uface = udg(faceNodes, :, elem);
    ug(:,:,k) = shapft*uface;

    [normal, jac] = faceNormal(xface, dshapft, ngf, nd);
    if any(jac <= eps(max(jac)))
        error('Degenerate boundary face detected for element %d, face %d.', elem, face);
    end

    xg(:,:,k) = xgk;
    nlg(:,:,k) = orientOutward(normal, xgk, xdg);
end

end

function validateInputs(mesh, master, udg, wid)
if ~isstruct(mesh)
    error('mesh must be a structure.');
end
if ~isstruct(master)
    error('master must be a structure.');
end
if ~isfield(mesh, 'f')
    error('mesh.f is required.');
end
if ~isfield(mesh, 'dgnodes')
    error('mesh.dgnodes is required.');
end
if ~isfield(master, 'perm')
    error('master.perm is required.');
end
if ndims(udg) ~= 3
    error('udg must have size npe x nc x ne.');
end
if isempty(wid)
    error('wid must contain at least one boundary marker.');
end

nd = size(mesh.dgnodes, 2);
if nd ~= 2 && nd ~= 3
    error('surfacefield supports only 2D and 3D meshes.');
end
if size(udg, 1) ~= size(mesh.dgnodes, 1) || size(udg, 3) ~= size(mesh.dgnodes, 3)
    error('udg and mesh.dgnodes must have matching nodal and element dimensions.');
end
if size(mesh.f, 2) ~= size(mesh.dgnodes, 3)
    error('mesh.f and mesh.dgnodes must have the same number of elements.');
end
if size(master.perm, 2) ~= size(mesh.f, 1)
    error('master.perm and mesh.f must have the same number of faces per element.');
end
end

function ngf = getNumberOfFaceGaussPoints(master)
if isfield(master, 'ngf')
    ngf = master.ngf;
elseif isfield(master, 'gwf')
    ngf = numel(master.gwf);
elseif isfield(master, 'gwfc')
    ngf = numel(master.gwfc);
elseif isfield(master, 'gpf')
    ngf = size(master.gpf, 1);
elseif isfield(master, 'gpfc')
    ngf = size(master.gpfc, 1);
else
    error('Cannot determine the number of face quadrature points.');
end
end

function [shapft, dshapft] = getFaceShapeFunctions(master, nd)
if isfield(master, 'shapfgt')
    shap = master.shapfgt;
elseif isfield(master, 'shapft')
    shap = master.shapft;
elseif isfield(master, 'shapfc')
    shap = master.shapfc;
else
    error('master.shapfgt, master.shapft, or master.shapfc is required.');
end

npf = size(master.perm, 1);
if size(shap, 2) == npf
    shapft = squeeze(shap(:,:,1));
    dshap = shap(:,:,2:nd);
else
    shapft = squeeze(shap(:,:,1))';
    dshap = zeros(size(shap, 2), size(shap, 1), nd-1);
    for d = 2:nd
        dshap(:,:,d-1) = squeeze(shap(:,:,d))';
    end
end
dshapft = reshape(permute(dshap, [1 3 2]), [size(shapft, 1)*(nd-1) npf]);
end

function markers = getBoundaryMarkers(mesh, wid)
markers = abs(wid(:)');
if isfield(mesh, 'boundarycondition') && ~isempty(mesh.boundarycondition)
    bc = mesh.boundarycondition(:)';
    for i = 1:numel(wid)
        markers = [markers find(bc == wid(i))]; %#ok<AGROW>
    end
end
markers = unique(markers);
end

function [normal, jac] = faceNormal(xface, dshapft, ngf, nd)
dpg = dshapft*xface;
dpg = permute(reshape(dpg, [ngf nd-1 nd]), [1 3 2]);

if nd == 2
    tangent = dpg(:,:,1);
    jac = sqrt(sum(tangent.^2, 2));
    normal = [tangent(:,2), -tangent(:,1)];
else
    t1 = dpg(:,:,1);
    t2 = dpg(:,:,2);
    normal = cross(t1, t2, 2);
    jac = sqrt(sum(normal.^2, 2));
end
normal = normal./jac;
end

function normal = orientOutward(normal, xg, xdg)
elemCenter = mean(xdg, 1);
faceCenter = mean(xg, 1);
if dot(mean(normal, 1), faceCenter - elemCenter) < 0
    normal = -normal;
end
end
