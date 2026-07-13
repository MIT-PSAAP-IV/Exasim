function [Cp,Cf,x,Cp2d,Cf2d,x2d,Ch,Ch2d] = getsurfacedata(master,mesh,UDG,param,wid,elemAvg,deltaT,xyMidChord)
%GETSURFACEDATA Extract aerodynamic surface data from an Exasim solution.
%
%   [Cp,Cf,x,Cp2d,Cf2d,x2d,Ch,Ch2d] =
%       getsurfacedata(master,mesh,UDG,param,wid,elemAvg,deltaT,proftype)
%
% The function uses current Exasim mesh/master fields:
%   mesh.f, mesh.dgnodes, master.perm, master.shapfgt/master.shapft,
%   and master.gwf/master.gwfc.
%
% wid may contain one or more wall boundary markers. If mesh.boundarycondition
% is present, wid may also refer to boundary-condition values.

proftype = 1;
if nargin < 6 || isempty(elemAvg); elemAvg = 0; end
if nargin < 7 || isempty(deltaT); deltaT = 1; end
if nargin < 8 || isempty(xyMidChord); proftype = 0; end

validateInputs(master,mesh,UDG,param,wid,elemAvg,deltaT);

nd = getProblemDimension(master,mesh);
perm = master.perm;
nfe = size(perm,2);
ngf = getNumberOfFaceGaussPoints(master);
[shapft,dshapft,gwf] = getFaceOperators(master,nd);
wallMarkers = getWallMarkers(mesh,wid);

gamma = param(1);
Minf = param(4);
pinf = 1/(gamma*Minf^2);

xc = [];
yc = [];
zc = [];
cpData = [];
cfData = [];
chData = [];

for elem = 1:size(UDG,3)
    xdg = mesh.dgnodes(:,1:nd,elem);
    udg = UDG(:,:,elem);

    for face = 1:nfe
        if ~isWallFace(mesh.f(face,elem),wallMarkers)
            continue;
        end

        faceNodes = perm(:,face);
        xface = xdg(faceNodes,:);
        uface = udg(faceNodes,:);

        xg = shapft*xface;
        ug = shapft*uface;
        [nlg,jac] = surfaceGeometry(xface,dshapft,ngf,nd);
        nlg = orientNormalsOutward(nlg,xg,xdg);

        [pressure,tauWall,heatFlux] = wallQuantities(ug,nlg,param,nd);

        if elemAvg == 0
            xNew = xg(:,1);
            yNew = xg(:,2);
            if nd == 3
                zNew = xg(:,3);
            end
            cpNew = pressure;
            cfNew = tauWall;
            chNew = heatFlux;
        elseif elemAvg == 1
            wjac = abs(jac).*gwf(:);
            denom = sum(wjac);
            xNew = sum(wjac.*xg(:,1))/denom;
            yNew = sum(wjac.*xg(:,2))/denom;
            if nd == 3
                zNew = sum(wjac.*xg(:,3))/denom;
            end
            cpNew = sum(wjac.*pressure)/denom;
            cfNew = sum(wjac.*tauWall)/denom;
            chNew = sum(wjac.*heatFlux)/denom;
        else
            error('elemAvg must be 0 or 1.');
        end

        xc = [xc; xNew];
        yc = [yc; yNew];
        if nd == 3
            zc = [zc; zNew];
        end
        cpData = [cpData; cpNew];
        cfData = [cfData; cfNew];
        chData = [chData; chNew];
    end
end

if isempty(cpData)
    error('No wall faces found for wid = %s.', mat2str(wid));
end

if nd == 2
    xraw = [xc(:) yc(:)];
else
    xraw = [xc(:) yc(:) zc(:)];
end

[x,cpOrdered,cfOrdered,chOrdered] = orderSurfaceData(xraw,cpData,cfData,chData,proftype,xyMidChord);

Cp = -2*(cpOrdered - pinf);
Cf = -2*cfOrdered;
Ch = chOrdered/(deltaT*gamma);

if nd == 3
    [Cp2d,Cf2d,x2d,Ch2d] = spanwiseProjectSurfaceData(Cp,Cf,x,Ch,proftype);
else
    Cp2d = [];
    Cf2d = [];
    x2d = [];
    Ch2d = [];
end

end

function validateInputs(master,mesh,UDG,param,wid,elemAvg,deltaT)
if ~isstruct(master)
    error('master must be a structure.');
end
if ~isstruct(mesh)
    error('mesh must be a structure.');
end
requiredMeshFields = {'f','dgnodes'};
for i = 1:numel(requiredMeshFields)
    if ~isfield(mesh,requiredMeshFields{i})
        error('mesh.%s is required.', requiredMeshFields{i});
    end
end
if ~isfield(master,'perm')
    error('master.perm is required.');
end
if ndims(UDG) ~= 3
    error('UDG must have size [npe,nc,ne].');
end
if size(UDG,1) ~= size(mesh.dgnodes,1) || size(UDG,3) ~= size(mesh.dgnodes,3)
    error('UDG and mesh.dgnodes must have matching nodal and element dimensions.');
end
if numel(param) < 4
    error('param must contain at least [gamma, Re, Pr, Mach].');
end
if isempty(wid)
    error('wid must contain at least one wall boundary marker.');
end
if elemAvg ~= 0 && elemAvg ~= 1
    error('elemAvg must be 0 or 1.');
end
if deltaT == 0
    error('deltaT must be nonzero.');
end
end

function nd = getProblemDimension(master,mesh)
if isfield(master,'nd')
    nd = master.nd;
elseif isfield(master,'dim')
    nd = master.dim;
else
    nd = size(mesh.dgnodes,2);
end
if nd ~= 2 && nd ~= 3
    error('getsurfacedata supports only 2D and 3D problems.');
end
end

function ngf = getNumberOfFaceGaussPoints(master)
if isfield(master,'ngf')
    ngf = master.ngf;
elseif isfield(master,'gwf')
    ngf = numel(master.gwf);
elseif isfield(master,'gwfc')
    ngf = numel(master.gwfc);
else
    error('Cannot determine the number of face quadrature points.');
end
end

function [shapft,dshapft,gwf] = getFaceOperators(master,nd)
if isfield(master,'shapfgt')
    shap = master.shapfgt;
elseif isfield(master,'shapft')
    shap = master.shapft;
elseif isfield(master,'shapfc')
    shap = master.shapfc;
else
    error('master.shapfgt, master.shapft, or master.shapfc is required.');
end

npf = size(master.perm,1);
if size(shap,2) == npf
    shapft = squeeze(shap(:,:,1));
    dshap = shap(:,:,2:nd);
else
    shapft = squeeze(shap(:,:,1))';
    dshap = zeros(size(shap,2),size(shap,1),nd-1);
    for d = 2:nd
        dshap(:,:,d-1) = squeeze(shap(:,:,d))';
    end
end
dshapft = reshape(permute(dshap,[1 3 2]),[size(shapft,1)*(nd-1) npf]);

if isfield(master,'gwf')
    gwf = master.gwf;
elseif isfield(master,'gwfc')
    gwf = master.gwfc;
else
    error('master.gwf or master.gwfc is required.');
end
end

function markers = getWallMarkers(mesh,wid)
markers = abs(wid(:)');
if isfield(mesh,'boundarycondition') && ~isempty(mesh.boundarycondition)
    bc = mesh.boundarycondition(:)';
    for i = 1:numel(wid)
        markers = [markers find(bc == wid(i))];
    end
end
markers = unique(markers);
end

function tf = isWallFace(faceMarker,wallMarkers)
tf = faceMarker ~= 0 && ismember(abs(faceMarker),wallMarkers);
end

function [nlg,jac] = surfaceGeometry(xface,dshapft,ngf,nd)
dpg = dshapft*xface;
dpg = permute(reshape(dpg,[ngf nd-1 nd]),[1 3 2]);

if nd == 2
    tangent = dpg(:,:,1);
    jac = sqrt(sum(tangent.^2,2));
    nlg = [tangent(:,2), -tangent(:,1)];
else
    t1 = dpg(:,:,1);
    t2 = dpg(:,:,2);
    nlg = cross(t1,t2,2);
    jac = sqrt(sum(nlg.^2,2));
end

if any(jac <= eps)
    error('Degenerate boundary face detected while computing surface geometry.');
end
nlg = nlg./jac;
end

function nlg = orientNormalsOutward(nlg,xg,xdg)
elemCenter = mean(xdg,1);
faceCenter = mean(xg,1);
if dot(mean(nlg,1),faceCenter - elemCenter) < 0
    nlg = -nlg;
end
end

function [pressure,tauWall,heatFlux] = wallQuantities(ug,nlg,param,nd)
gamma = param(1);
nc = size(ug,2);
[ncu,stateDim,gradDim] = inferStateLayout(nc,nd);

rho = ug(:,1);
mom = ug(:,2:stateDim+1);
rhoE = ug(:,ncu);
vel = mom./rho;
pressure = (gamma-1)*(rhoE - 0.5*sum(mom.*vel,2));

tauWall = zeros(size(rho));
heatFlux = zeros(size(rho));
if nc == ncu
    return;
end

[stress,thermalFlux] = viscousWallFlux(ug,param,stateDim,gradDim,ncu);
nState = zeros(size(rho,1),stateDim);
nState(:,1:nd) = nlg;

traction = zeros(size(rho,1),stateDim);
for i = 1:stateDim
    for j = 1:stateDim
        traction(:,i) = traction(:,i) + stress(:,i,j).*nState(:,j);
    end
end

if stateDim == 2
    tangent = [nState(:,2), -nState(:,1)];
    tauWall = sum(traction.*tangent,2);
else
    normalTraction = sum(traction.*nState,2);
    tangentTraction = traction - normalTraction.*nState;
    tauWall = sqrt(sum(tangentTraction.^2,2));
end
heatFlux = sum(thermalFlux.*nState,2);
end

function [ncu,stateDim,gradDim] = inferStateLayout(nc,geomDim)
if nc == 4 || nc == 12
    ncu = 4;
    stateDim = 2;
elseif nc == 5 || nc == 20
    ncu = 5;
    stateDim = 3;
else
    ncu = geomDim + 2;
    if nc ~= ncu && mod(nc,ncu) ~= 0
        error('Cannot infer Euler/Navier-Stokes state layout from UDG with %d components.', nc);
    end
    stateDim = ncu - 2;
end

if nc == ncu
    gradDim = 0;
else
    if mod(nc,ncu) ~= 0
        error('UDG has %d components, not compatible with ncu = %d.', nc, ncu);
    end
    gradDim = nc/ncu - 1;
    if gradDim < stateDim
        error('UDG provides %d gradient directions, expected at least %d.', gradDim, stateDim);
    end
end

if geomDim > stateDim
    error('Geometry dimension %d cannot exceed state dimension %d.', geomDim, stateDim);
end
end

function [stress,thermalFlux] = viscousWallFlux(ug,param,nd,gradDim,ncu)
gamma = param(1);
Re = param(2);
Pr = param(3);
Minf = param(4);
if numel(param) >= 9
    Tref = param(9);
else
    Tref = 1;
end

rho = ug(:,1);
mom = ug(:,2:nd+1);
rhoE = ug(:,ncu);
vel = mom./rho;
pressure = (gamma-1)*(rhoE - 0.5*sum(mom.*vel,2));

drho = zeros(size(rho,1),nd);
dmom = zeros(size(rho,1),nd,nd);
drhoE = zeros(size(rho,1),nd);
for d = 1:nd
    ids = ncu + (d-1)*ncu + (1:ncu);
    qd = ug(:,ids);
    drho(:,d) = qd(:,1);
    dmom(:,:,d) = qd(:,2:nd+1);
    drhoE(:,d) = qd(:,ncu);
end

gradVel = zeros(size(rho,1),nd,nd);
gradP = zeros(size(rho,1),nd);
gradT = zeros(size(rho,1),nd);
kinetic = 0.5*sum(vel.^2,2);
for d = 1:nd
    for i = 1:nd
        gradVel(:,i,d) = (dmom(:,i,d) - drho(:,d).*vel(:,i))./rho;
    end
    qgrad = zeros(size(rho));
    for i = 1:nd
        qgrad = qgrad + vel(:,i).*gradVel(:,i,d);
    end
    gradP(:,d) = (gamma-1)*(drhoE(:,d) - drho(:,d).*kinetic - rho.*qgrad);
    gradT(:,d) = (gradP(:,d).*rho - pressure.*drho(:,d))./((gamma-1)*rho.^2);
end

T = pressure./((gamma-1)*rho);
Tinf = 1/(gamma*(gamma-1)*Minf^2);
Tphys = Tref/Tinf*T;
muRef = 1/Re;
mu = getViscosity(muRef,Tref,Tphys,1);
kappa = mu*gamma/Pr;

divVel = zeros(size(rho));
for i = 1:nd
    divVel = divVel + gradVel(:,i,i);
end

stress = zeros(size(rho,1),nd,nd);
for i = 1:nd
    for j = 1:nd
        stress(:,i,j) = mu.*(gradVel(:,i,j) + gradVel(:,j,i));
        if i == j
            stress(:,i,j) = stress(:,i,j) - (2/3)*mu.*divVel;
        end
    end
end

thermalFlux = kappa.*gradT;
end

function [xout,cpout,cfout,chout] = orderSurfaceData(x,cp,cf,ch,proftype,xyMidChord)
[lower] = lowerSurfaceMask(x(:,1),x(:,2),proftype,xyMidChord);

xl = x(lower,:);
xu = x(~lower,:);
cpl = cp(lower);
cpu = cp(~lower);
cfl = cf(lower);
cfu = cf(~lower);
chl = ch(lower);
chu = ch(~lower);

[~,il] = sortrows(xl,[1 min(3,size(x,2))]);
[~,iu] = sortrows(xu,[1 min(3,size(x,2))]);

xout = [flipud(xl(il,:)); xu(iu,:)];
cpout = [flipud(cpl(il)); cpu(iu)];
cfout = [flipud(cfl(il)); cfu(iu)];
chout = [flipud(chl(il)); chu(iu)];
end

function lower = lowerSurfaceMask(x,y,proftype, xyMidChord)
switch proftype
    case 0
        lower = y < 0;
    case 1
        lower = y < getMeanLine(x, xyMidChord);
    otherwise
        error('Unsupported proftype = %d.', proftype);
end
end

function [Cp2d,Cf2d,x2d,Ch2d] = spanwiseProjectSurfaceData(Cp,Cf,x,Ch,proftype)
snap = 1.0e-9;
xy = unique(round(x(:,1:2)/snap)*snap,'rows');

Cp2d = zeros(size(xy,1),1);
Cf2d = zeros(size(xy,1),1);
Ch2d = zeros(size(xy,1),1);
tol = 1.0e-5;
for i = 1:size(xy,1)
    ids = hypot(x(:,1)-xy(i,1),x(:,2)-xy(i,2)) < tol;
    Cp2d(i) = mean(Cp(ids));
    Cf2d(i) = mean(Cf(ids));
    Ch2d(i) = mean(Ch(ids));
end

[x2d,Cp2d,Cf2d,Ch2d] = orderSurfaceData(xy,Cp2d,Cf2d,Ch2d,proftype,xyMidChord);
end

function yMeanLine = getMeanLine(x, xyMidChord)
yMeanLine = interp1(xyMidChord(:,1),xyMidChord(:,2),x,'linear','extrap');
end
