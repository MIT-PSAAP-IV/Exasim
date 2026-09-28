% Sharp-B axisymmetric equilibrium-air flow with backend AV and mesh adaptation.

caseDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(fileparts(caseDir)));
run(fullfile(repoRoot, 'frontends', 'Matlab', 'exasim_setup.m'));
addpath(caseDir, '-begin');
% Reuse the Sharp-B mesh construction without duplicating it.
addpath(fullfile(repoRoot, 'examples', 'MeshAdaptivity', ...
    'sharpb2_idealgas'), '-end');

dbFile = fullfile(repoRoot, 'apps', 'materialdatabases', ...
    'equilibriumAir5logdensityexasim.dat');
db = local_read_material_database(dbFile);

[pde,~] = initializeexasim();
pde.model = "ModelD";
pde.modelfile = "pdemodel";
pde.platform = "cpu";
pde.mpiprocs = 8;
pde.porder = 2;
pde.pgauss = 2*pde.porder;
pde.hybrid = 1;
pde.debugmode = 0;
pde.nd = 2;
pde.extendedW = 1;
pde.ncw = 15;
pde.materialdatabase = dbFile;
pde.datapath = caseDir;
pde.builddir = fullfile(caseDir, '.exasim');
pde.buildpath = pde.builddir;
% Scalar fields: rho, p, T, Mach, AV, Y_N, Y_O, Y_NO, Y_N2, Y_O2.
% Vector field: dimensional velocity.
pde.saveParaview = 1;

% Preserve the physical targets from MeshAdaptivity/sharpb2_idealgas.
Minf = 21.38;
Re = 9.84e5;
TinfPhys = 260.6;
TwallPhys = 1400.0;
LRef = 1.0;
cChem = 0.0;
thetaOut = 0.0;

% Determine the equilibrium freestream density consistently with Re.
rhoInfPhys = 1.0e-3;
for iter = 1:12
    xiInf = log(rhoInfPhys);
    eInfPhys = local_energy_for_temperature(db, xiInf, TinfPhys);
    propsInf = local_interp_database(db, xiInf, eInfPhys);
    velocityInfPhys = Minf*propsInf(6);
    rhoNew = Re*propsInf(3)/(velocityInfPhys*LRef);
    if abs(rhoNew-rhoInfPhys) <= 1e-12*max(1.0, rhoInfPhys)
        rhoInfPhys = rhoNew;
        break;
    end
    rhoInfPhys = rhoNew;
end
xiInf = log(rhoInfPhys);
eInfPhys = local_energy_for_temperature(db, xiInf, TinfPhys);
propsInf = local_interp_database(db, xiInf, eInfPhys);
pInfPhys = propsInf(1);
aInfPhys = propsInf(6);
velocityInfPhys = Minf*aInfPhys;

rhoRef = rhoInfPhys;
uRef = velocityInfPhys;
pRef = rhoRef*uRef^2;
eRef = uRef^2;
transportFactor = 1.0;
pOutPhys = pInfPhys;

% Match the Sharp-B ideal-gas AV continuation and mesh-adaptation controls.
nm = 1e2;
pde.AV = 1;
pde.AVcontinuationIter = 9;
pde.AVcontinuationLogScale = 2;
pde.AVcoeffStart = 0.005;
pde.AVcoeffEnd = 0.000016;
pde.AVdistfunction = 1;
pde.distanceboundaryconditions = 3; % isothermal-wall flow BC tag
pde.AVsmoothingMethod = 1;
pde.AVHelmholtzCoeff = 0.001;
AVmaxdiv = 60.0;
AVdistcoeff = nm;

pde.meshadaptenabled = 1;
pde.meshadaptfield = 2; % physical pressure from visscalars
pde.meshadaptavcomponent = 1;
pde.meshadaptalpha = 0.5;
pde.meshadaptqmin = 0.2;
pde.meshadaptqmax = 0.8;
pde.meshadaptHelmholtzCoeff = 0.001;
pde.meshadaptforcescale = 0.25;
pde.meshadaptsmoothingpasses = 30;
% Geometric boundaries: axis, lower farfield, upper farfield, wall, outflow.
% Type 3 permits tangential motion; type 2 fixes both displacement components.
pde.meshadaptboundaryconditions = [3;3;3;2;3];

rhoInf = 1.0;
uzInf = 1.0;
urInf = 0.0;
eInf = eInfPhys/eRef;
rhoEInf = rhoInf*(eInf + 0.5*(uzInf^2 + urInf^2));
uFreestream = [rhoInf; rhoInf*uzInf; rhoInf*urInf; rhoEInf];
pde.externalparam = uFreestream;
pde.physicsparam = [rhoRef, uRef, pRef, eRef, LRef, transportFactor, ...
    cChem, TwallPhys, pOutPhys, thetaOut, AVmaxdiv, AVdistcoeff, ...
    pde.AVcoeffStart, pde.AVcoeffEnd];

pde.tau = 4.0;
pde.GMRESrestart = 250;
pde.GMRESortho = 1;
pde.linearsolvertol = 1e-6;
pde.linearsolveriter = 500;
pde.preconditioner = 1;
pde.RBdim = 0;
pde.ppdegree = 0;
pde.NLtol = 1e-6;
pde.NLiter = 10;
pde.matvectol = 1e-6;

mesh = mkmesh_sharpb2(pde.porder);
% Symmetry, inflow, inflow, isothermal no-slip wall, outflow.
mesh.boundarycondition = [5 1 1 3 2];
master = Master(pde);

dist = meshdist3(mesh.f, mesh.dgnodes, master.perm, 4);
mesh.vdg = zeros(size(mesh.dgnodes,1), 2, size(mesh.dgnodes,3));
mesh.vdg(:,1,:) = dist;
mesh.udg = local_initial_udg(mesh, dist, db, xiInf, TinfPhys, ...
    TwallPhys, eRef, uzInf, urInf, nm);

fprintf('\nEquilibrium-air Sharp-B setup\n');
fprintf('  rho_inf = %.10g kg/m^3, p_inf = %.10g Pa, T_inf = %.10g K\n', ...
    rhoInfPhys, pInfPhys, TinfPhys);
fprintf('  a_inf = %.10g m/s, U_inf = %.10g m/s, Mach = %.8g, Re = %.8g\n', ...
    aInfPhys, velocityInfPhys, Minf, Re);

pde.gencode = 1;
[sol,~,mesh,master,dmd] = exasim(pde,mesh);

xdg = getsolution(fullfile(caseDir, 'dataout', 'outxdg'), dmd, master.npe);
vdg = getsolution(fullfile(caseDir, 'dataout', 'outvdg'), dmd, master.npe);
wdg = getsolutions(fullfile(caseDir, 'dataout', 'outwdg'), dmd);
adaptedMesh = mesh;
adaptedMesh.dgnodes = xdg;

rhoPhys = rhoRef*sol(:,1,:);
speedPhys = uRef*sqrt(sol(:,2,:).^2 + sol(:,3,:).^2)./sol(:,1,:);
mach = speedPhys./wdg(:,6,:);

figure(1); clf; meshplot(adaptedMesh,1); axis equal; axis tight;
figure(2); clf; scaplot(adaptedMesh,mach,[0 Minf],2,1);
axis equal; axis tight; colorbar;
figure(3); clf; scaplot(adaptedMesh,vdg(:,2,:),[],2,2);
axis equal; axis tight; colorbar;
figure(4); clf; scaplot(adaptedMesh,rhoPhys,[],2,2);
axis equal; axis tight; colorbar;

function UDG = local_initial_udg(mesh, dist, db, xiInf, Tinf, Twall, ...
                                  eRef, uzInf, urInf, slope)
rho = ones(size(dist));
uz = uzInf*tanh(slope*dist);
ur = urInf*tanh(slope*dist);
T = Tinf + (Twall-Tinf)*exp(-slope*dist);
e = local_energy_for_temperature(db, xiInf, T)/eRef;
UDG = zeros(size(mesh.dgnodes,1), 4, size(mesh.dgnodes,3));
UDG(:,1,:) = rho;
UDG(:,2,:) = rho.*uz;
UDG(:,3,:) = rho.*ur;
UDG(:,4,:) = rho.*(e + 0.5*(uz.^2 + ur.^2));
end

function e = local_energy_for_temperature(db, xi, T)
tableT = zeros(numel(db.e),1);
for j = 1:numel(db.e)
    props = local_interp_database(db, xi, db.e(j));
    tableT(j) = props(2);
end
if min(T(:)) < min(tableT) || max(T(:)) > max(tableT)
    error('Temperature range [%.8g, %.8g] K is outside the material table.', ...
        min(T(:)), max(T(:)));
end
e = interp1(tableT, db.e, T(:), 'linear');
e = reshape(e, size(T));
end

function props = local_interp_database(db, xi, e)
sz = size(xi);
xi = xi(:);
e = e(:);
if isscalar(xi) && numel(e) > 1, xi = repmat(xi, size(e)); end
if isscalar(e) && numel(xi) > 1, e = repmat(e, size(xi)); end
props = zeros(numel(xi), size(db.props,3));
for k = 1:numel(xi)
    i = find(db.xi <= xi(k), 1, 'last');
    j = find(db.e <= e(k), 1, 'last');
    i = min(max(i,1), numel(db.xi)-1);
    j = min(max(j,1), numel(db.e)-1);
    tx = (xi(k)-db.xi(i))/(db.xi(i+1)-db.xi(i));
    te = (e(k)-db.e(j))/(db.e(j+1)-db.e(j));
    w00 = squeeze(db.props(i,j,:)).';
    w10 = squeeze(db.props(i+1,j,:)).';
    w01 = squeeze(db.props(i,j+1,:)).';
    w11 = squeeze(db.props(i+1,j+1,:)).';
    props(k,:) = (1-tx)*(1-te)*w00 + tx*(1-te)*w10 + ...
        (1-tx)*te*w01 + tx*te*w11;
end
if ~isscalar(e) || ~isscalar(xi)
    props = reshape(props, [sz size(db.props,3)]);
end
end

function db = local_read_material_database(filename)
fid = fopen(filename, 'r');
if fid < 0, error('Could not open material database: %s', filename); end
cleanup = onCleanup(@() fclose(fid));
header = fscanf(fid, '%d', 5).';
rows = fscanf(fid, '%f', [header(1)+header(2), Inf]).';
if size(rows,1) ~= header(3)*header(4)*header(5)
    error('Material database row count does not match its header.');
end
xi = unique(rows(:,1));
e = unique(rows(:,2));
props = zeros(numel(xi), numel(e), header(2));
for k = 1:size(rows,1)
    [~,i] = min(abs(xi-rows(k,1)));
    [~,j] = min(abs(e-rows(k,2)));
    props(i,j,:) = rows(k,3:end);
end
db = struct('header',header,'xi',xi,'e',e,'props',props);
end
