%function [sol,pde,mesh,setupReport,initialReport,finalReport,comparisonReport] = pdeapp(runSolve, runIdealComparison)
%PDEAPP Equilibrium five-species-air Mach-8 hypersonic cylinder example.
%
% This example mirrors examples/NavierStokes/hypersoniccylinder_mach8 while
% replacing the constant-gamma ideal-gas closure with the Exasim
% CNSequilibrium5air material-database model.  The required database is
%
%   apps/materialdatabases/equilibriumAir5logdensityexasim.dat
%
% whose independent variables are xi=log(rho_dimensional) and dimensional
% specific internal energy e [J/kg].

runSolve = 1;
runIdealComparison = 1;

caseDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(fileparts(caseDir)));
run(fullfile(repoRoot, 'frontends', 'Matlab', 'exasim_setup.m'));
validPrefix = fullfile(fileparts(repoRoot), 'exasim_install');
if exist(fullfile(validPrefix, 'lib', 'cmake', 'Exasim', 'ExasimConfig.cmake'), 'file') == 2
    setenv('EXASIM_PREFIX', validPrefix);
end
addpath(caseDir, '-begin');
addpath(fullfile(repoRoot, 'frontends', 'Matlab', 'Modeling', 'CNSequilibrium5air'), '-begin');


dbFile = fullfile(repoRoot, 'apps', 'materialdatabases', 'equilibriumAir5logdensityexasim.dat');
db = local_read_material_database(dbFile);

% Numerical settings copied from the ideal-gas Mach-8 cylinder case.
[pde,~] = initializeexasim();
pde.model = "ModelD";
pde.modelfile = "pdemodel";
pde.platform = "cpu";
pde.mpiprocs = 4;
pde.hybrid = 1;
pde.extendedW = 1;
pde.ncw = 15;
pde.porder = 2;
pde.pgauss = 2*pde.porder;
pde.tau = 1.0;
pde.GMRESrestart = 200;
pde.linearsolvertol = 1e-7;
pde.linearsolveriter = 200;
pde.RBdim = 0;
pde.ppdegree = 20;
pde.NLtol = 1e-6;
pde.NLiter = 30;
pde.matvectol = 1e-6;
pde.materialdatabase = dbFile;
pde.datapath = caseDir;
pde.builddir = fullfile(caseDir, '.exasim');
pde.buildpath = pde.builddir;

% pde.dt = [0.1 1 10];

mesh = mkmesh_square(51,32,pde.porder,1,1,1,1,1);
mesh.p(1,:) = logdec(mesh.p(1,:), 3);
mesh.dgnodes(:,1,:) = logdec(mesh.dgnodes(:,1,:), 3);
mesh = mkmesh_halfcircle(mesh, 1, 3, 4.0, pi/2, 3*pi/2);
mesh.porder = pde.porder;
mesh.boundaryexpr = {@(p) sqrt(p(1,:).^2+p(2,:).^2)<1+1e-6, ...
                     @(p) p(1,:)>-1e-7, @(p) abs(p(1,:))<20};
mesh.periodicexpr = {};
% CNSequilibrium5air boundary columns:
% 1 supersonic inflow, 2 supersonic outflow, 3 isothermal no-slip wall.
% The ideal-gas case uses [isothermal wall, supersonic outflow, supersonic inflow].
mesh.boundarycondition = [3;2;1];

% Physical reference state.  Match the current ideal-gas Mach-8 cylinder
% setup using Tref=265 K, Twall=300 K, and LRef=1 m.  The dimensional
% freestream density is derived from the requested Reynolds number,
% rho_inf = Re*mu_inf/(U_inf*LRef), instead of choosing an arbitrary density
% and compensating with a transport multiplier.
Minf = 8.03;
Re = 1.835e5;
TinfPhys = 265.0;  % K, from the updated ideal-gas reference case
TwallPhys = 300.0;     % K, same physical isothermal wall temperature
LRef = 1.0;            % cylinder-radius length scale in the nondimensional mesh
cChem = 0.0;           % use kappa = kappa_equi + kappa_chem
thetaOut = 0.0;

eInfPhys = -1.097328937016845e+05;
pInfPhys = 97.467348012325772;
rhoInfPhys = 0.001276095150944;
aInfPhys = 3.232136939449924e+02;
velocityInfPhys = Minf*aInfPhys;
xiInf = log(rhoInfPhys);

rhoRef = rhoInfPhys;
uRef = velocityInfPhys;
pRef = rhoRef*uRef^2;
eRef = uRef^2;
transportFactor = 1.0;
pOutPhys = pInfPhys;

pde.AV = 1;
pde.AVcontinuationIter = 10;
pde.AVcontinuationLogScale = 1.5;
pde.AVcoeffStart = 0.060;
pde.AVcoeffEnd = 0.008;
pde.AVdistfunction = 1;
pde.distanceboundaryconditions = [3]; % boundary-condition IDs stored in backend mesh.bf
pde.AVsmoothingMethod = 1;
pde.AVHelmholtzCoeff = 0.025;
AVmaxdiv = 2.0; AVdistcoeff = 30;

pde.meshadaptenabled = 1;
pde.meshadaptfield = 1; % nondimensional pressure from pdemodel.visscalars
pde.meshadaptalpha = 0.5;
pde.meshadaptHelmholtzCoeff = 5e-2;
pde.meshadaptforcescale = 0.15; % params(3) in pdeapp_frontend.m
pde.meshadaptsmoothingpasses = 30;
pde.meshadaptboundaryconditions = [2;3;3];

uInf = 1.0;
vInf = 0.0;
rhoInf = 1.0;
eInf = eInfPhys/eRef;
rhoEInf = rhoInf*(eInf + 0.5*(uInf*uInf + vInf*vInf));
uFreestream = [rhoInf; rhoInf*uInf; rhoInf*vInf; rhoEInf];
pde.externalparam = uFreestream;
pde.physicsparam = [rhoRef, uRef, pRef, eRef, LRef, transportFactor, ...
                    cChem, TwallPhys, pOutPhys, thetaOut, ...
                    AVmaxdiv AVdistcoeff pde.AVcoeffStart pde.AVcoeffEnd];

mesh.f = facenumbering(mesh.p,mesh.t,pde.elemtype,mesh.boundaryexpr,mesh.periodicexpr);
dist = meshdist3(mesh.f,mesh.dgnodes,mesh.perm,[1]);
mesh.dist = dist;
mesh.vdg = zeros(size(mesh.dgnodes,1),2,size(mesh.dgnodes,3));
mesh.vdg(:,1,:) = dist;
mesh.udg = local_initial_udg(mesh, dist, db, xiInf, TinfPhys, TwallPhys, eRef, uInf, vInf);

[sol,pde,mesh,master,dmd] = exasim(pde,mesh);

xdg = getsolution('dataout/outxdg',dmd, master.npe);
vdg = getsolution('dataout/outvdg',dmd, master.npe);
wdg = getsolutions('dataout/outwdg', dmd);
mesh1 = mesh; mesh1.dgnodes = xdg;

figure(1); clf; scaplot(mesh1, wdg(:,1,:)/pRef,[],2,2);
axis equal; axis tight; colorbar; colormap jet;

figure(2); clf; scaplot(mesh1, vdg(:,2,:),[],2,2);
axis equal; axis tight; colorbar;

figure(3); clf; meshplot(mesh1,1); axis equal; axis tight;

function sol = local_stage_solve(pde, mesh, master, dist, sol, avAmplitude, avSlope, label)
fprintf('%s: uniform artificial viscosity amplitude %.6g, tanh slope %.6g\n', label, avAmplitude, avSlope);
mesh.vdg(:,1,:) = avAmplitude.*tanh(dist*avSlope);
mesh.udg = sol;
[pde,mesh,master,dmd] = preprocessing(pde,mesh); %#ok<ASGLU>
runcode(pde, 1);
sol = fetchsolution(pde,master,dmd, pde.datapath + "/dataout" + model_strn(pde));
end

function props = local_interp_database(db, xi, e)
sz = size(xi);
xi = xi(:); e = e(:);
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
    props(k,:) = (1-tx)*(1-te)*w00 + tx*(1-te)*w10 + (1-tx)*te*w01 + tx*te*w11;
end
if numel(sz) > 2 || (numel(sz) == 2 && min(sz) > 1)
    props = reshape(props, [sz size(db.props,3)]);
end
end

function e = local_energy_for_temperature(db, xi, T)
Tvec = T(:);
Ts = zeros(numel(db.e),1);
for j = 1:numel(db.e)
    props = local_interp_database(db, xi, db.e(j));
    Ts(j) = props(2);
end
if min(Tvec) < min(Ts) || max(Tvec) > max(Ts)
    error('Requested T range [%.8g, %.8g] K is outside database T range [%.8g, %.8g] at xi=%.8g.', ...
        min(Tvec), max(Tvec), min(Ts), max(Ts), xi);
end
e = interp1(Ts, db.e, Tvec, 'linear');
e = reshape(e, size(T));
end

function UDG = local_initial_udg(mesh, dist, db, xiInf, Tinf, Twall, eRef, uInf, vInf)
rho = ones(size(dist));
ux = uInf*tanh(10*dist);
uy = vInf*tanh(10*dist);
TnearWall = Tinf + (Twall - Tinf)*exp(-10*dist);
ePhys = local_energy_for_temperature(db, xiInf, TnearWall);
e = ePhys/eRef;
UDG = zeros(size(mesh.dgnodes,1),4,size(mesh.dgnodes,3));
UDG(:,1,:) = rho;
UDG(:,2,:) = rho.*ux;
UDG(:,3,:) = rho.*uy;
UDG(:,4,:) = rho.*(e + 0.5*(ux.^2 + uy.^2));
end

function db = local_read_material_database(filename)
fid = fopen(filename, 'r');
if fid < 0, error('Could not open material database: %s', filename); end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
header = sscanf(fgetl(fid), '%f').';
if numel(header) ~= 5 || header(1) ~= 2 || header(2) < 15
    error('Unexpected equilibrium-air database header in %s.', filename);
end
rows = fscanf(fid, '%f', [header(1)+header(2), Inf]).';
if size(rows,1) ~= header(3)*header(4)*header(5)
    error('Material database row count does not match header.');
end
xi = unique(rows(:,1));
e = unique(rows(:,2));
props = zeros(numel(xi), numel(e), header(2));
for k = 1:size(rows,1)
    [~,i] = min(abs(xi - rows(k,1)));
    [~,j] = min(abs(e - rows(k,2)));
    props(i,j,:) = rows(k,3:end);
end
db = struct('header', header, 'xi', xi, 'e', e, 'props', props);
end
