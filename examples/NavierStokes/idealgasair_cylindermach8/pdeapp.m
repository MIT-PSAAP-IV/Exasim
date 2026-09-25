%function [sol,pde,mesh,setupReport,initialReport,finalReport,comparisonReport] = pdeapp(runSolve, runComparison)
%PDEAPP Material-database ideal-gas-air Mach-8 cylinder example.
%
% This example mirrors ../hypersoniccylinder_mach8 but obtains ideal-gas
% thermodynamics and Sutherland-law transport from
% apps/materialdatabases/idealgasAirlogdensity.dat.

runSolve = 1;
runComparison = 1;

caseDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(fileparts(caseDir)));
run(fullfile(repoRoot, 'frontends', 'Matlab', 'exasim_setup.m'));
validPrefix = fullfile(fileparts(repoRoot), 'exasim_install');
if exist(fullfile(validPrefix, 'lib', 'cmake', 'Exasim', 'ExasimConfig.cmake'), 'file') == 2
    setenv('EXASIM_PREFIX', validPrefix);
end
addpath(caseDir, '-begin');

dbFile = fullfile(repoRoot, 'apps', 'materialdatabases', 'idealgasAirlogdensity.dat');
db = local_read_database(dbFile);

% Match the current analytic ideal-gas reference case.
gam = 1.4;
gam1 = gam - 1.0;
R = 287.05;
Re = 1.835e5;
Pr = 0.71;
Minf = 8.03;
Tref = 265.0;
Twall = 300.0;
LRef = 1.0;
Tinf = 1/(gam*gam1*Minf^2);
pinf = 1/(gam*Minf^2);
rinf = 1.0;
ruinf = 1.0;
rvinf = 0.0;
rEinf = 0.5 + pinf/gam1;

uRef = Minf*sqrt(gam*R*Tref);
eRef = uRef^2;
eInfPhys = eRef*(rEinf/rinf - 0.5*(ruinf^2+rvinf^2));
% The database viscosity is dimensional and tabulated/interpolated.  Use the
% database freestream viscosity directly in the Reynolds-number embedding
% rather than applying a transport multiplier.
propsInfAtUnitDensity = local_interp_database(db, 0.0, eInfPhys);
muInfPhys = propsInfAtUnitDensity(3);
rhoRef = Re*muInfPhys/(uRef*LRef);
pRef = rhoRef*uRef^2;
xiInf = log(rhoRef*rinf);
propsInf = local_interp_database(db, xiInf, eInfPhys);
transportFactor = 1.0;
muInfAnalytic = local_sutherland_viscosity(Tref);

[pde,~] = initializeexasim();
pde.model = "ModelD";
pde.modelfile = "pdemodel";
pde.platform = "cpu";
pde.mpiprocs = 1;
pde.hybrid = 1;
pde.extendedW = 1;
pde.ncw = 9;
pde.porder = 4;
pde.pgauss = 2*pde.porder;
pde.tau = 1.0;
pde.GMRESrestart = 100;
pde.linearsolvertol = 1e-8;
pde.linearsolveriter = 100;
pde.RBdim = 0;
pde.ppdegree = 20;
pde.NLtol = 1e-6;
pde.NLiter = 30;
pde.matvectol = 1e-6;
pde.materialdatabase = dbFile;
pde.datapath = caseDir;
pde.builddir = fullfile(caseDir, '.exasim');
pde.buildpath = pde.builddir;
pde.physicsparam = [gam Re Pr Minf rinf ruinf rvinf rEinf Tinf Tref Twall ...
                    rhoRef uRef pRef eRef LRef transportFactor];

mesh = mkmesh_cyl(pde.porder);
% isothermal wall, supersonic outflow, supersonic inflow
mesh.boundarycondition = [3;6;5];

setupReport = struct();
setupReport.database = dbFile;
setupReport.databaseHeader = db.header;
setupReport.databaseXiRange = [db.xi(1), db.xi(end)];
setupReport.databaseRhoRange = exp(setupReport.databaseXiRange);
setupReport.databaseERange = [db.e(1), db.e(end)];
setupReport.gamma = gam;
setupReport.R = R;
setupReport.Re = Re;
setupReport.Pr = Pr;
setupReport.Minf = Minf;
setupReport.Tref = Tref;
setupReport.Twall = Twall;
setupReport.TinfNondim = Tinf;
setupReport.rhoRef = rhoRef;
setupReport.uRef = uRef;
setupReport.pRef = pRef;
setupReport.eRef = eRef;
setupReport.LRef = LRef;
setupReport.transportFactor = transportFactor;
setupReport.freestreamProperties = local_props_struct(propsInf);
setupReport.hdg = pde.hybrid == 1;

fprintf('\nDatabase ideal-gas-air Mach-8 cylinder setup\n');
fprintf('  material database: %s\n', dbFile);
fprintf('  database xi=[%.8g, %.8g], rho=[%.8g, %.8g] kg/m^3, e=[%.8g, %.8g] J/kg\n', ...
    setupReport.databaseXiRange, setupReport.databaseRhoRange, setupReport.databaseERange);
fprintf('  discretization: HDG=%d, porder=%d, pgauss=%d\n', setupReport.hdg, pde.porder, pde.pgauss);
fprintf('  reference: Tref=%.8g K, Twall=%.8g K, rhoRef=%.8g kg/m^3, uRef=%.8g m/s, pRef=%.8g Pa, eRef=%.8g J/kg\n', ...
    Tref, Twall, rhoRef, uRef, pRef, eRef);
fprintf('  Reynolds embedding: database mu(Tref)=%.8g Pa s, analytic Sutherland mu(Tref)=%.8g Pa s, rhoRef=Re*mu_db/(uRef*LRef), transportFactor=%.8g\n', ...
    muInfPhys, muInfAnalytic, transportFactor);
fprintf('  freestream database state: xi=%.8g, e=%.8g J/kg, p=%.8g Pa, T=%.8g K, mu=%.8g Pa s, a=%.8g m/s\n', ...
    xiInf, eInfPhys, propsInf(1), propsInf(2), propsInf(3), propsInf(5));
fprintf('  kappa handling: kappa_db=cp*mu, kappa=kappa_db/Pr with Pr=%.8g\n', Pr);

mesh.f = facenumbering(mesh.p,mesh.t,pde.elemtype,mesh.boundaryexpr,mesh.periodicexpr);
dist = meshdist3(mesh.f,mesh.dgnodes,mesh.perm,[1]);
mesh.dist = dist;
mesh.vdg = zeros(size(mesh.dgnodes,1),1,size(mesh.dgnodes,3));
mesh.vdg(:,1,:) = 0.06.*tanh(dist*30);

ui = [rinf ruinf rvinf rEinf];
UDG = initu(mesh,{ui(1),ui(2),ui(3),ui(4),0,0,0,0,0,0,0,0});
UDG(:,2,:) = UDG(:,2,:).*tanh(10*dist);
UDG(:,3,:) = UDG(:,3,:).*tanh(10*dist);
TnearWall = Tinf * (Twall/Tref-1) * exp(-10*dist) + Tinf;
UDG(:,4,:) = TnearWall + 0.5*(UDG(:,2,:).*UDG(:,2,:) + UDG(:,3,:).*UDG(:,3,:));
mesh.udg = UDG;

initialReport = local_solution_diagnostics(mesh.udg, db, setupReport);
local_print_diagnostics('initial', initialReport, setupReport);

if runSolve
    [sol,pde,mesh,master] = exasim(pde,mesh);
    % Continue the artificial-viscosity field in small steps.  The final
    % state matches the analytic ideal-gas reference setup, while the
    % intermediate states keep Newton trial material queries inside the
    % finite material-database domain.
    sol = local_stage_solve(pde,mesh,master,dist,sol,0.050,30,'Iter 2');
    sol = local_stage_solve(pde,mesh,master,dist,sol,0.040,30,'Iter 3');
    sol = local_stage_solve(pde,mesh,master,dist,sol,0.030,30,'Iter 4');
    sol = local_stage_solve(pde,mesh,master,dist,sol,0.025,30,'Iter 5');
    sol = local_stage_solve(pde,mesh,master,dist,sol,0.025,5,'Iter 6');
    mesh.udg = sol;
else
    sol = mesh.udg;
    master = [];
end

finalReport = local_solution_diagnostics(sol, db, setupReport);
local_print_diagnostics('final', finalReport, setupReport);

comparisonReport = struct();
if runComparison
    comparisonReport = local_compare_reference(repoRoot, mesh, sol, finalReport, setupReport);
end

save(fullfile(caseDir, 'idealgasair_cylindermach8_report.mat'), ...
    'setupReport', 'initialReport', 'finalReport', 'comparisonReport');

try
    figure(1); clf; scaplot(mesh, finalReport.MachField, [], 2, 1);
    axis equal; axis tight; colorbar; title('Database ideal-gas-air Mach number');
catch plotError
    warning('Plotting skipped: %s', plotError.message);
end
%end

function sol = local_stage_solve(pde, mesh, master, dist, sol, avAmplitude, avSlope, label)
fprintf('%s: artificial viscosity amplitude %.6g, tanh slope %.6g\n', label, avAmplitude, avSlope);
mesh.vdg(:,1,:) = avAmplitude.*tanh(dist*avSlope);
mesh.udg = sol;
[pde,mesh,master,dmd] = preprocessing(pde,mesh); %#ok<ASGLU>
runcode(pde, 1);
sol = fetchsolution(pde,master,dmd, pde.datapath + "/dataout" + model_strn(pde));
end

function report = local_solution_diagnostics(UDG, db, setup)
rho = UDG(:,1,:);
ux = UDG(:,2,:)./rho;
uy = UDG(:,3,:)./rho;
E = UDG(:,4,:)./rho;
e = E - 0.5*(ux.^2 + uy.^2);
rhoPhys = setup.rhoRef*rho;
ePhys = setup.eRef*e;
xi = log(rhoPhys);
props = local_interp_database(db, xi(:), ePhys(:));
p = reshape(props(:,1), size(rho));
T = reshape(props(:,2), size(rho));
a = reshape(props(:,5), size(rho));
velocity = sqrt(ux.^2 + uy.^2)*setup.uRef;
Mach = velocity./a;
report = struct();
report.rhoRange = [min(rhoPhys(:)), max(rhoPhys(:))];
report.xiRange = [min(xi(:)), max(xi(:))];
report.eRange = [min(ePhys(:)), max(ePhys(:))];
report.pRange = [min(p(:)), max(p(:))];
report.TRange = [min(T(:)), max(T(:))];
report.MachRange = [min(Mach(:)), max(Mach(:))];
report.velocityRange = [min(velocity(:)), max(velocity(:))];
report.tableMargin = min([report.xiRange(1)-db.xi(1), db.xi(end)-report.xiRange(2), ...
                          report.eRange(1)-db.e(1), db.e(end)-report.eRange(2)]);
report.outsideTable = report.tableMargin < 0;
report.rhoField = rhoPhys;
report.pressureField = p;
report.temperatureField = T;
report.velocityField = velocity;
report.MachField = Mach;
report.energyField = setup.eRef*E;
report.uxField = ux*setup.uRef;
report.uyField = uy*setup.uRef;
end

function local_print_diagnostics(label, report, setup)
fprintf('  %s state ranges:\n', label);
fprintf('    rho=[%.8g, %.8g] kg/m^3, xi=[%.8g, %.8g] within [%.8g, %.8g]\n', ...
    report.rhoRange, report.xiRange, setup.databaseXiRange);
fprintf('    e=[%.8g, %.8g] J/kg within [%.8g, %.8g]\n', ...
    report.eRange, setup.databaseERange);
fprintf('    p=[%.8g, %.8g] Pa, T=[%.8g, %.8g] K, M=[%.8g, %.8g]\n', ...
    report.pRange, report.TRange, report.MachRange);
fprintf('    table margin=%.8g, outsideTable=%d\n', report.tableMargin, report.outsideTable);
if report.outsideTable
    warning('%s state is outside the material database bounds.', label);
end
end

function comparison = local_compare_reference(repoRoot, mesh, sol, finalReport, setup)
comparison = struct();
refFile = fullfile(repoRoot, 'examples', 'NavierStokes', 'hypersoniccylinder_mach8', 'dataout', 'outudg_np0.bin');
if ~exist(refFile, 'file')
    warning('Reference analytic ideal-gas solution not found: %s', refFile);
    return;
end
fid = fopen(refFile, 'r');
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
raw = fread(fid, 'double');
npe = size(sol,1); ne = size(sol,3);
if numel(raw) > 3 && raw(1) == npe && raw(3) == ne
    nc = raw(2);
    ref = reshape(raw(4:end), [npe, nc, ne]);
else
    nc = numel(raw)/(npe*ne);
    ref = reshape(raw, [npe, nc, ne]);
end
if size(ref,1) ~= npe || size(ref,3) ~= ne
    warning('Reference solution size does not match database case.');
    return;
end
gam = setup.gamma;
gam1 = gam - 1.0;
rho = setup.rhoRef*ref(:,1,:);
ux = ref(:,2,:)./ref(:,1,:)*setup.uRef;
uy = ref(:,3,:)./ref(:,1,:)*setup.uRef;
E = ref(:,4,:)./ref(:,1,:)*setup.eRef;
pnd = gam1*(ref(:,4,:) - 0.5*ref(:,1,:).*((ref(:,2,:)./ref(:,1,:)).^2 + (ref(:,3,:)./ref(:,1,:)).^2));
p = setup.pRef*pnd;
Tnd = pnd./(gam1*ref(:,1,:));
T = setup.Tref/setup.TinfNondim*Tnd;
V = sqrt(ux.^2 + uy.^2);
M = V./sqrt(gam*setup.R*T);

comparison.relativeDensityL2 = local_rel_l2(finalReport.rhoField, rho);
comparison.relativePressureL2 = local_rel_l2(finalReport.pressureField, p);
comparison.relativeTemperatureL2 = local_rel_l2(finalReport.temperatureField, T);
comparison.relativeUxL2 = local_rel_l2(finalReport.uxField, ux);
comparison.relativeUyL2 = local_rel_l2(finalReport.uyField, uy);
comparison.relativeVelocityL2 = local_rel_l2(finalReport.velocityField, V);
comparison.relativeMachL2 = local_rel_l2(finalReport.MachField, M);
comparison.relativeEnergyL2 = local_rel_l2(finalReport.energyField, E);
comparison.maxDensityAbs = max(abs(finalReport.rhoField(:)-rho(:)));
comparison.maxPressureAbs = max(abs(finalReport.pressureField(:)-p(:)));
comparison.maxTemperatureAbs = max(abs(finalReport.temperatureField(:)-T(:)));
comparison.maxMachAbs = max(abs(finalReport.MachField(:)-M(:)));
comparison.referenceMachRange = [min(M(:)), max(M(:))];
comparison.databaseMachRange = finalReport.MachRange;
comparison.referencePressureRange = [min(p(:)), max(p(:))];
comparison.databasePressureRange = finalReport.pRange;
comparison.referenceTemperatureRange = [min(T(:)), max(T(:))];
comparison.databaseTemperatureRange = finalReport.TRange;

fprintf('  analytic/database ideal-gas comparison:\n');
fprintf('    rel L2 rho=%.6g, p=%.6g, T=%.6g, ux=%.6g, uy=%.6g, |V|=%.6g, M=%.6g, rhoE=%.6g\n', ...
    comparison.relativeDensityL2, comparison.relativePressureL2, comparison.relativeTemperatureL2, ...
    comparison.relativeUxL2, comparison.relativeUyL2, comparison.relativeVelocityL2, ...
    comparison.relativeMachL2, comparison.relativeEnergyL2);
fprintf('    max abs rho=%.6g, p=%.6g Pa, T=%.6g K, M=%.6g\n', ...
    comparison.maxDensityAbs, comparison.maxPressureAbs, comparison.maxTemperatureAbs, comparison.maxMachAbs);
end

function r = local_rel_l2(a,b)
r = norm(a(:)-b(:))/max(norm(b(:)), eps);
end

function db = local_read_database(filename)
fid = fopen(filename, 'r');
if fid < 0, error('Could not open material database: %s', filename); end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
header = sscanf(fgetl(fid), '%f').';
if numel(header) ~= 5 || header(1) ~= 2 || header(2) ~= 9
    error('Unexpected ideal-gas air database header in %s.', filename);
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

function s = local_props_struct(props)
names = {'p','T','mu','kappa_db','a','p_xi','p_e','T_xi','T_e'};
for i = 1:numel(names)
    s.(names{i}) = props(i);
end
end

function mu = local_sutherland_viscosity(T)
T0 = 273.15;
mu0 = 1.716e-5;
S = 110.4;
mu = mu0*(T/T0)^1.5*(T0+S)/(T+S);
end
