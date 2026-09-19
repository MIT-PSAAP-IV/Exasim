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
pde.mpiprocs = 1;
pde.hybrid = 1;
pde.extendedW = 1;
pde.ncw = 15;
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

mesh = mkmesh_cyl(pde.porder);
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
TinfRequested = 265.0;  % K, from the updated ideal-gas reference case
TwallPhys = 300.0;     % K, same physical isothermal wall temperature
LRef = 1.0;            % cylinder-radius length scale in the nondimensional mesh
cChem = 0.0;           % use kappa = kappa_equi + kappa_chem
thetaOut = 1.0;

rhoInfPhys = 1.0e-3;
TinfPhys = TinfRequested;
freestreamAdjusted = false;
for iter = 1:8
    xiInf = log(rhoInfPhys);
    [TminAtRho, TmaxAtRho] = local_temperature_range_at_xi(db, xiInf);
    if TinfPhys <= TminAtRho || TinfPhys >= TmaxAtRho
        error(['Requested freestream Tinf=%.8g K is outside the database ' ...
               'temperature range [%.8g, %.8g] K at rho=%.8g kg/m^3.'], ...
              TinfPhys, TminAtRho, TmaxAtRho, rhoInfPhys);
    end
    eInfPhys = local_energy_for_temperature(db, xiInf, TinfPhys);
    propsInf = local_interp_database(db, xiInf, eInfPhys);
    muInfPhys = propsInf(3);
    aInfPhys = propsInf(6);
    velocityInfPhys = Minf*aInfPhys;
    rhoNew = Re*muInfPhys/(velocityInfPhys*LRef);
    if abs(rhoNew-rhoInfPhys) <= 1.0e-10*max(1.0, rhoInfPhys)
        rhoInfPhys = rhoNew;
        break;
    end
    rhoInfPhys = rhoNew;
end
xiInf = log(rhoInfPhys);
[TminAtRho, TmaxAtRho] = local_temperature_range_at_xi(db, xiInf);
eInfPhys = local_energy_for_temperature(db, xiInf, TinfPhys);
propsInf = local_interp_database(db, xiInf, eInfPhys);
pInfPhys = propsInf(1);
muInfPhys = propsInf(3);
aInfPhys = propsInf(6);
velocityInfPhys = Minf*aInfPhys;

rhoRef = rhoInfPhys;
uRef = velocityInfPhys;
pRef = rhoRef*uRef^2;
eRef = uRef^2;
transportFactor = 1.0;
pOutPhys = pInfPhys;

uInf = 1.0;
vInf = 0.0;
rhoInf = 1.0;
eInf = eInfPhys/eRef;
rhoEInf = rhoInf*(eInf + 0.5*(uInf*uInf + vInf*vInf));
uFreestream = [rhoInf; rhoInf*uInf; rhoInf*vInf; rhoEInf];
pde.externalparam = uFreestream;
pde.physicsparam = [rhoRef, uRef, pRef, eRef, LRef, transportFactor, ...
                    cChem, TwallPhys, pOutPhys, thetaOut];

setupReport = struct();
setupReport.database = dbFile;
setupReport.databaseHeader = db.header;
setupReport.databaseXiRange = [db.xi(1), db.xi(end)];
setupReport.databaseRhoRange = exp(setupReport.databaseXiRange);
setupReport.databaseERange = [db.e(1), db.e(end)];
setupReport.requestedTinf = TinfRequested;
setupReport.TinfPhys = TinfPhys;
setupReport.freestreamAdjusted = freestreamAdjusted;
setupReport.TRangeAtRhoInf = [TminAtRho, TmaxAtRho];
setupReport.freestreamProperties = local_props_struct(propsInf);
setupReport.Minf = Minf;
setupReport.Re = Re;
setupReport.hdg = pde.hybrid == 1;
setupReport.rhoRef = rhoRef;
setupReport.uRef = uRef;
setupReport.pRef = pRef;
setupReport.eRef = eRef;
setupReport.LRef = LRef;
setupReport.transportFactor = transportFactor;
setupReport.cChem = cChem;
setupReport.TwallPhys = TwallPhys;
setupReport.uFreestream = uFreestream;

fprintf('\nEquilibrium five-species-air Mach-8 cylinder setup\n');
fprintf('  material database: %s\n', dbFile);
fprintf('  database xi=[%.8g, %.8g], rho=[%.8g, %.8g] kg/m^3, e=[%.8g, %.8g] J/kg\n', ...
    setupReport.databaseXiRange, setupReport.databaseRhoRange, setupReport.databaseERange);
if freestreamAdjusted
    fprintf('  requested ideal-gas Tinf=%.8g K is outside the table at rho=%.8g kg/m^3; using Tinf=%.8g K\n', ...
        TinfRequested, rhoInfPhys, TinfPhys);
end
fprintf('  discretization: HDG=%d, porder=%d, pgauss=%d\n', setupReport.hdg, pde.porder, pde.pgauss);
fprintf('  freestream: rho=%.8g kg/m^3, T=%.8g K, p=%.8g Pa, e=%.8g J/kg, a=%.8g m/s, |V|=%.8g m/s, M=%.8g\n', ...
    rhoInfPhys, TinfPhys, pInfPhys, eInfPhys, aInfPhys, velocityInfPhys, Minf);
fprintf('  nondim: p*=%.8g, e*=%.8g, rhoE*=%.8g, transportFactor=%.8g, cChem=%.8g\n', ...
    pInfPhys/pRef, eInf, rhoEInf, transportFactor, cChem);

mesh.f = facenumbering(mesh.p,mesh.t,pde.elemtype,mesh.boundaryexpr,mesh.periodicexpr);
dist = meshdist3(mesh.f,mesh.dgnodes,mesh.perm,[1]);
mesh.dist = dist;
mesh.vdg = zeros(size(mesh.dgnodes,1),1,size(mesh.dgnodes,3));
mesh.vdg(:,1,:) = 0.06.*tanh(dist*30);

mesh.udg = local_initial_udg(mesh, dist, db, xiInf, TinfPhys, TwallPhys, eRef, uInf, vInf);
initialReport = local_solution_diagnostics(mesh.udg, db, rhoRef, eRef, uRef);
local_print_diagnostics('initial', initialReport, setupReport);

if runSolve
    [sol,pde,mesh,master] = exasim(pde,mesh);
    sol = local_stage_solve(pde,mesh,master,dist,sol,0.050,30,'Iter 2');
    sol = local_stage_solve(pde,mesh,master,dist,sol,0.040,30,'Iter 3');
    sol = local_stage_solve(pde,mesh,master,dist,sol,0.035,30,'Iter 4');
    sol = local_stage_solve(pde,mesh,master,dist,sol,0.030,30,'Iter 5');
    sol = local_stage_solve(pde,mesh,master,dist,sol,0.025,30,'Iter 6');
    sol = local_stage_solve(pde,mesh,master,dist,sol,0.025,5,'Iter 7');
    mesh.udg = sol;
end

finalReport = local_solution_diagnostics(sol, db, rhoRef, eRef, uRef);
local_print_diagnostics('final', finalReport, setupReport);

comparisonReport = struct();
if runIdealComparison
    comparisonReport = local_compare_ideal_case(repoRoot, mesh, sol, finalReport);
end

save(fullfile(caseDir, 'equilibrium5air_cylindermach8_report.mat'), ...
    'setupReport', 'initialReport', 'finalReport', 'comparisonReport');

try
    figure(1); clf;
    scaplot(mesh, finalReport.MachField, [0 Minf], 2, 1);
    axis equal; axis tight; colorbar; title('Equilibrium 5-air Mach number');
catch plotError
    warning('Plotting skipped: %s', plotError.message);
end

function sol = local_stage_solve(pde, mesh, master, dist, sol, avAmplitude, avSlope, label)
fprintf('%s: uniform artificial viscosity amplitude %.6g, tanh slope %.6g\n', label, avAmplitude, avSlope);
mesh.vdg(:,1,:) = avAmplitude.*tanh(dist*avSlope);
mesh.udg = sol;
[pde,mesh,master,dmd] = preprocessing(pde,mesh); %#ok<ASGLU>
runcode(pde, 1);
sol = fetchsolution(pde,master,dmd, pde.datapath + "/dataout" + model_strn(pde));
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

function report = local_solution_diagnostics(UDG, db, rhoRef, eRef, uRef)
rho = UDG(:,1,:);
ux = UDG(:,2,:)./rho;
uy = UDG(:,3,:)./rho;
e = UDG(:,4,:)./rho - 0.5*(ux.^2 + uy.^2);
rhoPhys = rhoRef*rho;
ePhys = eRef*e;
xi = log(rhoPhys);
props = local_interp_database(db, xi(:), ePhys(:));
p = reshape(props(:,1), size(rho));
T = reshape(props(:,2), size(rho));
a = reshape(props(:,6), size(rho));
velocityPhys = sqrt(ux.^2 + uy.^2)*uRef;
Mach = velocityPhys./a;
Ysum = sum(props(:,11:15),2);
report = struct();
report.xiRange = [min(xi(:)), max(xi(:))];
report.rhoRange = exp(report.xiRange);
report.eRange = [min(ePhys(:)), max(ePhys(:))];
report.pRange = [min(p(:)), max(p(:))];
report.TRange = [min(T(:)), max(T(:))];
report.velocityRange = [min(velocityPhys(:)), max(velocityPhys(:))];
report.MachRange = [min(Mach(:)), max(Mach(:))];
report.speciesMassFractionRanges = reshape([min(props(:,11:15),[],1); max(props(:,11:15),[],1)], [2,5]);
report.YsumErrorMax = max(abs(Ysum - 1));
report.tableMargin = min([report.xiRange(1)-db.xi(1), db.xi(end)-report.xiRange(2), ...
                          report.eRange(1)-db.e(1), db.e(end)-report.eRange(2)]);
report.outsideTable = report.tableMargin < 0;
report.rhoField = rhoPhys;
report.pressureField = p;
report.temperatureField = T;
report.velocityField = velocityPhys;
report.MachField = Mach;
report.YN2 = reshape(props(:,11), size(rho));
report.YO2 = reshape(props(:,12), size(rho));
report.YNO = reshape(props(:,13), size(rho));
report.YN = reshape(props(:,14), size(rho));
report.YO = reshape(props(:,15), size(rho));

end

function local_print_diagnostics(label, report, setupReport)
fprintf('  %s state ranges:\n', label);
fprintf('    xi=[%.8g, %.8g] within [%.8g, %.8g]\n', ...
    report.xiRange, setupReport.databaseXiRange);
fprintf('    e =[%.8g, %.8g] J/kg within [%.8g, %.8g]\n', ...
    report.eRange, setupReport.databaseERange);
fprintf('    p =[%.8g, %.8g] Pa, T=[%.8g, %.8g] K, |V|=[%.8g, %.8g] m/s, M=[%.8g, %.8g]\n', ...
    report.pRange, report.TRange, report.velocityRange, report.MachRange);
fprintf('    table margin=%.8g, outsideTable=%d, max |sumY-1|=%.3e\n', ...
    report.tableMargin, report.outsideTable, report.YsumErrorMax);
fprintf('    species max: Y_N2=%.6g, Y_O2=%.6g, Y_NO=%.6g, Y_N=%.6g, Y_O=%.6g\n', ...
    report.speciesMassFractionRanges(2,:));
if report.outsideTable
    warning('%s state is outside the material database bounds.', label);
end
end

function comparison = local_compare_ideal_case(repoRoot, mesh, sol, finalReport)
comparison = struct();
idealFile = fullfile(repoRoot, 'examples', 'NavierStokes', ...
    'hypersoniccylinder_mach8', 'dataout', 'outudg_np0.bin');
if ~exist(idealFile, 'file')
    warning('Ideal-gas data file not found: %s', idealFile);
    return;
end
fid = fopen(idealFile, 'r');
if fid < 0
    warning('Could not open ideal-gas data file: %s', idealFile);
    return;
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
raw = fread(fid, 'double');
if isempty(raw)
    warning('Could not read ideal-gas data file: %s', idealFile);
    return;
end
npe = size(sol,1); ne = size(sol,3);
if numel(raw) > 3 && raw(1) == npe && raw(3) == ne
    nc = raw(2);
    ideal = reshape(raw(4:end), [npe, nc, ne]);
else
    nc = numel(raw)/(npe*ne);
    ideal = reshape(raw, [npe, nc, ne]);
end
if size(ideal,1) ~= size(sol,1) || size(ideal,3) ~= size(sol,3)
    warning('Ideal-gas solution size does not match equilibrium-air mesh.');
    return;
end
gam = 1.4; R = 287.05; Minf = 8.03; Re = 1.835e5; Tref = 265.0; LRef = 1.0;
rhoIG = ideal(:,1,:);
uIG = ideal(:,2,:)./rhoIG;
vIG = ideal(:,3,:)./rhoIG;
pIG = (gam-1)*(ideal(:,4,:) - 0.5*rhoIG.*(uIG.^2+vIG.^2));
TIG = pIG./((gam-1)*rhoIG);
MIG = sqrt(uIG.^2+vIG.^2)./sqrt(gam*pIG./rhoIG);

TinfND = 1/(gam*(gam-1)*Minf^2);
uRefIdeal = Minf*sqrt(gam*R*Tref);
rhoRef = Re*local_sutherland_viscosity(Tref)/(uRefIdeal*LRef);
pRefIdeal = rhoRef*uRefIdeal^2;
rhoIGPhys = rhoRef*rhoIG;
pIGPhys = pRefIdeal*pIG;
TIGPhys = (Tref/TinfND)*TIG;
velocityIGPhys = sqrt(uIG.^2+vIG.^2)*uRefIdeal;

comparison.relativeDensityL2 = local_rel_l2(finalReport.rhoField, rhoIGPhys);
comparison.relativePressureL2 = local_rel_l2(finalReport.pressureField, pIGPhys);
comparison.relativeTemperatureL2 = local_rel_l2(finalReport.temperatureField, TIGPhys);
comparison.relativeVelocityL2 = local_rel_l2(finalReport.velocityField, velocityIGPhys);
comparison.relativeMachL2 = local_rel_l2(finalReport.MachField, MIG);
comparison.idealMachRange = [min(MIG(:)), max(MIG(:))];
comparison.equilibriumMachRange = finalReport.MachRange;
comparison.idealPressureRange = [min(pIGPhys(:)), max(pIGPhys(:))];
comparison.equilibriumPressureRange = finalReport.pRange;
comparison.idealTemperatureRange = [min(TIGPhys(:)), max(TIGPhys(:))];
comparison.equilibriumTemperatureRange = finalReport.TRange;
comparison.idealVelocityRange = [min(velocityIGPhys(:)), max(velocityIGPhys(:))];
comparison.equilibriumVelocityRange = finalReport.velocityRange;
comparison.idealShockStandOff = local_shock_standoff(mesh, MIG, Minf);
comparison.equilibriumShockStandOff = local_shock_standoff(mesh, finalReport.MachField, Minf);
fprintf('  ideal/equilibrium comparison on common nodes:\n');
fprintf('    relative density L2 difference = %.6g\n', comparison.relativeDensityL2);
fprintf('    relative pressure L2 difference = %.6g\n', comparison.relativePressureL2);
fprintf('    relative temperature L2 difference = %.6g\n', comparison.relativeTemperatureL2);
fprintf('    relative velocity L2 difference = %.6g\n', comparison.relativeVelocityL2);
fprintf('    relative Mach L2 difference = %.6g\n', comparison.relativeMachL2);
fprintf('    ideal M=[%.6g, %.6g], equilibrium M=[%.6g, %.6g]\n', ...
    comparison.idealMachRange, comparison.equilibriumMachRange);
fprintf('    estimated centerline shock stand-off: ideal=%.6g, equilibrium=%.6g\n', ...
    comparison.idealShockStandOff, comparison.equilibriumShockStandOff);
end

function mu = local_sutherland_viscosity(T)
T0 = 273.15;
mu0 = 1.716e-5;
S = 110.4;
mu = mu0*(T/T0).^1.5*(T0+S)./(T+S);
end

function r = local_rel_l2(a, b)
r = norm(a(:)-b(:))/max(norm(b(:)), eps);
end

function delta = local_shock_standoff(mesh, MachField, Minf)
% Estimate the upstream centerline shock stand-off from the first crossing
% of M < 0.95 Minf on nodes with |y| small.  This is only a diagnostic.
x = mesh.dgnodes(:,1,:);
y = mesh.dgnodes(:,2,:);
mask = abs(y(:)) < 0.03 & x(:) < -1.0;
xs = x(mask);
Ms = MachField(mask);
if numel(xs) < 2
    delta = NaN;
    return;
end
[xs, order] = sort(xs, 'descend');
Ms = Ms(order);
idx = find(Ms < 0.95*Minf, 1, 'first');
if isempty(idx)
    delta = NaN;
else
    delta = abs(xs(idx)) - 1.0;
end
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

function [Tmin, Tmax] = local_temperature_range_at_xi(db, xi)
Ts = zeros(numel(db.e),1);
for j = 1:numel(db.e)
    props = local_interp_database(db, xi, db.e(j));
    Ts(j) = props(2);
end
Tmin = min(Ts); Tmax = max(Ts);
end

function s = local_props_struct(props)
names = {'p','T','mu','kappa_equi','kappa_chem','a_equi','p_xi','p_e','T_xi','T_e', ...
         'Y_N2','Y_O2','Y_NO','Y_N','Y_O'};
for i = 1:numel(names)
    s.(names{i}) = props(i);
end
end
