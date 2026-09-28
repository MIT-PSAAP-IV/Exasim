
caseDir  = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(fileparts(caseDir)));
run(fullfile(repoRoot,'NewExasim','Exasim', 'frontends', 'Matlab', 'exasim_setup.m'));

addpath('CNSequilibrium5air');
addpath(caseDir, '-begin');

[pde, mesh] = initializeexasim();
mesh = mkmesh_auplate2d4UR(2);

pde.model     = "ModelD";
pde.modelfile = "pdemodel";     % must resolve to the equilibrium pdemodel.m

pde.datapath = caseDir;
pde.builddir = fullfile(caseDir, '.exasim');
pde.buildpath = pde.builddir;

pde.platform  = "cpu";
pde.mpiprocs  = 8;
pde.porder    = 2;
pde.pgauss    = 2*pde.porder;
pde.hybrid    = 1;              % HDG
pde.debugmode = 0;
pde.nd        = 2;

% This part of the previous script was already correct: ncw=15 matches
% pdemodel.m's w vector, and extendedW=1 enables the materialstate/
% database-driven auxiliary-variable path.
pde.extendedW = 1;
pde.ncw = 15;

dbFile = fullfile(repoRoot,'NewExasim','Exasim','apps', 'materialdatabases', ...
    'equilibriumAir5logdensityexasim.dat');
pde.materialdatabase = dbFile;
db = local_read_material_database(dbFile);

%% ---- Freestream / reference state -------------------------------------
L_ref        = 1.0;
T_wall       = 296.0;
rho_phys_inf = 0.1081;
v_phys_inf   = 3444.0;
T_phys_inf   = 1020.0;

% For an equilibrium-chemistry gas, p is NOT independent once (rho,T) are
% fixed -- it comes out of the table. Keep the old ideal-gas number only
% as a sanity check against the table value.
p_ideal_check = 287*rho_phys_inf*T_phys_inf;

xiInf = log(rho_phys_inf);
[TminAtRho, TmaxAtRho] = local_temperature_range_at_xi(db, xiInf);
if T_phys_inf <= TminAtRho || T_phys_inf >= TmaxAtRho
    error(['Freestream Tinf=%.6g K is outside the database range ' ...
           '[%.6g, %.6g] K at rho=%.6g kg/m^3.'], ...
          T_phys_inf, TminAtRho, TmaxAtRho, rho_phys_inf);
end
e_phys_inf  = local_energy_for_temperature(db, xiInf, T_phys_inf);
propsInf    = local_interp_database(db, xiInf, e_phys_inf);
p_phys_inf  = propsInf(1);
mu_phys_inf = propsInf(3);
a_phys_inf  = propsInf(6);

fprintf('freestream: rho=%.6g kg/m^3, T=%.6g K, p(table)=%.6g Pa (ideal-gas check=%.6g Pa), a=%.6g m/s, M=%.4g\n', ...
    rho_phys_inf, T_phys_inf, p_phys_inf, p_ideal_check, a_phys_inf, v_phys_inf/a_phys_inf);

% Reference scaling matching pdemodel.m's mu(1:10) convention:
%   mu = [rho_ref u_ref p_ref e_ref L_ref transportFactor cChem T_wall p_out theta_out]
rho_ref = rho_phys_inf;
u_ref   = v_phys_inf;
p_ref   = rho_ref*u_ref^2;
e_ref   = u_ref^2;
transportFactor = 1.0;
cChem   = 1.0;              % kappa = kappa_equi + cChem*kappa_chem -- ON,
                             % since kappa_chem is the reactive/recombination
                             % contribution to wall heat transfer, which
                             % matters most right at a cold (296 K) wall.
p_out   = p_phys_inf;
theta_out = 1.0;


%---adaptive additions
pde.AV = 1;
pde.AVcontinuationIter = 15;
pde.AVcontinuationLogScale = 1.5;
pde.AVcoeffStart = 0.020;
pde.AVcoeffEnd = 5e-5;
pde.AVdistfunction = 1;
pde.distanceboundaryconditions = [3]; % boundary-condition IDs stored in backend mesh.bf
pde.AVsmoothingMethod = 1;
pde.AVHelmholtzCoeff = 0.025;
AVmaxdiv = 4.0; AVdistcoeff = 1e3;


pde.meshadaptenabled = 0;
% pde.meshadaptfield = 1;
% pde.meshadaptalpha = 0.5;
% pde.meshadaptHelmholtzCoeff = 5e-2;
% pde.meshadaptforcescale = 0.2; % params(3) in pdeapp_frontend.m
% pde.meshadaptsmoothingpasses = 30;
% pde.meshadaptboundaryconditions = [4;2;2;2;2;2];
%-----------------


pde.physicsparam = [rho_ref, u_ref, p_ref, e_ref, L_ref, ...
                     transportFactor, cChem, T_wall, p_out, theta_out,...
                    AVmaxdiv AVdistcoeff pde.AVcoeffStart pde.AVcoeffEnd ];

% Nondimensional freestream conserved state -> eta(1:ncu) for the
% supersonic/characteristic inflow boundary conditions.
uInf = 1.0; vInf = 0.0; rhoInf = 1.0;
eInf = e_phys_inf/e_ref;
rhoEInf = rhoInf*(eInf + 0.5*(uInf^2 + vInf^2));
pde.externalparam = [rhoInf; rhoInf*uInf; rhoInf*vInf; rhoEInf];

%% ---- Boundary conditions ------------------------------------------------
mesh.boundarycondition = [5 1 3 3 2 1];


figure(1); clf; meshplot(mesh); axis equal; axis tight;

master = Master(pde);

% dist = meshdist3(mesh.f, mesh.dgnodes, master.perm, [3 4]); % distance to wall
% mesh.vdg = zeros(size(mesh.dgnodes,1), 1, size(mesh.dgnodes,3));
% nm = 1e3;
% ld = 1./(1 + 200*mesh.dgnodes(:,1,:).^2);
% mesh.vdg(:,1,:) = 0.002*tanh(nm*dist.*ld);

mesh.vdg = zeros(size(mesh.dgnodes,1),2,size(mesh.dgnodes,3));
dist = meshdist3(mesh.f, mesh.dgnodes, master.perm, [3 4]); % distance to wall
nm = 1e3;
ld = 1./(1 + 200*mesh.dgnodes(:,1,:).^2);
mesh.vdg(:,1,:) = dist.*ld;


figure(2); clf; scaplot(mesh, mesh.vdg(:,1,:), [], 1, 0); axis on; axis equal; axis tight;

% Wall-temperature-blended initial condition (same idea as the cylinder
% case's local_initial_udg), sized for ncu = 4.
mesh.udg = local_initial_udg(mesh, dist, db, xiInf, T_phys_inf, T_wall, e_ref, uInf, vInf);


mesh.porder = pde.porder;
mesh.xpe    = master.xpe;
mesh.telem  = master.telem;

pde.tau = 10.0;
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

pde.dt = [1e-4,1e-4,1e-3*(2.^(0:11))];




%% ---- Solve, then continuation in artificial viscosity -------------------
pde.gencode = 1;



[sol, pde, mesh, master, dmd] = exasim(pde, mesh);

xdg = getsolution('dataout/outxdg',dmd, master.npe);
vdg = getsolution('dataout/outvdg',dmd, master.npe);
wdg = getsolutions('dataout/outwdg', dmd);
%mesh = mkmesh_auplate2d4UR(2);
mesh1 = mesh; mesh1.dgnodes = xdg;

figure(1); clf; scaplot(mesh1, wdg(:,1,:)/pRef,[],2,2);
axis equal; axis tight; colorbar; colormap jet;

figure(2); clf; scaplot(mesh1, vdg(:,2,:),[],2,2);
axis equal; axis tight; colorbar;

figure(3); clf; meshplot(mesh1,1); axis equal; axis tight;

sol = fetchsolution(pde, master, dmd, pde.datapath + "/dataout"); sol = sol(:,:,:,end);
wdg = getsolutions(pde.datapath + "/dataout/outwdg", dmd);        wdg = wdg(:,:,:,end);


udg = sol;
save(fullfile(caseDir, 'auplate_equilibrium_sol.mat'), 'udg', 'wdg', 'mesh', 'pde');

rho = udg(:,1,:);
figure(2); clf; scaplot(mesh, rho(:,1,:), [], 1); colorbar; colormap('jet'); title('\rho');
figure(3); clf; scaplot(mesh, udg(:,2,:)./rho(:,1,:), [], 1); colorbar; colormap('jet'); title('u');
figure(4); clf; scaplot(mesh, udg(:,3,:)./rho(:,1,:), [], 1); colorbar; colormap('jet'); title('v');
figure(5); clf; scaplot(mesh, udg(:,4,:), [], 1); colorbar; colormap('jet'); title('\rho E');
figure(6); clf; scaplot(mesh, wdg(:,2,:), [], 1); colorbar; colormap('jet'); title('T');

%% ---- helpers (mirrors examples/.../equilibrium5air_cylindermach8/pdeapp.m) --

function UDG = local_initial_udg(mesh, dist, db, xiInf, Tinf, Twall, eRef, uInf, vInf)
rho = ones(size(dist));
ux = uInf*tanh(10*dist);
uy = vInf*tanh(10*dist);
TnearWall = Tinf + (Twall - Tinf)*exp(-10*dist);
ePhys = local_energy_for_temperature(db, xiInf, TnearWall);
e = ePhys/eRef;
UDG = zeros(size(mesh.dgnodes,1), 4, size(mesh.dgnodes,3));
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
