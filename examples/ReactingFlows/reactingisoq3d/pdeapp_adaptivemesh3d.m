% Export the adaptive reacting isoq mesh as a 3D Text2Code application.
%
% This script mirrors pdeapp_adaptivemesh.m through the PDE setup, but rotates
% the adapted 2D axisymmetric mesh into a 3D quarter-domain and exports the
% resulting application with exporttext2code. It intentionally does not call
% runcode or otherwise run Exasim.

run(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', 'frontends', 'Matlab', 'exasim_setup.m'));

appdir = fileparts(mfilename('fullpath'));
load(fullfile(appdir, 'sol2d.mat'));

if isfield(mesh2d, 'udg') && ~isempty(mesh2d.udg)
    udg2d = mesh2d.udg;
else
    error('pdeapp_adaptivemesh3d:MissingUDG2D', ...
          'mesh2d in sol2d.mat must contain mesh2d.udg.');
end

if isfield(mesh2d, 'vdg') && ~isempty(mesh2d.vdg)
    vdg2d = mesh2d.vdg;
else
    error('pdeapp_adaptivemesh3d:MissingVDG2D', ...
          'mesh2d in sol2d.mat must contain mesh2d.vdg for the 2D AV field.');
end

if isfield(mesh2d, 'wdg') && ~isempty(mesh2d.wdg)
    wdg2d = mesh2d.wdg;
else
    error('pdeapp_adaptivemesh3d:MissingWDG2D', ...
          'mesh2d in sol2d.mat must contain mesh2d.wdg.');
end

[pde,~] = initializeexasim();

srcdir = fullfile(appdir, '..', '..', '..', 'frontends', 'Matlab');
addpath(char(fullfile(srcdir, 'Modeling', 'CNS5air')));
addpath(char(appdir));
addpath(char(fullfile(appdir, '..', 'reactingisoq2d')));

% Define a 3D Cartesian five-species-air model.
pde.model = "ModelD";
pde.modelfile = "pdemodel_cart";

% Computing platform and discretization settings.
pde.platform = "cpu";
pde.mpiprocs = 12;
pde.hybrid = 1;
pde.debugmode = 0;
pde.nd = 3;
pde.porder = 2;
pde.pgauss = 2*pde.porder;
pde.elemtype = 1;
pde.tau = 8.0;
pde.gencode = 1;

pde.GMRESrestart = 500;
pde.GMRESortho = 1;
pde.linearsolvertol = 1e-6;
pde.linearsolveriter = 500;
pde.preconditioner = 1;
pde.RBdim = 0;
pde.ppdegree = 0;
pde.NLtol = 1e-6;
pde.NLiter = 10;
pde.matvectol = 1e-6;
pde.dae_alpha = 0;
pde.dae_beta = 0;
pde.dae_gamma = 0;

pde.dt = [1e-4 1e-3 2e-3*ones(1,14)];
pde.nstage = 1;
pde.torder = 1;
pde.saveSolFreq = 1;

L_ref    = 1;
T_wall   = 300;
rho_phys_inf = 1.047e-3;
v_phys_inf  = 2500;
T_phys_inf  = 266.5;
p_phys_inf  = 288*rho_phys_inf*T_phys_inf;

[rho_species_inf, ~, rhov_phys_inf, rhoE_phys_inf] = getEquilibriumState(p_phys_inf, T_phys_inf, v_phys_inf);
[rho_ref, v_ref, rhoe_ref, ~, T_ref, mu_ref, kappa_ref, ~, cp_ref, ~] = getReferenceState(p_phys_inf, T_phys_inf, v_phys_inf);

% Nondimensional constants.
Re = rho_ref * L_ref * v_ref / mu_ref;
Pr = mu_ref * cp_ref / kappa_ref;
Ec = v_ref^2 / (cp_ref * T_ref);

U_ref = [rho_ref v_ref rhoe_ref T_ref mu_ref kappa_ref cp_ref L_ref Ec Pr Re T_wall];
U_inf = [rho_species_inf/rho_ref rhov_phys_inf/(rho_ref*v_ref) 0 0 rhoE_phys_inf/rhoe_ref];
pde.physicsparam = U_ref;
pde.externalparam = U_inf;
pde.externalparam(10:14) = U_inf(1:5);
pde.externalparam(15:19) = 0;

% Rotate the adapted 2D axisymmetric mesh into a 3D quarter-domain.
nz = 16;
theta = linspace(0, pi/2, nz+1);
mesh = mkmesh_isoq3d3(mesh2d, nz);

r2d = mesh2d.dgnodes(:,2,:);
dR = min(r2d(:));
ymin = min(mesh.p(2,:));
zmin = min(mesh.p(3,:));
xmax = max(mesh.p(1,:));
tol = 1e-8;

mesh.boundaryexpr = { ...
    @(p) abs(p(2,:)-ymin)<tol, ...
    @(p) abs(p(3,:)-zmin)<tol, ...
    @(p) abs(p(1,:)-xmax)<tol, ...
    @(p) abs(p(2,:).^2+p(3,:).^2-dR^2)<tol, ...
    @(p) (-1e-3<p(1,:)) & (p(1,:)<xmax+1e-3) & (sqrt(p(2,:).^2+p(3,:).^2)<0.06+dR), ...
    @(p) abs(p(1,:))<20+1e-6};

% symmetry, symmetry, outflow, symmetry, wall, inflow
mesh.boundarycondition = [5 5 2 5 8 1];

mesh.vdg = extrudesol(vdg2d, pde.porder, nz);
mesh.wdg = extrudesol(wdg2d, pde.porder, nz);

udg2d = udg2d(:,1:8,:);
mesh.udg = extrudesol(udg2d, pde.porder, nz);
mesh.udg(:,9,:) = mesh.udg(:,8,:);
[mesh.udg(:,7,:), mesh.udg(:,8,:)] = extrudevelocity(udg2d(:,7,:), pde.porder, theta);

% pdeapp.gencode = 0;
% % generate input files and store them in datain folder
% [pde,mesh,master,dmd] = preprocessing(pde,mesh);

% generate source codes and store them in app folder
% kkgencode(pde);
% compilerstr = cmakecompile(pde);

% runcode(pde,1);

% Export the Text2Code application. Do not run Exasim from this setup script.
exasimroot = fullfile(appdir, '..', '..', '..');
exportdir = fullfile(exasimroot, 'apps', 'navierstokes', 'reactingisoq3d');
exporttext2code(pde, mesh, exportdir);
