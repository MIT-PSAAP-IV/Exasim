% Put the Exasim MATLAB frontend on the path. For an installed Exasim use
% run('<prefix>/share/exasim/matlab/exasim_setup.m') instead.
run(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', 'frontends', 'Matlab', 'exasim_setup.m'));

load adaptivemesh.mat

% initialize pde structure and mesh structure
[pde,~] = initializeexasim();
% pde.buildpath=string(pwd());
% pde.exasimpath = string(pwd());
srcdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', 'frontends', 'Matlab');
addpath(char(srcdir + "/Modeling/CNS5air/"));

% Define a PDE model: governing equations, initial solutions, and boundary conditions
pde.model = "ModelD";          % ModelC, ModelD, ModelW
pde.modelfile = "pdemodel_axial";    % name of a file defining the PDE model

% Choose computing platform and set number of processors
pde.platform = "cpu";         % choose this option if NVIDIA GPUs are available
pde.mpiprocs = 4;             % number of MPI processors
pde.hybrid = 1;               % 0 -> LDG, 1 -> HDG
pde.debugmode = 0;
pde.nd = 2;
pde.porder = 2;               % polynomial degree
pde.pgauss = 2*(pde.porder);
pde.nd = 2;
pde.elemtype = 1;
pde.tau = 8.0;  % We nondimensionalize by the free-stream speed of sound, so tau should be greater than the free-stream Mach (~26 here)
pde.gencode = 1;

pde.GMRESrestart = 250;         %try 50
pde.GMRESortho = 1;
pde.linearsolvertol = 1e-6; % GMRES tolerance
pde.linearsolveriter = 500; %try 100
pde.preconditioner = 1;
pde.RBdim = 0;
pde.ppdegree = 0;
pde.NLtol = 1e-6;              % Newton tolerance
pde.NLiter = 10;                % Newton iterations
pde.matvectol=1e-6;             % tolerance for matrix-vector multiplication
pde.dae_alpha = 0;
pde.dae_beta = 0;
pde.dae_gamma = 0;

pde.dt = [1e-3 5e-3 1e-2 3e-2 8e-2];
pde.nstage = 1;
pde.torder = 1;
pde.saveSolFreq = 1;

L_ref    = 1;
T_wall   = 300;
rho_phys_inf = 1.047e-3;
v_phys_inf  = 2500;
T_phys_inf  = 266.5;
p_phys_inf  = 288*rho_phys_inf*T_phys_inf;

[rho_species_inf, rho_phys_inf, rhov_phys_inf, rhoE_phys_inf] = getEquilibriumState(p_phys_inf, T_phys_inf, v_phys_inf);
[rho_ref, v_ref, rhoe_ref, p_ref, T_ref, mu_ref, kappa_ref, lambda_ref, cp_ref, cv_ref] = getReferenceState(p_phys_inf, T_phys_inf, v_phys_inf);

% Nondimensional constants
Re = rho_ref * L_ref * v_ref / mu_ref;
Pr = mu_ref * cp_ref / kappa_ref;
Ec = v_ref^2 / (cp_ref * T_ref);

U_ref = [rho_ref v_ref rhoe_ref T_ref mu_ref kappa_ref cp_ref L_ref Ec Pr Re T_wall];
U_inf = [rho_species_inf/rho_ref rhov_phys_inf/(rho_ref*v_ref) 0 rhoE_phys_inf/rhoe_ref];
pde.physicsparam = U_ref;
pde.externalparam = U_inf;
pde.externalparam(9:13) = U_inf(1:5);
pde.externalparam(14:18) = 0;

master = Master(pde);
dist = meshdist3(mesh.f,mesh.dgnodes,master.perm,[4]); % distance to the wall

mesh.boundarycondition = [5 2 1 8]; % symmetry, outflow, inflow, wall
%fb = [f_in, f_out, f_iso, f_slip, f_grad, f_noncat, f_cat, f_cat_gam, f_cat_gam_consistent];
mesh.vdg = 1e-4*tanh(dist*100);

% generate input files and store them in datain folder
[pde,mesh,master,dmd] = preprocessing(pde,mesh);

% generate source codes and store them in app folder
kkgencode(pde);
compilerstr = cmakecompile(pde);
runcode(pde,1);

udg = getsolutions(pde.datapath + "/dataout/outudg", dmd);
wdg = getsolutions(pde.datapath + "/dataout/outwdg", dmd);
udg = udg(:,:,:,end);
wdg = wdg(:,:,:,end);

rho = sum(udg(:,1:5,:),2);
for i = 1:5
  figure(i); clf; scaplot(mesh, udg(:,i,:)./rho(:,1,:), [], 1); colorbar; colormap('jet'); axis tight;
end
figure(6); clf; scaplot(mesh, rho(:,1,:), [], 1); colorbar; colormap('jet'); axis tight;
figure(7); clf; scaplot(mesh, v_ref*udg(:,6,:)./rho(:,1,:), [], 1); colorbar; colormap('jet'); axis tight;
figure(8); clf; scaplot(mesh, v_ref*udg(:,7,:)./rho(:,1,:), [], 1); colorbar; colormap('jet'); axis tight;
figure(9); clf; scaplot(mesh, T_ref*wdg(:,1,:,end), [], 1); colorbar; colormap('jet'); axis tight;

rho = sum(udg(:,1:5,:), 2);

figure(1); clf;
set(gcf, 'Color', 'w');
for i = 1:5
    subplot(2, 3, i);
    scaplot(mesh, udg(:,i,:)./rho(:,1,:), [], 1);
    colorbar;
    colormap('jet');
    axis tight;
    title(sprintf('Y_%d', i));
    set(gca,'FontSize',16); axis tight;
end
subplot(2, 3, 6);
meshplot(mesh,1); axis on; axis equal; axis tight;
title("Mesh");
set(gca,'FontSize',16); axis tight;
exportgraphics(gcf, 'species_mass_fractions.png', 'Resolution', 300);

figure(10); clf;
subplot(1,3,1);
scaplot(mesh, v_ref*udg(:,6,:)./rho(:,1,:), [], 1);
colorbar; colormap('jet'); axis tight;
title('u');
set(gca,'FontSize',16); axis tight;

subplot(1,3,2);
scaplot(mesh, v_ref*udg(:,7,:)./rho(:,1,:), [], 1);
colorbar; colormap('jet'); axis tight;
title('v');
set(gca,'FontSize',16); axis tight;

subplot(1,3,3);
scaplot(mesh, T_ref*wdg(:,1,:,end), [], 1);
colorbar; colormap('jet'); axis tight;
title('T');
set(gca,'FontSize',16); axis tight;

set(gcf, 'Color', 'w');
exportgraphics(gcf, 'flow_fields.png', 'Resolution', 300);
