% Add Exasim to Matlab search path
cdir = pwd(); ii = strfind(cdir, "Exasim");
ii = ii(end);
run(cdir(1:(ii+5)) + "/install/setpath.m");

load solp2.mat
load iter7final.mat
load fixed_av.mat
porder = 2;
meshold = mesh;
meshnew = mkmesh_cone2d2dns(porder);
sol1 = fieldatdgnodes(meshold, master, sol, meshnew.dgnodes);
fixed_av = fieldatdgnodes(meshold, master, fixed_av, meshnew.dgnodes);
w1 = fieldatdgnodes(meshold, master, wdg, meshnew.dgnodes);

mesh = meshnew;


% initialize pde structure and mesh structure
[pde,~] = initializeexasim();
% pde.buildpath=string(pwd()); 
% pde.exasimpath = string(pwd());
%addpath(char(srcdir + "/Modeling/CNS5air/"));
pde.buildpath = "/data/data1/GitHub/Exasim/build" + "/transient";
if exist(pde.buildpath, 'dir') == 0
    mkdir(pde.buildpath);
end
addpath(char("CNS5air/"));
%addpath(char("STG5species/"));


% Define a PDE model: governing equations, initial solutions, and boundary conditions
pde.model = "ModelD";          % ModelC, ModelD, ModelW
pde.modelfile = "pdemodel5";    % name of a file defining the PDE model

% Choose computing platform and set number of processors
pde.platform = "cpu";         % choose this option if NVIDIA GPUs are available
pde.mpiprocs = 64;             % number of MPI processors
pde.hybrid = 0;               % 0 -> LDG, 1 -> HDG
pde.debugmode = 0;
pde.nd = 2;
pde.porder = 2;               % polynomial degree
pde.pgauss = 2*(pde.porder);
pde.nd = 2;
pde.elemtype = 1;
pde.tau = 10.0;  % We nondimensionalize by the free-stream speed of sound, so tau should be greater than the free-stream Mach (~26 here)
pde.gencode = 1;

%original
% avcoeff = 0; % fixed bow-shock AV coefficient
% pde.GMRESrestart = 250;         %try 50
% pde.GMRESortho = 1;
% pde.linearsolvertol = 1e-7; % GMRES tolerance
% pde.linearsolveriter = 500; %try 100
% pde.preconditioner = 1;
% pde.precMatrixType = 2;
% app.ptcMatrixType=0;
% pde.RBdim = 0%5;
% pde.ppdegree = 0;
% pde.NLtol = 1e-14;              % Newton tolerance
% pde.NLiter = 2;                % Newton iterations
% pde.matvectol=1e-6;             % tolerance for matrix-vector multiplication
%imported from fixed av case
avcoeff = 5e-7; %8e-7
pde.GMRESrestart = 85;%250;         %try 50
pde.GMRESortho = 1;
pde.linearsolvertol = 1e-6; % GMRES tolerance
pde.linearsolveriter = 100;%500; %try 100
pde.preconditioner = 1;
pde.precMatrixType = 2;
app.ptcMatrixType=0;
pde.RBdim = 0;
pde.ppdegree = 0;
pde.NLtol = 1e-6;              % Newton tolerance
pde.NLiter = 10;                % Newton iterations
pde.matvectol=1e-6;             % tolerance for matrix-vector multiplication
pde.dae_alpha = 0;
pde.dae_beta = 0;
pde.dae_gamma = 0;

L_ref = 1;
T_wall = 293.2;

rho_phys_inf = 0.087;
v_phys_inf   = 3809;
T_phys_inf   = 1355;
p_phys_inf   = 34.2e3;

alpha = 1e4;

% Chemistry order: [N, O, NO, N2, O2]
Yinf = [1e-6;
        0.007;
        0.073;
        0.733;
        0.187];

Yinf = Yinf / sum(Yinf);

rho_species_inf = rho_phys_inf * Yinf;
rhov_phys_inf   = rho_phys_inf * v_phys_inf;

[rhoE_phys_inf, p_check_phys, ~] = energyFromSpecies( ...
    rho_species_inf, T_phys_inf, v_phys_inf, alpha);

rho_ref = rho_phys_inf;
v_ref   = v_phys_inf;
T_ref   = T_phys_inf;
rhoe_ref = rho_ref * v_ref^2;

[~, ~, ~, ~, mu_ref, kappa_ref, ~, cp, ~] = ...
    transportcoefficients(T_ref, rho_species_inf(:), alpha);

mu_ref    = double(mu_ref);
kappa_ref = double(kappa_ref);
cp        = double(cp);

cp_ref = sum(cp(:) .* Yinf);

Re = rho_ref * L_ref * v_ref / mu_ref;
Pr = mu_ref * cp_ref / kappa_ref;
Ec = v_ref^2 / (cp_ref * T_ref);

U_ref = [rho_ref v_ref rhoe_ref T_ref mu_ref kappa_ref cp_ref L_ref Ec Pr Re T_wall avcoeff];
U_inf = [(rho_species_inf(:)'/rho_ref) rhov_phys_inf/(rho_ref*v_ref) 0 rhoE_phys_inf/rhoe_ref];



pde.physicsparam = U_ref;
pde.externalparam = U_inf;
pde.externalparam(9:13) = U_inf(1:5);
pde.externalparam(14:18) = 0;

pde.dt = [5e-5*ones(1,50000)];   % time step sizes
pde.soltime = 1:length(pde.dt); % steps at which solution are collected
%pde.visdt = 0.1; % visualization timestep size
pde.nstage = 3;
pde.torder = 3;
pde.saveSolFreq = 500;
pde.saveSolBouFreq = 4;
pde.ibs = 3;

pde.AV = 1; % composite AV is applied inside pdemodel5.flux
pde.frozenAVflag = 1;
pde.AVsmoothingIter = 2;


mesh.boundarycondition = [5 1 3 3 2 1]; % symmetry, inflow, inflow, wall, outflow
%fb = [f_in, f_out, f_iso, f_slip, f_grad, f_noncat, f_cat, f_cat_gam, f_cat_gam_consistent];

% mesh.udg = 0*rho_species;
% mesh.udg(:,1:5,:) = rho_species/rho_ref;
% mesh.udg(:,6:7,:) = rhov_phys/(rho_ref*v_ref);
% mesh.udg(:,8,:) = rhoE_phys/rhoe_ref;
% mesh.wdg = T_phys/T_ref;


mesh.udg = sol1;
mesh.wdg = w1;
rho_stg = sum(sol1(:,1:5,:), 2);


[~,cgelcon,rowent2elem,colent2elem,~] = mkcgent2dgent(mesh.dgnodes,1e-8);
[~, ~, jac] = volgeom(master.shapent,permute(mesh.dgnodes,[1 3 2]));
jac = reshape(jac,[],1,size(mesh.dgnodes,3));
jac = dg2cg2(jac, cgelcon, colent2elem, rowent2elem);
hm = sqrt(dg2cg2(jac, cgelcon, colent2elem, rowent2elem)); % adapted to this mesh

%load godlike.mat

x1_min = -0.04;   % replace with your values
x1_max = 1.4;
% Extract x-coords — same shape as the tanh result
x1_coords = mesh.dgnodes(:,1,:);
% Build a smooth or hard mask
x1_mask = double((x1_coords >= x1_min) & (x1_coords <= x1_max));
% Compute tanh once
tanh_val = tanh(meshdist3(mesh.f, mesh.dgnodes, master.perm, [3 4]) * 50000);
%7/10 was 9500

x_min = 0.135;
x_max = 1.0;
x_coords = mesh.dgnodes(:,1,:);
x_mask = double((x_coords >= x_min) & (x_coords <= x_max));
dist = meshdist3(mesh.f, mesh.dgnodes, master.perm, [3 4]);
wall_mask = double(dist > 4.5e-3);
muter = wall_mask .* x_mask + 1.0 .* (1 - x_mask);

fixed_av = fixed_av.*muter;


tx = [0.9853,   0.987762, 0.988056];
ty = [0.088140, 0.09176,  0.0884031];

xc = mesh.dgnodes(:,1,:);
yc = mesh.dgnodes(:,2,:);

d1 = (tx(2)-tx(1)).*(yc-ty(1)) - (ty(2)-ty(1)).*(xc-tx(1));
d2 = (tx(3)-tx(2)).*(yc-ty(2)) - (ty(3)-ty(2)).*(xc-tx(2));
d3 = (tx(1)-tx(3)).*(yc-ty(3)) - (ty(1)-ty(3)).*(xc-tx(3));

has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0);
has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0);
tri_mask = double(~(has_neg & has_pos));




mesh.vdg = zeros(size(sol1,1), 13, size(sol1,3));
mesh.vdg(:,1:5,:) = sol1(:,1:5,:);
mesh.vdg(:,6,:) = sol1(:,6,:)./rho_stg;
mesh.vdg(:,7,:) = sol1(:,7,:)./rho_stg;
mesh.vdg(:,8,:) = w1(:,1,:);
mesh.vdg(:,9,:) = hm;
mesh.vdg(:,10,:) = (tanh_val .* x1_mask + 1.0 .* (1 - x1_mask)) .* (1 - 0.5 .* tri_mask);
%mesh.vdg(:,10,:) = tanh_val .* x1_mask + 1.0 .* (1 - x1_mask);
mesh.vdg(:,11,:) = double(fixed_av > 1e-6);
mesh.vdg(:,12,:) = .000065*fixed_av;
mesh.vdg(:,13,:) = 0;
%test = mesh.vdg(:,10,:);
% scaplot(mesh,mesh.vdg(:,12,:));
% figure(2);clf;scaplot(mesh,mesh.vdg(:,12,:));
% colorbar;clim([1e-7 1e-6]);

pde.exportapp = 'test';
pde.frontendprovider = true;
pde.buildandrun = false;
[sol,pde,mesh,master,dmd] = exasim(pde,mesh);

% stgNmode = 200;
% gridLength = 1e-03;
% turbLengthFactor = 10;
% visc = mu_ref; %mu_ref is dimensional?
% turbIntensity = 1/100;
% Ustg = v_phys_inf;
% pde.stgdata = stghomogeneousturbulence(gridLength, turbLengthFactor, visc, turbIntensity, Ustg, stgNmode+1);
% pde.stgparam = [1 0 0];
% pde.stgNmode = stgNmode;
% pde.stgib = [1];
% 
% 
% %pde.preprocessingmode=1; 
% [pde,mesh,master,dmd] = preprocessing(pde,mesh);
% kkgencode(pde);
% compilerstr = cmakecompile(pde);
% runcode(pde,1);



udg = getsolutions(pde.buildpath + "/dataout/out", dmd);
wdg = getsolutions(pde.buildpath + "/dataout/outwdg", dmd);
% sol2 = getsolution(pde.buildpath + "/dataout/out",dmd,9);
% wdg2 = getsolution(pde.buildpath + "/dataout/out_wdg",dmd,9);
 udg = readout(pde.buildpath + "/dataout/outudg_t5200_np", dmd, 9);
 wdg = readwut(pde.buildpath + "/dataout/outwdg_t5200_np", dmd, 9);

udg = udg(:,:,:,end);
wdg = wdg(:,:,:,end);
rho = sum(udg(:,1:5,:),2);
for i = 1:5  
  figure(i); clf; scaplot(mesh, udg(:,i,:)./rho(:,1,:), [], 1); colorbar; colormap('jet')

end
figure(6); clf; scaplot(mesh, rho(:,1,:), [0 1.8], 1); colorbar; colormap('jet');
figure(7); clf; scaplot(mesh, udg(:,6,:)./rho(:,1,:), [], 1); colorbar; colormap('jet')
figure(8); clf; scaplot(mesh, udg(:,7,:)./rho(:,1,:), [0 0.2], 1); colorbar; colormap('jet')
figure(9); clf; scaplot(mesh, udg(:,8,:), [0 1.25], 1); colorbar; colormap('jet');
figure(10); clf; scaplot(mesh, wdg(:,1,:,end), [1 2], 1); colorbar; colormap('jet');
figure(11);clf;scaplot(mesh, wdg(:,1,:,end)./rho(:,1,:), [], 1); colorbar; colormap('jet');


% ── figure(12): static pressure ──────────────────────────────────────────

Mw = [0.014007, 0.015999, 0.030006, 0.028014, 0.031999];


% Recover dimensional fields
T_dim     = wdg(:,1,:,end) * T_ref;       % (ngp, 1, nelem)  [K]
rho_i_dim = udg(:,1:5,:)   * rho_ref;     % (ngp, 5, nelem)  [kg/m³]

% Flatten to 2-D so the pressure function can be called vectorised
ngp_   = size(rho_i_dim, 1);
nelem_ = size(rho_i_dim, 3);

T_flat     = T_dim(:);                                              % (ngp*nelem, 1)
rho_i_flat = reshape(permute(rho_i_dim,[1 3 2]), ngp_*nelem_, 5);  % (ngp*nelem, 5)

% pressure() as written uses scalar * ; replicate its logic with .* for arrays
RU     = 8.314471468617452;
p_flat = T_flat .* sum(rho_i_flat ./ Mw, 2) .* RU;   % (ngp*nelem, 1) [Pa]
p_dim  = reshape(p_flat, ngp_, 1, nelem_);             % (ngp, 1, nelem)

figure(12); clf; scaplot(mesh, p_dim, [3e4 6e4], 1); colorbar; colormap('jet');
title('Static pressure [Pa]');
