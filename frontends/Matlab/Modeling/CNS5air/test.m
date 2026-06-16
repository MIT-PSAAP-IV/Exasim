% Add Exasim to Matlab search path
cdir = pwd(); ii = strfind(cdir, "Exasim");
ii = ii(end);
run(cdir(1:(ii+5)) + "/install/setpath.m");

% initialize pde structure and mesh structure
[pde,~] = initializeexasim();
pde.buildpath=string(pwd()); 
pde.exasimpath = string(pwd());
addpath(char(srcdir + "/Modeling/reactingflow/kineticsMatlab/"));
addpath(char(srcdir + "/Modeling/reactingflow/transportMatlab/"));
addpath(char(srcdir + "/Modeling/reactingflow/thermodynamicsMatlab/"));
addpath(char(srcdir + "/Modeling/five_species_air/"));

% Define a PDE model: governing equations, initial solutions, and boundary conditions
pde.model = "ModelD";          % ModelC, ModelD, ModelW
pde.modelfile = "pdemodel_axial_air5";    % name of a file defining the PDE model
pde.buildpath = string(pwd());

% Choose computing platform and set number of processors
pde.platform = "gpu";         % choose this option if NVIDIA GPUs are available
pde.mpiprocs = 1;             % number of MPI processors
pde.hybrid = 1;               % 0 -> LDG, 1 -> HDG
pde.debugmode = 0;
pde.nd = 2;

% Re = 2.36e5;                     % Reynolds number
% Pr = 0.71;                       % Prandtl number    

gam = 1.4;                       % specific heat ratio
Minf = 8.98;                     % Mach number
L_ref    = 1;
T_wall   = 300;
%rho_phys_inf = 0.001547;
v_phys_inf  = 5956;
T_phys_inf  = 901;
p_phys_inf  = 400;

[rho_species_inf, rho_phys_inf, rhov_phys_inf, rhoE_phys_inf] = getEquilibriumState(p_phys_inf, T_phys_inf, v_phys_inf);
[rho_ref, v_ref, rhoe_ref, p_ref, T_ref, mu_ref, kappa_ref, lambda_ref, cp_ref, cv_ref] = getReferenceState(p_phys_inf, T_phys_inf, v_phys_inf);

% Nondimensional constants
Re = rho_ref * L_ref * v_ref / mu_ref;
Pr = mu_ref * cp_ref / kappa_ref;
Ec = v_ref^2 / (cp_ref * T_ref);

U_ref = [rho_ref v_ref rhoe_ref T_ref mu_ref kappa_ref cp_ref L_ref Ec Pr Re T_wall];
U_inf = [rho_species_inf/rho_ref rhov_phys_inf/(rho_ref*v_ref) 0 rhoE_phys_inf/rhoe_ref];

% Set solp2_path to your reactingcylinder solution (e.g. the repo's
% examples/NavierStokes/reactingcylinder/solp2.mat); no absolute dev path here.
load(solp2_path);
[rho_species, rhov_phys, rhoE_phys, rho_phys, T_phys, p_phys] = initializeFromIdealGasSolution(sol, Minf, rho_ref, v_ref, T_ref);
e = rho_phys - sum(rho_species,2);
max(abs(e(:)))

udg = 0*rho_species;
udg(:,1:5,:) = rho_species/rho_ref;
udg(:,6:7,:) = rhov_phys/(rho_ref*v_ref);
udg(:,8,:) = rhoE_phys/rhoe_ref;
wdg = T_phys/T_ref;
dist = meshdist3(mesh.f,mesh.dgnodes,master.perm,[1]); % distance to the wall
vdg = 1e-3*tanh(dist*10);
qdg = gradu(permute(master.shapen(:,:,2:3),[2 1 3]), mesh.dgnodes, -udg);

mu = U_ref;
eta = U_ref;
t = [];
for j = 1:size(udg,3)  
  u = udg(5,:,j);
  q = qdg(5,:,j);
  w = wdg(5,:,j);
  v = vdg(5,:,j);
  x = mesh.dgnodes(5,:,j);
  f = fluxcart2d(u(:), q(:), w(:), v(:), x(:), t, mu, eta);
  s = sourcend(u(:), q(:), w(:), v(:), x(:), t, mu, eta);
  e = eosnd(u(:), q(:), w(:), v(:), x(:), t, mu, eta);
  e1 = sourcew2d(u(:), q(:), w(:), v(:), x(:), t, mu, eta);
  s1 = source2d(u(:), q(:), w(:), v(:), x(:), t, mu, eta);
  f1 = flux2d(u(:), q(:), w(:), v(:), x(:), t, mu, eta);
  max(abs(e(:)-e1(:)))
  max(abs(s(:)-s1(:)))
  max(abs(f(:)-f1(:)))

  f2 = fluxaxial2d(u(:), q(:), w(:), v(:), x(:), t, mu, eta);
  s2 = sourceaxial2d(u(:), q(:), w(:), v(:), x(:), t, mu, eta);
  f3 = fluxaxialns_air5(u(:), q(:), w(:), v(:), x(:), t, mu, eta);
  s3 = sourceaxialns_air5(u(:), q(:), w(:), v(:), x(:), t, mu, eta);
  e3 = sourcewaxialns_air5(u(:), q(:), w(:), v(:), x(:), t, mu, eta);
  max(abs(f2(:)-f3(:)))
  max(abs(s2(:)-s3(:)))
  max(abs(e(:)-e3(:)))
end 

figure(1); clf; scaplot(mesh, udg(:,6,:),[0 1.5],1);
colorbar; colormap('jet'); axis on; axis equal; axis tight; set(gca,'FontSize',16);
figure(2); clf; scaplot(mesh, qdg(:,6,:),[-4 5],1);
colorbar; colormap('jet'); axis on; axis equal; axis tight; set(gca,'FontSize',16);

qdg = gradu(permute(master.shapen(:,:,2:3),[2 1 3]), mesh.dgnodes, mesh.dgnodes(:,1,:));
figure(2); clf; scaplot(mesh, qdg(:,1,:),[],1);
colorbar; colormap('jet'); axis on; axis equal; axis tight; set(gca,'FontSize',16);

% f = fluxcart2d(u, q, w, v, x, t, mu, eta);


% figure(1); clf; scaplot(mesh, vdg,[],1);
% colorbar; colormap('jet'); axis on; axis equal; axis tight; set(gca,'FontSize',16);

% rhoE_phys_inf = p_phys_inf/(gam-1) + 0.5*rho_phys_inf*v_phys_inf^2;
% info = equilibrate(p_phys_inf, T_phys_inf, [v_phys_inf 0]);
% rho_species_inf = info.rho_species;
% 
% rho_ref = rho_phys_inf;
% v_ref = v_phys_inf;
% T_ref = T_phys_inf;
% info = equilibrate(rho_ref * T_ref * 287, T_ref, [v_ref 0]);
% rho_species_ref = info.rho_species;
% rho_ref = sum(rho_species_ref);
% rhoe_ref = rho_ref*v_ref^2;
% p_ref = rho_ref*v_ref^2;
% 
% [~, ~, ~, ~, mu_ref, kappa_ref, lambda, cp, cv] = transportcoefficients(T_ref, rho_species_ref(:), 1e3);
% cp_ref = sum(cp.*rho_species_ref(:)/rho_ref);
% cv_ref = sum(cv.*rho_species_ref(:)/rho_ref);
% % Nondimensional constants
% Re = rho_ref * L_ref * v_ref / mu_ref;
% Pr = mu_ref * cp_ref / kappa_ref;
% Ec = v_ref^2 / (cp_ref * T_ref);
% %Re*Pr*Ec-rho_ref*L_ref*v_ref^3/(kappa_ref*T_ref);
% 
% mu = [rho_ref v_ref rhoe_ref T_ref mu_ref kappa_ref cp_ref L_ref Ec Pr Re T_wall];
% eta = [rho_species_inf/rho_ref sum(rho_species_inf)*v_phys_inf/(rho_ref*v_ref) 0 rhoE_phys_inf/rhoe_ref];

% [rho0_phys, v0_phys, T0_phys, p0_phys] = computePhysicalStateFromNondim(sol, Minf, rho_ref, v_ref, T_ref);
% e = T_phys - T0_phys;
% max(abs(e(:)))
% e = p_phys - p0_phys;
% max(abs(e(:)))
% e = sol(:,1,:).*rho_ref - rho0_phys;
% max(abs(e(:)))

figure(1); clf; scaplot(mesh, rho_phys,[rho_ref 10*rho_ref],1);
colorbar; colormap('jet'); axis on; axis equal; axis tight; set(gca,'FontSize',16);

figure(2); clf; scaplot(mesh, sol(:,1,:).*rho_ref,[rho_ref 10*rho_ref],1);
colorbar; colormap('jet'); axis on; axis equal; axis tight; set(gca,'FontSize',16);

figure(3); clf; scaplot(mesh, T_phys,[],1);
colorbar; colormap('jet'); axis on; axis equal; axis tight; set(gca,'FontSize',16);

figure(4); clf; scaplot(mesh, p_phys,[],1);
colorbar; colormap('jet'); axis on; axis equal; axis tight; set(gca,'FontSize',16);

figure(5); clf; scaplot(mesh, rhoE_phys,[],1);
colorbar; colormap('jet'); axis on; axis equal; axis tight; set(gca,'FontSize',16);

% 
% 
% % Set discretization parameters, physical parameters, and solver parameters
% pde.porder = 2;             % polynomial degree
% porder = pde.porder;
% pde.pgauss = 2*(pde.porder);
% pde.elemtype = 1;
% ns=5;
% pde.tau = 28.0;  % We nondimensionalize by the free-stream speed of sound, so tau should be greater than the free-stream Mach (~26 here)
% dfactor = 10;
% pde.gencode = 1;
% 
% % These values are found with a call to Mutation using equilibrium composition at the free-stream
% rho_inf = [0, 0, 0, 1.79883e-03, 5.51171e-04];
% u_inf   =  6921;
% T_inf   = 260.6;
% T_wall  = 300;
% L_ref   = 1.0;  
% 
% 
% 
% [constants] = init_rho_i_T(rho_inf(:), T_inf, u_inf, L_ref);
% 
% rho_ref = constants.rho_ref;
% u_ref = constants.u_ref;
% rhoE_ref = constants.rhoE_ref; 
% T_ref = constants.T_ref;
% mu_ref = constants.mu_ref;
% kappa_ref = constants.kappa_ref;
% cp_ref = constants.cp_ref; 
% 
% rho_inf = constants.rho_inf;
% rhou_inf = constants.rhou_inf;
% rhov_inf = constants.rhov_inf;
% rhoE_inf = constants.rhoE_inf;
% 
% % % Reference quantities for nondimensionalizing
% avfactor = 50;       % when nondim uses a_ref rather than u_inf, I found I needed to start with a larger viscosity param; perhaps due to conditioning
% 
% % Nondimensional constants
% Re = rho_ref * L_ref * u_ref / mu_ref;
% Pr = mu_ref * cp_ref / kappa_ref;
% Ec = u_ref^2 / (cp_ref * T_ref);
% 
% % Load into Exasim data structures
% pde.physicsparam(1:ns) = rho_inf;
% pde.physicsparam(ns+1) = rhou_inf;
% pde.physicsparam(ns+2) = rhov_inf;
% pde.physicsparam(ns+3) = rhoE_inf;
% pde.physicsparam(9)  = Pr;
% pde.physicsparam(10) = Re;
% pde.physicsparam(11) = Ec;
% 
% % Nondimensional terms
% pde.externalparam = zeros(1,8);
% pde.externalparam(1) = rho_ref;  
% pde.externalparam(2) = u_ref;    
% pde.externalparam(3) = rhoE_ref; 
% pde.externalparam(4) = T_ref;
% pde.externalparam(5) = mu_ref;
% pde.externalparam(6) = kappa_ref;
% pde.externalparam(7) = cp_ref; 
% pde.externalparam(8) = L_ref;
% pde.externalparam(9) = Ec;
% pde.externalparam(10) = Pr;
% pde.externalparam(11) = Re;
% pde.externalparam(12) = T_wall;
% % pde.externalparam(13) = 1; % gamma wall
% 
% pde.GMRESrestart = 250;         %try 50
% pde.linearsolvertol = 1e-5; % GMRES tolerance
% pde.linearsolveriter = 500; %try 100
% pde.RBdim = 0;
% pde.ppdegree = 0;
% pde.NLtol = 1e-6;              % Newton tolerance
% pde.NLiter = 20;                 % Newton iterations
% pde.matvectol=1e-6;             % tolerance for matrix-vector multiplication
% pde.timestepOffset=0;
% pde.dae_alpha = 0.0;
% pde.dae_beta = 0.0;
% pde.AV = 0;%
% pde.AVsmoothingIter = 0;
% 
% % Make mesh
% load solp2.mat; 
% mesh.boundarycondition = [5 1 1 3 2]; % symmetry, inflow, inflow, wall, outflow
% dist = meshdist3(mesh.f,mesh.dgnodes,master.perm,[4]); % distance to the wall
% 
% % intial solution
% ui = [ rho_inf(:) / rho_ref; rhou_inf / (rho_ref *u_ref); 0.0; rhoE_inf / (rhoE_ref)];
% UDG = initu(mesh,{ui(1),ui(2),ui(3),ui(4),ui(5),ui(6),ui(7),ui(8),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0});
% UDG(:,1:ns,:) = rho_species / rho_ref;
% UDG(:,ns+1,:) = rho .* v_phys(:,1,:)/(rho_ref *u_ref);
% UDG(:,ns+2,:) = rho .* v_phys(:,2,:)/(rho_ref *u_ref);
% UDG(:,ns+3,:) = rhoE/(rhoE_ref);
% 
% mesh.udg = UDG;
% mesh.wdg = T_phys/T_ref;%
% %mesh.vdg = 0.1*tanh(dist*dfactor); % For the free-stream solution with this nondim, I typically need a large amount of viscosity
% mesh.vdg = 20*mesh.vdg;
% 
% % generate input files and store them in datain folder
% [pde,mesh,master,dmd] = preprocessing(pde,mesh);
% 
% % pde.gencode = 0;
% % generate source codes and store them in app folder
% if pde.gencode==1
%   kkgencode(pde);
%   compilerstr = cmakecompile(pde);
% end
% 
% %%
% runcode(pde,1);
% 
