%% summary

%% Kn = 0.05, Mach 10, Argon flow over cylinder (Lofthouse case)
%{
No-slip phase:
  ├─ Iter 1: Create initial guess
  ├─ Iter 2-6: Refine with NO-SLIP BC
  └─ SAVE: sol0 and T0 (no-slip result)

Slip phase:
  ├─ Iter 10-12: Use sol from Iter 9 as initial guess
  ├─ Change sigmaV, sigmaT in pde.physicsparam
  ├─ Keep mesh and spatial discretization same
  ├─ Solver converges quickly (warm start)
  └─ SAVE: T1 (slip result)

Comparison:
  └─ Plot T0 vs T1 on same axes

%}

% Put the Exasim MATLAB frontend on the path. For an installed Exasim use
% run('<prefix>/share/exasim/matlab/exasim_setup.m') instead.
run(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', 'frontends', 'Matlab', 'exasim_setup.m'));

% initialize pde structure and mesh structure
[pde,~] = initializeexasim();

% Define a PDE model: governing equations, initial solutions, and boundary conditions
pde.model = "ModelD";          % ModelC, ModelD, ModelW
pde.modelfile = "pdemodel";    % name of a file defining the PDE model

% Choose computing platform and set number of processors
pde.platform = "cpu";         % choose this option if NVIDIA GPUs are available
pde.mpiprocs = 4;              % number of MPI processors
pde.hybrid = 1;
pde.porder = 2;          % polynomial degree

% mesh = mkmesh_cyl(pde.porder);
% % iso-thermal wall, supersonic outflow, supersonic inflow
% mesh.boundarycondition = [1;3;2]; 

D    = 12.0 * 0.0254;       % cylinder diameter [m], 12 inches
%mesh = mkmesh_fullcyl(pde.porder, D/2);
%mesh.boundarycondition = [1;2;3]; 

mesh = mkmesh_cyl(pde.porder, D/2, 1/2, 1.6/2, 2.0, 60, 60);
mesh.boundarycondition = [4;3;2]; 

R = 208.13;
gam = 5/3;                     % specific heat ratio
Re = 4778;                     % Reynolds number
Pr = 2/3;                      % Prandtl number    
Minf = 10;                     % Mach number
Tref  = 200;
rho_ref = 2.818e-5;            % physical freestream density [kg/m^3]
Twall = 500;
%gas dependent (For Argon)
omega = 0.7340; 
mu_ref = 5.0711e-05;
Tmu_ref = 1000;

%artificial viscosity factor for final simulation
vis_factor = 5e-3;

pinf = 1/(gam*Minf^2);
Tinf = pinf/(gam-1);
alpha = 0;                % angle of attack
rinf = 1.0;                     % freestream density
ruinf = cos(alpha);             % freestream horizontal velocity
rvinf = sin(alpha);             % freestream vertical velocity
pinf = 1/(gam*Minf^2);          % freestream pressure
rEinf = 0.5+pinf/(gam-1);       % freestream energy
T_scale = Tref/Tinf;

mu_inf  = mu_ref * (Tref  / Tmu_ref)^omega;    % freestream dynamic viscosity [Pa s]
sigmaV = 0;
sigmaT = 0;

sigmaV_final=1.0;
sigmaT_final=1.875;

pde.gencode = 0;
pde.physicsparam = [gam Re Pr Minf rinf ruinf rvinf rEinf Tinf Tref Twall mu_inf mu_ref Tmu_ref omega R sigmaV sigmaT rho_ref];
pde.tau = 10.0;                  % DG stabilization parameter
pde.GMRESrestart = 500;         %try 50
pde.linearsolvertol = 1e-8; % GMRES tolerance
pde.linearsolveriter = 500; %try 100
pde.RBdim = 0;
pde.ppdegree = 0;
pde.NLtol = 1e-6;              % Newton tolerance
pde.NLiter = 12;                 % Newton iterations
pde.matvectol=1e-6;             % tolerance for matrix-vector multiplication

% initial artificial viscosity
mesh.f = facenumbering(mesh.p,mesh.t,pde.elemtype,mesh.boundaryexpr,mesh.periodicexpr);
dist = meshdist3(mesh.f,mesh.dgnodes,mesh.perm,[1]); % distance to the wall
mesh.vdg = zeros(size(mesh.dgnodes,1),1,size(mesh.dgnodes,3));
nm = 25;
%initial artificial viscosity for stability 
mesh.vdg(:,1,:) = 0.05*tanh(dist*nm);
figure(1); clf; scaplot(mesh, mesh.vdg,[],2); colormap('jet'); colorbar;

% intial solution
ui = [rinf ruinf rvinf rEinf];
UDG = initu(mesh,{ui(1),ui(2),ui(3),ui(4)}); % freestream 
UDG(:,2,:) = UDG(:,2,:).*tanh(nm*dist);
UDG(:,3,:) = UDG(:,3,:).*tanh(nm*dist);
TnearWall = Tinf * (Twall/Tref-1) * exp(-nm*dist) + Tinf;
UDG(:,4,:) = TnearWall + 0.5*(UDG(:,2,:).*UDG(:,2,:) + UDG(:,3,:).*UDG(:,3,:));
mesh.udg = UDG;

figure(2); clf; scaplot(mesh, eulereval(mesh.udg, 'M',gam,Minf),[],1); colormap('jet'); colorbar;


%%%% No slip BC (iter 1-6 ) sigma_v=0, sigma_T=0;
[sol,pde,mesh,master] = exasim(pde,mesh);
figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[],1); colormap('jet'); colorbar;


%%%%% Iterations 2-6 (reduce artificial viscosity)
disp("Iter 2")
pde.physicsparam = [gam Re Pr Minf rinf ruinf rvinf rEinf Tinf Tref Twall mu_inf mu_ref Tmu_ref omega R sigmaV sigmaT rho_ref];
%reducing artificial viscosity 
mesh.vdg(:,1,:) = 0.02.*tanh(dist*nm);
%use previous solution
mesh.udg = sol;
[pde,mesh,master,dmd] = preprocessing(pde,mesh);

dataout_custom = fullfile(pwd, 'dataout');
fprintf('Looking for dataout in: %s\n', dataout_custom);

runcode(pde, 1); % run C++ code
sol = fetchsolution(pde,master,dmd, pde.datapath + "/dataout" + model_strn(pde));
figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[],2); colormap('jet'); colorbar;

disp("Iter 3")
mesh.vdg(:,1,:) = 0.005.*tanh(dist*nm);
mesh.udg = sol;
[pde,mesh,master,dmd] = preprocessing(pde,mesh);
runcode(pde, 1); % run C++ code
sol = fetchsolution(pde,master,dmd, pde.datapath + "/dataout" + model_strn(pde));
figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[],2); colormap('jet'); colorbar;

disp("Iter 4")
mesh.vdg(:,1,:) = 0.0025.*tanh(dist*nm);
mesh.udg = sol;
[pde,mesh,master,dmd] = preprocessing(pde,mesh);
runcode(pde, 1); % run C++ code
sol = fetchsolution(pde,master,dmd, pde.datapath + "/dataout" + model_strn(pde));
figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[],2); colormap('jet'); colorbar;

disp("Iter 5")
mesh.vdg(:,1,:) = 0.0020.*tanh(dist*nm);
mesh.udg = sol;
[pde,mesh,master,dmd] = preprocessing(pde,mesh);
runcode(pde, 1); % run C++ code
sol = fetchsolution(pde,master,dmd, pde.datapath + "/dataout" + model_strn(pde));
figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[],2);colormap('jet'); colorbar;
figure(2); clf; scaplot(mesh, T_scale*eulereval(sol, 't',gam,Minf),[],2); colormap('jet'); colorbar;

disp("Iter 6")
mesh.vdg(:,1,:) = 0.0015.*tanh(dist*nm);
mesh.udg = sol;
[pde,mesh,master,dmd] = preprocessing(pde,mesh);
runcode(pde, 1); % run C++ code
sol = fetchsolution(pde,master,dmd, pde.datapath + "/dataout" + model_strn(pde));
figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[],2);colormap('jet'); colorbar;
figure(2); clf; scaplot(mesh, T_scale*eulereval(sol, 't',gam,Minf),[],2); colormap('jet'); colorbar;

disp("Iter 7")
mesh.vdg(:,1,:) = 0.0011.*tanh(dist*nm);
mesh.udg = sol;
[pde,mesh,master,dmd] = preprocessing(pde,mesh);
runcode(pde, 1); % run C++ code
sol = fetchsolution(pde,master,dmd, pde.datapath + "/dataout" + model_strn(pde));
figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[],2);colormap('jet'); colorbar;
figure(2); clf; scaplot(mesh, T_scale*eulereval(sol, 't',gam,Minf),[],2); colormap('jet'); colorbar;

disp("Iter 8")
mesh.vdg(:,1,:) = 0.0008.*tanh(dist*nm);
mesh.udg = sol;
[pde,mesh,master,dmd] = preprocessing(pde,mesh);
runcode(pde, 1); % run C++ code
sol = fetchsolution(pde,master,dmd, pde.datapath + "/dataout" + model_strn(pde));
figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[],2);colormap('jet'); colorbar;
figure(2); clf; scaplot(mesh, T_scale*eulereval(sol, 't',gam,Minf),[],2); colormap('jet'); colorbar;

disp("Iter 9")
mesh.vdg(:,1,:) = 0.0006.*tanh(dist*nm);
mesh.udg = sol;
[pde,mesh,master,dmd] = preprocessing(pde,mesh);
runcode(pde, 1); % run C++ code
sol = fetchsolution(pde,master,dmd, pde.datapath + "/dataout" + model_strn(pde));
figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[],2);colormap('jet'); colorbar;
figure(2); clf; scaplot(mesh, T_scale*eulereval(sol, 't',gam,Minf),[],2); colormap('jet'); colorbar;

%% EXTRACT  the no-slip temperature 
Tphys = T_scale*eulereval(sol, 't',gam,Minf); %convert to physcial temp. 
X = mesh.dgnodes(:,1,:);   %get x-coordinates
Y = mesh.dgnodes(:,2,:);   %get y-coordinates
%find nodes on symmetry line 
ind = find(abs(Y(:)) < 1e-6 & X(:) < 0);
X0 = X(ind); Y0 = Y(ind);
[X0, ii] = sort(X0);
Y0 = Y0(ii);
T0 = Tphys(ind);
T0 = T0(ii);
figure(3); clf; plot(2*X0/D+1, T0);

%lets save the no-slip solution ....
sol0 = sol;

disp("Iter 10")
sigmaV = 0.2;
sigmaT = 0.2;
pde.physicsparam = [gam Re Pr Minf rinf ruinf rvinf rEinf Tinf Tref Twall mu_inf mu_ref Tmu_ref omega R sigmaV sigmaT rho_ref];
mesh.udg = sol;
[pde,mesh,master,dmd] = preprocessing(pde,mesh);
runcode(pde, 1); % run C++ code
sol = fetchsolution(pde,master,dmd, pde.datapath + "/dataout" + model_strn(pde));
figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[],2);colormap('jet'); colorbar;
figure(2); clf; scaplot(mesh, T_scale*eulereval(sol, 't',gam,Minf),[],2); colormap('jet'); colorbar;

disp("Iter 11")
sigmaV = 0.5;
sigmaT = 0.5;
pde.physicsparam = [gam Re Pr Minf rinf ruinf rvinf rEinf Tinf Tref Twall mu_inf mu_ref Tmu_ref omega R sigmaV sigmaT rho_ref];
mesh.udg = sol;
[pde,mesh,master,dmd] = preprocessing(pde,mesh);
runcode(pde, 1); % run C++ code
sol = fetchsolution(pde,master,dmd, pde.datapath + "/dataout" + model_strn(pde));
figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[],2);colormap('jet'); colorbar;
figure(2); clf; scaplot(mesh, T_scale*eulereval(sol, 't',gam,Minf),[],2); colormap('jet'); colorbar;

disp("Iter 12")
sigmaV = sigmaV_final;
sigmaT = sigmaT_final;
pde.physicsparam = [gam Re Pr Minf rinf ruinf rvinf rEinf Tinf Tref Twall mu_inf mu_ref Tmu_ref omega R sigmaV sigmaT rho_ref];
mesh.udg = sol;
[pde,mesh,master,dmd] = preprocessing(pde,mesh);
runcode(pde, 1); % run C++ code
sol = fetchsolution(pde,master,dmd, pde.datapath + "/dataout" + model_strn(pde));
figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[],2);colormap('jet'); colorbar;
figure(2); clf; scaplot(mesh, T_scale*eulereval(sol, 't',gam,Minf),[],2); colormap('jet'); colorbar;

%lets extract the slip temperature and compare
Tphys = T_scale*eulereval(sol, 't',gam,Minf); %get slip solution temp
%extract at same locations as no-slip solution
T1 = Tphys(ind);
T1 = T1(ii);
%T0 is the no-slip temperature (extracted and saved earlier)
figure(3); clf; plot(2*X0/D+1, T0, '-', 2*X0/D+1, T1, '--', 'LineWidth',2);
set(gca,'FontSize',20); axis tight;
xlabel('$x/R$','Interpreter','latex'); 
ylabel('$\mbox{Temperature (K)}$','Interpreter','latex'); 
title('Kn = 0.01');
legend("No-slip", "slip", "Location", "NorthWest");
axis([-1 0 0 7000]);

