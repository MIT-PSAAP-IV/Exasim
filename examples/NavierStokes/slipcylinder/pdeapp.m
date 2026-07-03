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
pde.mpiprocs = 1;              % number of MPI processors
pde.hybrid = 1;
pde.porder = 2;          % polynomial degree

% mesh = mkmesh_cyl(pde.porder);
% % iso-thermal wall, supersonic outflow, supersonic inflow
% mesh.boundarycondition = [1;3;2]; 

% D       = 12.0 * 0.0254;       % cylinder diameter [m], 12 inches
mesh = mkmesh_fullcyl(pde.porder, D/2);
mesh.boundarycondition = [4;2;3]; 

R = 287.05;
gam = 5/3;                     % specific heat ratio
Re = 1000;                     % Reynolds number
Pr = 2/3;                      % Prandtl number    
Minf = 25;                     % Mach number
Tref  = 200;
Twall = 1500;
pinf = 1/(gam*Minf^2);
Tinf = pinf/(gam-1);
alpha = 0;                % angle of attack
rinf = 1.0;                     % freestream density
ruinf = cos(alpha);             % freestream horizontal velocity
rvinf = sin(alpha);             % freestream vertical velocity
pinf = 1/(gam*Minf^2);          % freestream pressure
rEinf = 0.5+pinf/(gam-1);       % freestream energy

R = 208.13;
omega = 0.7340; 
mu_ref = 5.0711e-05;
Tmu_ref = 1000;
mu_inf  = mu_ref * (Tref  / Tmu_ref)^omega;    % freestream dynamic viscosity [Pa s]
sigmaV = 1;
sigmaT = 1.875;

pde.physicsparam = [gam Re Pr Minf rinf ruinf rvinf rEinf Tinf Tref Twall mu_inf mu_ref Tmu_ref omega R sigmaV sigmaT];
pde.tau = 5.0;                  % DG stabilization parameter
pde.GMRESrestart = 500;         %try 50
pde.linearsolvertol = 1e-8; % GMRES tolerance
pde.linearsolveriter = 500; %try 100
pde.RBdim = 0;
pde.ppdegree = 0;
pde.NLtol = 1e-6;              % Newton tolerance
pde.NLiter = 30;                 % Newton iterations
pde.matvectol=1e-6;             % tolerance for matrix-vector multiplication

% initial artificial viscosity
mesh.f = facenumbering(mesh.p,mesh.t,pde.elemtype,mesh.boundaryexpr,mesh.periodicexpr);
dist = meshdist3(mesh.f,mesh.dgnodes,mesh.perm,[1]); % distance to the wall
mesh.vdg = zeros(size(mesh.dgnodes,1),1,size(mesh.dgnodes,3));
nm = 30;
mesh.vdg(:,1,:) = 0.1*tanh(dist*nm);

% intial solution
ui = [rinf ruinf rvinf rEinf];
UDG = initu(mesh,{ui(1),ui(2),ui(3),ui(4)}); % freestream 
UDG(:,2,:) = UDG(:,2,:).*tanh(nm*dist);
UDG(:,3,:) = UDG(:,3,:).*tanh(nm*dist);
TnearWall = Tinf * (Twall/Tref-1) * exp(-nm*dist) + Tinf;
UDG(:,4,:) = TnearWall + 0.5*(UDG(:,2,:).*UDG(:,2,:) + UDG(:,3,:).*UDG(:,3,:));
mesh.udg = UDG;

[sol,pde,mesh,master] = exasim(pde,mesh);
figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[],1); colormap('jet'); colorbar;

disp("Iter 2")
Re = 500;
pde.physicsparam = [gam Re Pr Minf rinf ruinf rvinf rEinf Tinf Tref Twall mu_inf mu_ref Tmu_ref omega R sigmaV sigmaT];
mesh.vdg(:,1,:) = 0.01.*tanh(dist*nm);
mesh.udg = sol;
[pde,mesh,master,dmd] = preprocessing(pde,mesh);
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
figure(2); clf; scaplot(mesh, eulereval(sol, 't',gam,Minf),[],2); colormap('jet'); colorbar;

disp("Iter 5")
mesh.vdg(:,1,:) = 0.002.*tanh(dist*nm);
mesh.udg = sol;
[pde,mesh,master,dmd] = preprocessing(pde,mesh);
runcode(pde, 1); % run C++ code
sol = fetchsolution(pde,master,dmd, pde.datapath + "/dataout" + model_strn(pde));
figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[],2); colormap('jet'); colorbar;
figure(2); clf; scaplot(mesh, eulereval(sol, 't',gam,Minf),[],2); colormap('jet'); colorbar;

disp("Iter 6")
mesh.vdg(:,1,:) = 0.0015.*tanh(dist*nm);
mesh.udg = sol;
[pde,mesh,master,dmd] = preprocessing(pde,mesh);
runcode(pde, 1); % run C++ code
sol = fetchsolution(pde,master,dmd, pde.datapath + "/dataout" + model_strn(pde));
figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[],2);colormap('jet'); colorbar;
figure(2); clf; scaplot(mesh, T_scale*eulereval(sol, 't',gam,Minf),[],2); colormap('jet'); colorbar;
