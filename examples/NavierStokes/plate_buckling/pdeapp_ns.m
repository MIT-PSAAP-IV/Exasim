% Put the Exasim MATLAB frontend on the path. For an installed Exasim use
% run('<prefix>/share/exasim/matlab/exasim_setup.m') instead.
run(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', 'frontends', 'Matlab', 'exasim_setup.m'));

% initialize pde structure and mesh structure
[pde,~] = initializeexasim();

% Define a PDE model: governing equations, initial solutions, and boundary conditions
pde.model = "ModelD";          % ModelC, ModelD, ModelW
pde.modelfile = "pdemodel_ns";    % name of a file defining the PDE model

% Choose computing platform and set number of processors
pde.platform = "cpu";         % choose this option if NVIDIA GPUs are available
pde.mpiprocs = 16;              % number of MPI processors
pde.hybrid = 1;
pde.porder = 2;          % polynomial degree (must match mesh)
pde.pgauss = 2*pde.porder;
pde.debugmode = 0;
pde.nd = 2;

gam = 1.451;                      % specific heat ratio
Re = 9.84e5;                    % Reynolds number
Pr = 0.72;                      % Prandtl number
Minf = 7.7;                   % Mach number
Tref  = 477;
Twall = 300;
pinf = 1/(gam*Minf^2);
Tinf = pinf/(gam-1);
alpha = 0;                % angle of attack
rinf = 1.0;                     % freestream density
ruinf = cos(alpha);             % freestream horizontal velocity
rvinf = sin(alpha);             % freestream vertical velocity
pinf = 1/(gam*Minf^2);          % freestream pressure
rEinf = 0.5+pinf/(gam-1);       % freestream energy

pde.physicsparam = [gam Re Pr Minf rinf ruinf rvinf rEinf Tinf Tref Twall];
pde.tau = 4.0;                  % DG stabilization parameter
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

master = Master(pde);

% initial artificial viscosity
dist = meshdist3(mesh.f,mesh.dgnodes,master.perm,[3]); % distance to the wall
nm = 1e2;
mesh.vdg = zeros(size(mesh.dgnodes,1),1,size(mesh.dgnodes,3));
mesh.vdg(:,1,:) = 0.005*tanh(nm*dist);
mesh.dist = dist; % save

mesh.porder = pde.porder;
mesh.xpe = master.xpe;
mesh.telem = master.telem;
figure(2); clf; scaplot(mesh,mesh.vdg(:,1,:),[],1,0); axis on; axis equal; axis tight;

% intial solution
ui = [rinf ruinf rvinf rEinf];
UDG = initu(mesh,{ui(1),ui(2),ui(3),ui(4),0,0,0,0,0,0,0,0}); % freestream
UDG(:,2,:) = UDG(:,2,:).*tanh(nm*dist);
UDG(:,3,:) = UDG(:,3,:).*tanh(nm*dist);
TnearWall = Tinf * (Twall/Tref-1) * exp(-nm*dist) + Tinf;
UDG(:,4,:) = TnearWall + 0.5*(UDG(:,2,:).*UDG(:,2,:) + UDG(:,3,:).*UDG(:,3,:));
mesh.udg = UDG;

figure(3); clf; scaplot(mesh,TnearWall,[],1); axis on; axis equal; axis tight;

%%
pde.gencode = 1;
[sol,pde,mesh,master] = exasim(pde,mesh);
figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[0 Minf],1); colorbar;

%%
disp("Iter 2")
mesh.vdg(:,1,:) = 0.004*tanh(nm*dist);
mesh.udg = sol;
[pde,mesh,master,dmd] = preprocessing(pde,mesh);
runcode(pde, 1); % run C++ code
sol = fetchsolution(pde,master,dmd, fullfile(pde.datapath, 'dataout'));
figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[0 Minf],1); colorbar;

% disp("Iter 3")
% mesh.vdg(:,1,:) = 0.003*tanh(nm*dist);
% mesh.udg = sol;
% [pde,mesh,master,dmd] = preprocessing(pde,mesh);
% runcode(pde, 1); % run C++ code
% sol = fetchsolution(pde,master,dmd, fullfile(pde.datapath, 'dataout'));
% figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[0 Minf],1); colorbar;
% 
% disp("Iter 4")
% mesh.vdg(:,1,:) = 0.0025*tanh(nm*dist);
% mesh.udg = sol;
% [pde,mesh,master,dmd] = preprocessing(pde,mesh);
% runcode(pde, 1); % run C++ code
% sol = fetchsolution(pde,master,dmd, fullfile(pde.datapath, 'dataout'));
% figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[0 Minf],1); colorbar;
