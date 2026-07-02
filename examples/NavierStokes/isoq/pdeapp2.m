% Put the Exasim MATLAB frontend on the path. For an installed Exasim use
% run('<prefix>/share/exasim/matlab/exasim_setup.m') instead.
run(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', 'frontends', 'Matlab', 'exasim_setup.m'));


% initialize pde structure and mesh structure
[pde,mesh] = initializeexasim();
pde.model = "ModelD";  
pde.modelfile = "pdemodel_axialns";

% Choose computing platform and set number of processors
pde.platform = "cpu";         % choose this option if NVIDIA GPUs are available
pde.mpiprocs = 4;             % number of MPI processors
pde.porder = 2;          % polynomial degree
pde.pgauss = 2*pde.porder;
pde.hybrid = 1;               % 0 -> LDG, 1 -> HDG
pde.debugmode = 0;
pde.nd = 2;

gam = 1.4;                      % specific heat ratio
Re = 1.835e5;                     % Reynolds number
Pr = 0.71;                      % Prandtl number    
Minf = 7;                       % Mach number
Tref  = 124.49;
Twall = 294.44;
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
pde.NLiter = 4;                % Newton iterations
pde.matvectol=1e-6;             % tolerance for matrix-vector multiplication

load adaptivemesh.mat
mesh.boundarycondition = [5 2 1 3]; % symmetry, outflow, inflow, wall
master = Master(pde);
dist = meshdist3(mesh.f,mesh.dgnodes,master.perm,[4]); % distance to the wall
mesh.udg = sol;

pde.gencode = 1;
[sol,pde,mesh,master,dmd] = exasim(pde,mesh);
figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[0 Minf],1); colorbar;
figure(2); clf; scaplot(mesh, mesh.vdg,[],1); colorbar;

disp("Iter 2")
divmax = 100; pdeapp_hm2;
mesh.vdg(:,1,:) = 0.0002*av;
mesh.udg = sol;
preprocessing(pde,mesh);
runcode(pde, 1); % run C++ code
sol = fetchsolution(pde,master,dmd, pwd() + "/dataout");
figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[0 Minf],1); colorbar;
figure(2); clf; scaplot(mesh, mesh.vdg(:,1,:),[],2); colorbar;

disp("Iter 3")
divmax = 100; pdeapp_hm2;
mesh.vdg(:,1,:) = 0.00015*av;
mesh.udg = sol;
preprocessing(pde,mesh);
runcode(pde, 1); % run C++ code
sol = fetchsolution(pde,master,dmd, pwd() + "/dataout");
figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[0 Minf],1); colorbar;
figure(2); clf; scaplot(mesh, mesh.vdg(:,1,:),[],2); colorbar;

% disp("Iter 2")
% nm = 3e2;
% mesh.vdg(:,1,:) = 0.0006*tanh(nm*dist);
% mesh.udg = sol;
% preprocessing(pde,mesh);
% runcode(pde, 1); % run C++ code
% sol = fetchsolution(pde,master,dmd, pwd() + "/dataout");
% figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[0 Minf],1); colorbar;
% 
% disp("Iter 3")
% nm = 2e2;
% mesh.vdg(:,1,:) = 0.0004*tanh(nm*dist);
% mesh.udg = sol;
% preprocessing(pde,mesh);
% runcode(pde, 1); % run C++ code
% sol = fetchsolution(pde,master,dmd, pwd() + "/dataout");
% figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[0 Minf],1); colorbar;
% 
% disp("Iter 4")
% nm = 0.5e2;
% mesh.vdg(:,1,:) = 0.0004*tanh(nm*dist);
% mesh.udg = sol;
% preprocessing(pde,mesh);
% runcode(pde, 1); % run C++ code
% sol = fetchsolution(pde,master,dmd, pwd() + "/dataout");
% figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[0 Minf],1); colorbar;
% figure(2); clf; scaplot(mesh, mesh.vdg,[],1); colorbar;
% 
% disp("Iter 5")
% divmax = 20; pdeapp_hm;
% mesh.vdg(:,1,:) = 0.0004*av;
% mesh.udg = sol;
% preprocessing(pde,mesh);
% runcode(pde, 1); % run C++ code
% sol = fetchsolution(pde,master,dmd, pwd() + "/dataout");
% figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[0 Minf],1); colorbar;
% figure(2); clf; scaplot(mesh, mesh.vdg(:,1,:),[],2); colorbar;
% 
% disp("Iter 6")
% divmax = 100; pdeapp_hm;
% mesh.vdg(:,1,:) = 0.00035*av;
% mesh.udg = sol;
% preprocessing(pde,mesh);
% runcode(pde, 1); % run C++ code
% sol = fetchsolution(pde,master,dmd, pwd() + "/dataout");
% figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[0 Minf],1); colorbar;
% figure(2); clf; scaplot(mesh, mesh.vdg(:,1,:),[],2); colorbar;
% 
% disp("Iter 7")
% divmax = 100; pdeapp_hm;
% mesh.vdg(:,1,:) = 0.0003*av;
% mesh.udg = sol;
% preprocessing(pde,mesh);
% runcode(pde, 1); % run C++ code
% sol = fetchsolution(pde,master,dmd, pwd() + "/dataout");
% figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[0 Minf],1); colorbar;
% figure(2); clf; scaplot(mesh, mesh.vdg(:,1,:),[],2); colorbar;
% 
% disp("Iter 8")
% divmax = 100; pdeapp_hm;
% mesh.vdg(:,1,:) = 0.00025*av;
% mesh.udg = sol;
% preprocessing(pde,mesh);
% runcode(pde, 1); % run C++ code
% sol = fetchsolution(pde,master,dmd, pwd() + "/dataout");
% figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[0 Minf],1); colorbar;
% figure(2); clf; scaplot(mesh, mesh.vdg(:,1,:),[],2); colorbar;
% 
% disp("Iter 9")
% divmax = 100; pdeapp_hm;
% mesh.vdg(:,1,:) = 0.00023*av;
% mesh.udg = sol;
% preprocessing(pde,mesh);
% runcode(pde, 1); % run C++ code
% sol = fetchsolution(pde,master,dmd, pwd() + "/dataout");
% figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[0 Minf],1); colorbar;
% figure(2); clf; scaplot(mesh, mesh.vdg(:,1,:),[],2); colorbar;
% 
% % disp("Iter 7")
% % divmax = 100; pdeapp_hm;
% % mesh.vdg(:,1,:) = 0.00025*av;
% % mesh.udg = sol;
% % preprocessing(pde,mesh);
% % runcode(pde, 1); % run C++ code
% % sol = fetchsolution(pde,master,dmd, pwd() + "/dataout");
% % figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[0 Minf],1); colorbar;
% % figure(2); clf; scaplot(mesh, mesh.vdg(:,1,:),[],2); colorbar;
% % 
% % figure(3); clf; scaplot(mesh, sol(:,2,:),[],2); colorbar;
