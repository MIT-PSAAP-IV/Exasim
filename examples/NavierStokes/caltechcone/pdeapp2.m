% Add Exasim to Matlab search path
cdir = pwd(); ii = strfind(cdir, "Exasim");
run(cdir(1:(ii+5)) + "/install/setpath.m");

% initialize pde structure and mesh structure
[pde,mesh] = initializeexasim();

% pde.buildpath = "/data/data1/GitHub/Exasim/build" + "/nonreacting";
% if exist(pde.buildpath, 'dir') == 0
%     mkdir(pde.buildpath);
% end


pde.model = "ModelD";  
pde.modelfile = "pdemodel_axialns";

% Choose computing platform and set number of processors
pde.platform = "cpu";         % choose this option if NVIDIA GPUs are available
pde.mpiprocs = 8;             % number of MPI processors
pde.porder = 2;          % polynomial degree
pde.pgauss = 2*pde.porder;
pde.hybrid = 1;               % 0 -> LDG, 1 -> HDG
pde.debugmode = 0;
pde.nd = 2;

gam = 1.4;                      % specific heat ratio
Re = 6.35e6;%9.84e5;                    % Reynolds number
Pr = 0.71;                      % Prandtl number    
Minf = 5.13;%21.38;                   % Mach number
Tref  = 1355;%260.6;
Twall = 243.9;
pinf = 1/(gam*Minf^2);
Tinf = pinf/(gam-1);
alpha = 0;                % angle of attack
rinf = 1.0;                     % freestream density
ruinf = cos(alpha);             % freestream horizontal velocity
rvinf = sin(alpha);             % freestream vertical velocity
pinf = 1/(gam*Minf^2);          % freestream pressure
rEinf = 0.5+pinf/(gam-1);       % freestream energy

pde.physicsparam = [gam Re Pr Minf rinf ruinf rvinf rEinf Tinf Tref Twall];
pde.tau = 10.0;%4.0;                  % DG stabilization parameter
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

mesh = mkmesh_cone2d2(pde.porder);
mesh.boundarycondition = [5 1 3 3 2 1];
% mesh = mkmesh_sharpb2(pde.porder);
% mesh.boundarycondition = [5 1 1 3 2];%mesh.boundarycondition = [5 1 3 3 2 1]; %symmetry, inflow, wall, wall, outflow, top
figure(1);clf;meshplot(mesh);

master = Master(pde);

% initial artificial viscosity
dist = meshdist3(mesh.f,mesh.dgnodes,master.perm,[3 4]); % distance to the wall
mesh.vdg = zeros(size(mesh.dgnodes,1),1,size(mesh.dgnodes,3));
nm = 1e3;
mesh.vdg(:,1,:) = 0.002*tanh(nm*dist);

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

pde.gencode = 1;
[sol,pde,mesh,master,dmd] = exasim(pde,mesh);
figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[],1,1); colorbar; colormap('jet')

nm = 200;
disp("Iter 2")
mesh.vdg(:,1,:) = 0.0004*tanh(nm*dist);
mesh.udg = sol;
preprocessing(pde,mesh);
runcode(pde, 1); % run C++ code
sol = fetchsolution(pde,master,dmd, pde.buildpath + '/dataout');
figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[],1,1); colorbar; colormap('jet')

nm = 100;
disp("Iter 3")
mesh.vdg(:,1,:) = 0.0003*tanh(nm*dist);
mesh.udg = sol;
preprocessing(pde,mesh);
runcode(pde, 1); % run C++ code
sol = fetchsolution(pde,master,dmd, pde.buildpath + '/dataout');
figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[],1,1); colorbar; colormap('jet')

nm = 100;
disp("Iter 4")
mesh.vdg(:,1,:) = 0.00025*tanh(nm*dist);
mesh.udg = sol;
preprocessing(pde,mesh);
runcode(pde, 1); % run C++ code
sol = fetchsolution(pde,master,dmd, pde.buildpath + '/dataout');
figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[],1); colorbar; colormap('jet')

disp("Iter 5")
divmax = 6; pdeapp_hm2;
mesh.vdg(:,1,:) = 0.0005.*av;
mesh.udg = sol;
pde.dt =[];
preprocessing(pde,mesh);
runcode(pde, 1); % run C++ code
sol = fetchsolution(pde,master,dmd, pde.buildpath + '/dataout');
figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[],1); colorbar; colormap('jet')
figure(2); clf; scaplot(mesh, mesh.vdg(:,1,:),[],1); colorbar;

disp("Iter 6")
mesh.vdg(:,1,:) = 0.0003.*av;
mesh.udg = sol;
pde.dt =[];
preprocessing(pde,mesh);
runcode(pde, 1); % run C++ code
sol = fetchsolution(pde,master,dmd, pde.buildpath + '/dataout');
figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[],1); colorbar; colormap('jet')
figure(2); clf; scaplot(mesh, mesh.vdg(:,1,:),[],1); colorbar;

disp("Iter 7")
pdeapp_hm2;
mesh.vdg(:,1,:) = 0.0002.*av;
mesh.udg = sol;
pde.dt =[];
preprocessing(pde,mesh);
runcode(pde, 1); % run C++ code
sol = fetchsolution(pde,master,dmd, pde.buildpath + '/dataout');
figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[],1); colorbar; colormap('jet')
figure(2); clf; scaplot(mesh, mesh.vdg(:,1,:),[],1); colorbar;

disp("Iter 8")
mesh.vdg(:,1,:) = 0.00012.*av;
mesh.udg = sol;
pde.dt =[];
preprocessing(pde,mesh);
runcode(pde, 1); % run C++ code
sol = fetchsolution(pde,master,dmd, pde.buildpath + '/dataout');
figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[],1); colorbar; colormap('jet')
figure(2); clf; scaplot(mesh, mesh.vdg(:,1,:),[],1); colorbar;

