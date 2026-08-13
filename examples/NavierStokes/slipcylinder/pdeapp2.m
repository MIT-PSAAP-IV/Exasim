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

mesh = mkmesh_cyl(pde.porder, D/2, 1, 1.2, 5);
mesh.boundarycondition = [4;3;2]; 

R = 287.05;
gam = 5/3;                     % specific heat ratio
Re = 956;                     % Reynolds number
Pr = 2/3;                      % Prandtl number    
Minf = 10;                     % Mach number
Tref  = 200;
rho_ref = 5.636e-6;            % physical freestream density [kg/m^3]
Twall = 500;
pinf = 1/(gam*Minf^2);
Tinf = pinf/(gam-1);
alpha = 0;                % angle of attack
rinf = 1.0;                     % freestream density
ruinf = cos(alpha);             % freestream horizontal velocity
rvinf = sin(alpha);             % freestream vertical velocity
pinf = 1/(gam*Minf^2);          % freestream pressure
rEinf = 0.5+pinf/(gam-1);       % freestream energy
T_scale = Tref/Tinf;

R = 208.13;
omega = 0.7340; 
mu_ref = 5.0711e-05;
Tmu_ref = 1000;
mu_inf  = mu_ref * (Tref  / Tmu_ref)^omega;    % freestream dynamic viscosity [Pa s]
sigmaV = 0;
sigmaT = 0;

pde.gencode = 1;
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
nm = 50;
mesh.vdg(:,1,:) = 0.1*tanh(dist*nm);
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

[sol,pde,mesh,master] = exasim(pde,mesh);
figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[],1); colormap('jet'); colorbar;

disp("Iter 2")
pde.physicsparam = [gam Re Pr Minf rinf ruinf rvinf rEinf Tinf Tref Twall mu_inf mu_ref Tmu_ref omega R sigmaV sigmaT rho_ref];
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
mesh.vdg(:,1,:) = 0.0014.*tanh(dist*nm);
mesh.udg = sol;
[pde,mesh,master,dmd] = preprocessing(pde,mesh);
runcode(pde, 1); % run C++ code
sol = fetchsolution(pde,master,dmd, pde.datapath + "/dataout" + model_strn(pde));
figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[],2); colormap('jet'); colorbar;
figure(2); clf; scaplot(mesh, eulereval(sol, 't',gam,Minf),[],2); colormap('jet'); colorbar;

disp("Iter 6")
mesh.vdg(:,1,:) = 0.001.*tanh(dist*nm);
mesh.udg = sol;
[pde,mesh,master,dmd] = preprocessing(pde,mesh);
runcode(pde, 1); % run C++ code
sol = fetchsolution(pde,master,dmd, pde.datapath + "/dataout" + model_strn(pde));
figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[],2);colormap('jet'); colorbar;
figure(2); clf; scaplot(mesh, T_scale*eulereval(sol, 't',gam,Minf),[],2); colormap('jet'); colorbar;

disp("Iter 7")
mesh.vdg(:,1,:) = 0.0006.*tanh(dist*nm);
mesh.udg = sol;
[pde,mesh,master,dmd] = preprocessing(pde,mesh);
runcode(pde, 1); % run C++ code
sol = fetchsolution(pde,master,dmd, pde.datapath + "/dataout" + model_strn(pde));
figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[],2);colormap('jet'); colorbar;
figure(2); clf; scaplot(mesh, T_scale*eulereval(sol, 't',gam,Minf),[],2); colormap('jet'); colorbar;

Tphys = T_scale*eulereval(sol, 't',gam,Minf);
X = mesh.dgnodes(:,1,:);
Y = mesh.dgnodes(:,2,:);
ind = find(abs(Y(:)) < 1e-6 & X(:) < 0);
X0 = X(ind); Y0 = Y(ind);
[X0, ii] = sort(X0);
Y0 = Y0(ii);
T0 = Tphys(ind);
T0 = T0(ii);
figure(3); clf; plot(2*X0/D+1, T0);

% uqw = zeros(12,1);
% for i = 1:12
%   tm = sol(:,i,:);
%   tn = tm(ind);
%   tn = tn(ii);  
%   uqw(i) = tn(end);
% end
% [dutdn, dTdn, lambda] = wallstate(uqw(1:4), uqw(5:12), pde.physicsparam, [1 0]);
% [lambda*dTdn Twall/Tref * Tinf]

sol0 = sol;

disp("Iter 8")
sigmaV = 0.2;
sigmaT = 0.2;
pde.physicsparam = [gam Re Pr Minf rinf ruinf rvinf rEinf Tinf Tref Twall mu_inf mu_ref Tmu_ref omega R sigmaV sigmaT rho_ref];
mesh.vdg(:,1,:) = 0.0006.*tanh(dist*nm);
mesh.udg = sol;
[pde,mesh,master,dmd] = preprocessing(pde,mesh);
runcode(pde, 1); % run C++ code
sol = fetchsolution(pde,master,dmd, pde.datapath + "/dataout" + model_strn(pde));
figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[],2);colormap('jet'); colorbar;
figure(2); clf; scaplot(mesh, T_scale*eulereval(sol, 't',gam,Minf),[],2); colormap('jet'); colorbar;

disp("Iter 9")
sigmaV = 0.5;
sigmaT = 0.5;
pde.physicsparam = [gam Re Pr Minf rinf ruinf rvinf rEinf Tinf Tref Twall mu_inf mu_ref Tmu_ref omega R sigmaV sigmaT rho_ref];
mesh.vdg(:,1,:) = 0.0006.*tanh(dist*nm);
mesh.udg = sol;
[pde,mesh,master,dmd] = preprocessing(pde,mesh);
runcode(pde, 1); % run C++ code
sol = fetchsolution(pde,master,dmd, pde.datapath + "/dataout" + model_strn(pde));
figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[],2);colormap('jet'); colorbar;
figure(2); clf; scaplot(mesh, T_scale*eulereval(sol, 't',gam,Minf),[],2); colormap('jet'); colorbar;

disp("Iter 10")
sigmaV = 1.0;
sigmaT = 1.875;
pde.physicsparam = [gam Re Pr Minf rinf ruinf rvinf rEinf Tinf Tref Twall mu_inf mu_ref Tmu_ref omega R sigmaV sigmaT rho_ref];
mesh.vdg(:,1,:) = 0.0006.*tanh(dist*nm);
mesh.udg = sol;
[pde,mesh,master,dmd] = preprocessing(pde,mesh);
runcode(pde, 1); % run C++ code
sol = fetchsolution(pde,master,dmd, pde.datapath + "/dataout" + model_strn(pde));
figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[],2);colormap('jet'); colorbar;
figure(2); clf; scaplot(mesh, T_scale*eulereval(sol, 't',gam,Minf),[],2); colormap('jet'); colorbar;

Tphys = T_scale*eulereval(sol, 't',gam,Minf);
T1 = Tphys(ind);
T1 = T1(ii);
figure(3); clf; plot(2*X0/D+1, T0, '-', 2*X0/D+1, T1, '--', 'LineWidth',2);
set(gca,'FontSize',20); axis tight;
xlabel('$x/R$','Interpreter','latex'); 
ylabel('$\mbox{Temperature (K)}$','Interpreter','latex'); 
title('Kn = 0.25');
legend("No-slip", "slip", "Location", "NorthWest");
axis([-2 0 0 6500]);

