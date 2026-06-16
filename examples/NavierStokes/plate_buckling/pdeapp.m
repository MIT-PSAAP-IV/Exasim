% Add Exasim to Matlab search path
cdir = pwd(); ii = strfind(cdir, "Exasim");
run(cdir(1:(ii+5)) + "/install/setpath.m");

% initialize pde structure and mesh structure
[pde,mesh] = initializeexasim();
pde.model = "ModelD";  
pde.modelfile = "pdemodel_ns";

% Choose computing platform and set number of processors
pde.platform = "cpu";         % choose this option if NVIDIA GPUs are available
pde.mpiprocs = 15;             % number of MPI processors
pde.porder = 2;          % polynomial degree
pde.pgauss = 2*pde.porder;
pde.hybrid = 1;               % 0 -> LDG, 1 -> HDG
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

% Mesh size: nx1 (nose cap), nxf (plate/tail), ny (across)
nx1 = 36; nxf = 120; ny = 100;

% Wall bump parameters (Gaussian: amp * exp(-((x-loc)/width)^2))
bump_amp = 1.2e-2;
bump_loc = 0.20;
bump_width = 0.05;

% Generate smooth mesh (no geometry bump — deformation via elasticity)
mesh = mkmesh_thermal_buckling(pde.porder, nx1, nxf, ny, 0);
figure(1);clf;meshplot(mesh);

% --- Linear elasticity mesh deformation ---
if abs(bump_amp) > 0
    E = 1.0;
    nu = 0.30;
    mu_lam = E / (2 * (1 + nu));
    lambda = nu * E / ((1 + nu) * (1 - 2 * nu));

    pde_el = pde;
    pde_el.model = "ModelD";
    pde_el.modelfile = "pdemodel_elastic";
    pde_el.physicsparam = [mu_lam, lambda, bump_amp, bump_loc, bump_width];
    pde_el.tau = 2*(mu_lam + lambda);
    pde_el.linearsolvertol = 1e-8;
    pde_el.GMRESrestart = 100;
    pde_el.linearsolveriter = 200;
    pde_el.NLtol = 1e-8;
    pde_el.NLiter = 1;    % linear PDE, 1 Newton step
    pde_el.gencode = 1;

    mesh_el = mesh;
    mesh_el.boundarycondition = [1, 1, 2, 1];  % fixed, fixed, wall_bump, fixed

    master_el = Master(pde_el);
    [sol_el, ~, ~, ~, ~] = exasim(pde_el, mesh_el);

    % Deform the flow mesh
    mesh.dgnodes = mesh.dgnodes - sol_el(:, 1:2, :);
    figure(1);clf;meshplot(mesh);

    pde.gencode = 1;  % force recompile for NS model
end
% --- End elasticity deformation ---

master = Master(pde);

% initial artificial viscosity
dist = meshdist3(mesh.f,mesh.dgnodes,master.perm,[3]); % distance to the wall (group 3)
mesh.vdg = zeros(size(mesh.dgnodes,1),1,size(mesh.dgnodes,3));
nm = 1e2;
mesh.vdg(:,1,:) = 0.005*tanh(nm*dist);

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
[sol,pde,mesh,master,dmd] = exasim(pde,mesh);
figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[0 Minf],1); colorbar;
%%
disp("Iter 2")
mesh.vdg(:,1,:) = 0.004*tanh(nm*dist);
mesh.udg = sol;
preprocessing(pde,mesh);
runcode(pde, 1); % run C++ code
sol = fetchsolution(pde,master,dmd, pde.datapath+'/dataout');
figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[0 Minf],1); colorbar;

disp("Iter 3")
mesh.vdg(:,1,:) = 0.003*tanh(nm*dist);
mesh.udg = sol;
preprocessing(pde,mesh);
runcode(pde, 1); % run C++ code
sol = fetchsolution(pde,master,dmd, pde.datapath + '/dataout');
figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[0 Minf],1); colorbar;

disp("Iter 4")
mesh.vdg(:,1,:) = 0.0025*tanh(nm*dist);
mesh.udg = sol;
preprocessing(pde,mesh);
runcode(pde, 1); % run C++ code
sol = fetchsolution(pde,master,dmd, pde.datapath + '/dataout');
figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[0 Minf],1); colorbar;

disp("Iter 5")
mesh.vdg(:,1,:) = 0.0024*tanh(nm*dist);
mesh.udg = sol;
preprocessing(pde,mesh);
runcode(pde, 1); % run C++ code
sol = fetchsolution(pde,master,dmd, pde.datapath + '/dataout');
figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[0 Minf],1); colorbar;

disp("Iter 6")
mesh.vdg(:,1,:) = 0.0002*tanh(nm*dist);
mesh.udg = sol;
preprocessing(pde,mesh);
runcode(pde, 1); % run C++ code
sol = fetchsolution(pde,master,dmd, pde.buildpath + '/dataout');
figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[0 Minf],1); colorbar;

% disp("Iter 7")
% mesh.vdg(:,1,:) = 0.00014*tanh(nm*dist);
% mesh.udg = sol;
% preprocessing(pde,mesh);
% runcode(pde, 1); % run C++ code
% sol = fetchsolution(pde,master,dmd, pde.buildpath + '/dataout');
% figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[0 Minf],1); colorbar;
% 
% disp("Iter 8")
% divmax = 8; pdeapp_hm;
% mesh.vdg(:,1,:) = 0.0003*av;
% mesh.udg = sol;
% preprocessing(pde,mesh);
% runcode(pde, 1); % run C++ code
% sol = fetchsolution(pde,master,dmd, pde.buildpath + '/dataout');
% figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[0 Minf],1); colorbar;
% figure(2); clf; scaplot(mesh, mesh.vdg(:,1,:),[],2); colorbar;
% 
% disp("Iter 9")
% divmax = 12; pdeapp_hm;
% mesh.vdg(:,1,:) = 0.0003*av;
% mesh.udg = sol;
% preprocessing(pde,mesh);
% runcode(pde, 1); % run C++ code
% sol = fetchsolution(pde,master,dmd, pde.buildpath + '/dataout');
% figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[0 Minf],1); colorbar;
% figure(2); clf; scaplot(mesh, mesh.vdg(:,1,:),[],2); colorbar;

mesh1 = mesh;
mesh1.dgnodes(:,2,:) = -mesh.dgnodes(:,2,:);
Tphys = Tref/Tinf * eulereval(sol, 't',gam,Minf);
figure(3); clf; scaplot(mesh, Tphys,[],1); 
hold on;
scaplot(mesh1, Tphys,[],1); 
colorbar; colormap('jet'); axis on; axis equal; axis tight; set(gca,'FontSize',16);
title("Temperature");
exportgraphics(gca,"temperature.png",'Resolution',200);

R = 340.8;
Tphys = Tref/Tinf * eulereval(sol, 't',gam,Minf);
rhophys = sol(:,1,:)*2.35e-3;
Pphys = rhophys.*Tphys*R;
figure(3); clf; scaplot(mesh, Pphys,[],1); 
hold on;
scaplot(mesh1, Pphys,[],1); 
colorbar; colormap('jet'); axis on; axis equal; axis tight; set(gca,'FontSize',16);
title("Pressure");
exportgraphics(gca,"pressure.png",'Resolution',200);

figure(3); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[0 Minf],1);
hold on;
scaplot(mesh1, eulereval(sol, 'M',gam,Minf),[0 Minf],1);
colorbar; colormap('jet'); axis on; axis equal; axis tight; set(gca,'FontSize',16);
title("Mach");
exportgraphics(gca,"mach.png",'Resolution',200);


% disp("Iter 7")
% divmax = 100; pdeapp_hm;
% mesh.vdg(:,1,:) = 0.00025*av;
% mesh.udg = sol;
% preprocessing(pde,mesh);
% runcode(pde, 1); % run C++ code
% sol = fetchsolution(pde,master,dmd, pde.buildpath + '/dataout');
% figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[0 Minf],1); colorbar;
% figure(2); clf; scaplot(mesh, mesh.vdg(:,1,:),[],2); colorbar;
