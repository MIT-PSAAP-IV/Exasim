% Put the Exasim MATLAB frontend on the path. For an installed Exasim use
% run('<prefix>/share/exasim/matlab/exasim_setup.m') instead.
run(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', 'frontends', 'Matlab', 'exasim_setup.m'));

load('solp2coarse.mat');
mesh2d = mesh; udg2d = sol(:,1:4,:); vdg2d = mesh.vdg;

% initialize pde structure and mesh structure
[pde,~] = initializeexasim();
pde.model = "ModelD";  
pde.modelfile = "pdemodel";

% Choose computing platform and set number of processors
pde.platform = "cpu";         % choose this option if NVIDIA GPUs are available
pde.mpiprocs = 8;             % number of MPI processors
pde.porder = 2;          % polynomial degree
pde.pgauss = 2*pde.porder;
pde.hybrid = 1;               % 0 -> LDG, 1 -> HDG
pde.debugmode = 0;
pde.nd = 3;

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
rwinf = 0;
pinf = 1/(gam*Minf^2);          % freestream pressure
rEinf = 0.5+pinf/(gam-1);       % freestream energy

pde.physicsparam = [gam Re Pr Minf rinf ruinf rvinf rwinf rEinf Tinf Tref Twall];
pde.tau = 10.0;                  % DG stabilization parameter
pde.GMRESrestart = 200;         %try 50
pde.GMRESortho = 1;
pde.linearsolvertol = 1e-6; % GMRES tolerance
pde.linearsolveriter = 200; %try 100
pde.preconditioner = 1;
pde.RBdim = 0;
pde.ppdegree = 0;
pde.NLtol = 1e-6;              % Newton tolerance
pde.NLiter = 10;                % Newton iterations
pde.matvectol=1e-6;             % tolerance for matrix-vector multiplication

pde.dt = 1e-4*(1.2.^(0:7));
pde.nstage = 1;
pde.torder = 1;
pde.saveSolFreq = 4;

nz = 4;
mesh = mkmesh_isoq3d3(mesh2d, nz);
% symmetry, symmetry, outflow, symmetry, wall, inflow
mesh.boundarycondition = [4, 4, 2, 4, 3, 1]; % Set boundary condition for each boundary

mesh.vdg = extrudesol(vdg2d, pde.porder, nz);

tt = linspace(0, pi/2, nz+1);
[vx3d, vy3d] = extrudevelocity(udg2d(:,3,:), pde.porder, tt);
udg3d = extrudesol(udg2d, pde.porder, nz);
mesh.udg = udg3d(:,1:2,:);
mesh.udg(:,3,:) = vx3d;
mesh.udg(:,4,:) = vy3d;
mesh.udg(:,5,:) = udg3d(:,4,:);

pde.gencode = 1;
[sol,pde,mesh,master,dmd] = exasim(pde,mesh);

visfield = sol(:,1:5,:,end);
pres = eulereval3d(visfield,'p',1.4,Minf);
mach = eulereval3d(visfield,'M',1.4,Minf);
visfield(:,2:4,:) = visfield(:,2:4,:)./visfield(:,1,:);
visfield(:,5,:) = pres;
visfield(:,6,:) = mach;
pde.visscalars = {"density", 1, "pressure", 5, "mach", 6};                % list of scalar fields for visualization
pde.visvectors = {"velocity", [2 3 4]}; % list of vector fields for visualization
pde.paraview = "/Applications/ParaView-6.0.0.app/Contents/MacOS/paraview";
vis(visfield,pde,mesh);                        % visualize the numerical solution

% visfield = mesh.udg;
% pres = eulereval3d(visfield,'p',1.4,Minf);
% mach = eulereval3d(visfield,'M',1.4,Minf);
% visfield(:,2:4,:) = visfield(:,2:4,:)./visfield(:,1,:);
% visfield(:,5,:) = pres;
% visfield(:,6,:) = mach;
% pde.visscalars = {"density", 1, "pressure", 5, "mach", 6};                % list of scalar fields for visualization
% pde.visvectors = {"velocity", [2 3 4]}; % list of vector fields for visualization
% pde.paraview = "/Applications/ParaView-6.0.0.app/Contents/MacOS/paraview";
% vis(visfield,pde,mesh);                        % visualize the numerical solution


% % master = Master(pde);
% % initial artificial viscosity
% %dist = meshdist3(mesh.f,mesh.dgnodes,master.perm,5); % distance to the wall
% dist = mesh.dist;
% mesh.vdg = zeros(size(mesh.dgnodes,1),1,size(mesh.dgnodes,3));
% nm = 200;
% mesh.vdg(:,1,:) = 0.001*tanh(nm*dist);
% 
% % intial solution
% ui = [rinf ruinf rvinf rwinf rEinf];
% UDG = initu(mesh,{ui(1),ui(2),ui(3),ui(4),ui(5)}); % freestream 
% UDG(:,2,:) = UDG(:,2,:).*tanh(nm*dist);
% TnearWall = Tinf * (Twall/Tref-1) * exp(-nm*dist) + Tinf;
% UDG(:,5,:) = TnearWall + 0.5*(UDG(:,2,:).*UDG(:,2,:) + UDG(:,3,:).*UDG(:,3,:) + UDG(:,4,:).*UDG(:,4,:));
% mesh.udg = UDG;
% 
% [pde,mesh,master,dmd] = preprocessing(pde,mesh);

% kkgencode(pde);
% compilerstr = cmakecompile(pde); % use cmake to compile C++ source codes 
% runcode(pde, 1); % run C++ code
% sol = fetchsolution(pde,master,dmd, pde.buildpath + '/dataout');

% mesh.xpe = master.xpe;
% pde.visscalars = {"av", 1};
% pde.visvectors = {};
% vis(mesh.vdg,pde,mesh);    

% pde.visscalars = {"density", 1, "vx", 2, "vy", 3, "vz", 4, "energy", 5};
% pde.visvectors = {};
% vis(mesh.udg,pde,mesh);    
% 
