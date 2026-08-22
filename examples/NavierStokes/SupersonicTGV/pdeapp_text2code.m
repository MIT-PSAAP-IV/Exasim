% Put the Exasim MATLAB frontend on the path. For an installed Exasim use
% run('<prefix>/share/exasim/matlab/exasim_setup.m') instead.
run(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', 'frontends', 'Matlab', 'exasim_setup.m'));

% initialize pde structure and mesh structure
[pde,mesh] = initializeexasim();

% Define a PDE model: governing equations, initial solutions, and boundary conditions
pde.model = "ModelD";          % ModelC, ModelD, ModelW
pde.modelfile = "pdemodel";    % name of a file defining the PDE model

% Choose computing platform and set number of processors
%pde.platform = "gpu";         % choose this option if NVIDIA GPUs are available
pde.mpiprocs = 12;             % number of MPI processors
pde.hybrid = 1;

% Set discretization parameters, physical parameters, and solver parameters
pde.porder = 2;          % polynomial degree
pde.pgauss = 2*pde.porder;
pde.torder = 3;          % time-stepping order of accuracy
pde.nstage = 3;          % time-stepping number of stages

% Lusher and Sandham, AIAA J. 2020: Mref = 1.25 TGV, fixed
% nondimensional time step Delta t = 5e-4, advanced to t = 20.
Deltat = 1e-2;
nsteps = round(20/Deltat);
pde.dt = Deltat*ones(1,nsteps);   % time step sizes
pde.saveSolFreq = 20;
pde.saveSolOpt = 0;
% [8e-3, 4e-3, 2e-3, 1.6e-3, 1.5e-3, 1.4e-3, 1.34e-3]

nspatial = 16;
gam = 1.4;                      % specific heat ratio
Re = 1600;                      % Reynolds number
Pr = 0.71;                      % Prandtl number    
Minf = 1.25;                    % reference Mach number
rhoRef = 1.0;                   % nondimensional reference density
hm = 2*pi/nspatial;             % AV sensor length scale
avcoeff = 2.0e-3;               % AV coefficient
pde.physicsparam = [gam Re Pr Minf rhoRef hm avcoeff pde.porder];
pde.tau = 5.0;                  % DG stabilization parameter
pde.GMRESortho = 1;
pde.GMRESrestart=24;
pde.linearsolvertol=1e-7;
pde.linearsolveriter=24;
pde.preconditioner=1;
pde.precMatrixType=2;
pde.NLiter=1;
pde.NLtol = 1e-8;
pde.ppdegree = 0;
pde.RBdim = 5;
pde.gencode = 1;

% Artificial viscosity for shock capturing; pdemodel uses av = v(1).
pde.AV = 1;
pde.frozenAVflag = 1;
pde.AVsmoothingIter = 2;

% Create a periodic cube 0 <= x,y,z <= 2*pi*L with L = 1.
[mesh.p,mesh.t] = cubemesh(nspatial,nspatial,nspatial,1);
mesh.p = 2*pi*mesh.p;
% expressions for domain boundaries
mesh.boundaryexpr = {@(p) abs(p(2,:))<1e-8, @(p) abs(p(1,:)-2*pi)<1e-8, @(p) abs(p(2,:)-2*pi)<1e-8, @(p) abs(p(1,:))<1e-8, @(p) abs(p(3,:))<1e-8, @(p) abs(p(3,:)-2*pi)<1e-8};
mesh.boundarycondition = [1;1;1;1;1;1];
% Set periodic boundary conditions
mesh.periodicexpr = {2, @(p) p([2 3],:), 4, @(p) p([2 3],:); 1, @(p) p([1 3],:), 3, @(p) p([1 3],:); 5, @(p) p([1 2],:), 6, @(p) p([1 2],:)};

exasimroot = fullfile(fileparts(mfilename('fullpath')), '..', '..', '..');
exportdir = fullfile(exasimroot, 'apps', 'navierstokes', 'supersonicTGV');
exporttext2code(pde, mesh, exportdir);

%[sol,pde,mesh] = exasim(pde,mesh);

