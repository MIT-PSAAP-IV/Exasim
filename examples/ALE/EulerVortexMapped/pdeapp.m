% Add Exasim to Matlab search path
cdir = pwd(); ii = strfind(cdir, "Exasim");
run(cdir(1:(ii+5)) + "/install/setpath.m");

% initialize pde structure and mesh structure
[pde,mesh] = initializeexasim();

% Define a PDE model: governing equations, initial solutions, and boundary conditions
pde.model = "ModelC";          % ModelC, ModelD, ModelW
pde.modelfile = "pdemodel";    % name of a file defining the PDE model

% Choose computing platform and set number of processors
pde.mpiprocs = 16;             % number of MPI processors
pde.hybrid = 0;

% Set PATH/LD paths to user's Metis installation
metisbin = fullfile(getenv("HOME"), "tools", "metis", "bin");
metislib = fullfile(getenv("HOME"), "tools", "metis", "lib");
setenv("PATH", metisbin + ":" + getenv("PATH"));
setenv("LD_LIBRARY_PATH", metislib + ":" + getenv("LD_LIBRARY_PATH"));

% Set discretization parameters, physical parameters, and solver parameters
pde.porder = 2;          % polynomial degree
pde.torder = 3;          % time-stepping order of accuracy
pde.nstage = 3;          % time-stepping number of stages

t0 = sqrt(10^2 + 5^2);   % final time from paper
pde.dt = 0.05*ones(1,ceil(t0/0.05));  % ~224 steps to reach t0
pde.soltime = length(pde.dt);         % collect only final step
pde.visdt = 1.0;          % frequent output for animation (every ~1.0 time units)

pde.linearsolveriter = 50;
pde.GMRESrestart = 50;
pde.NLiter = 2;

pde.physicsparam = [1.4 0.5 0.3 1.5 5 5];  % [gam, Minf, eps, rc, x0, y0]
pde.tau = 3.0;             % DG stabilization parameter

% enable ALE formulation (sinusoidal deformation from paper)
pde.ALE = 1;

% place datain/dataout in the example folder
pde.buildpath = pwd();

% create a 64x48 grid on [0,20]x[0,15]
[mesh.p,mesh.t] = squaremesh(64,48,1,1);
mesh.p = [20*mesh.p(1,:); 15*mesh.p(2,:)];
% expressions for domain boundaries (bottom, right, top, left)
mesh.boundaryexpr = {@(p) abs(p(2,:))<1e-8, @(p) abs(p(1,:)-20)<1e-8, @(p) abs(p(2,:)-15)<1e-8, @(p) abs(p(1,:))<1e-8};
mesh.boundarycondition = [1;1;1;1];
% Set periodic boundary conditions (right<->left, bottom<->top)
mesh.periodicexpr = {2, @(p) p(2,:), 4, @(p) p(2,:); 1, @(p) p(1,:), 3, @(p) p(1,:)};

% generate input files and C++ code
[pde,mesh,master,dmd] = preprocessing(pde,mesh);
kkgencode(pde);

% Build: cmake --build . --target cpumpiEXASIM
% Run:   cd /home/tsili/dv/Exasim/build && ./cpumpiEXASIM 1 /path/to/EulerVortexMapped/datain/ /path/to/EulerVortexMapped/dataout/out
