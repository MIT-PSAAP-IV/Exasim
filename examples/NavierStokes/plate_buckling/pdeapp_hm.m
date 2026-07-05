% Put the Exasim MATLAB frontend on the path. For an installed Exasim use
% run('<prefix>/share/exasim/matlab/exasim_setup.m') instead.
run(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', 'frontends', 'Matlab', 'exasim_setup.m'));

% initialize pdehm structure and mesh structure
[pdehm,~] = initializeexasim();
pdehm.modelnumber = 1;   % isolate from the NS model (build .exasim/models/1, data dataout1)

% Define a pdehm model: governing equations, initial solutions, and boundary conditions
pdehm.model = "ModelD";          % ModelC, ModelD, ModelW
pdehm.modelfile = "pdemodel_hm";    % name of a file defining the pdehm model

% Choose computing platform and set number of processors
pdehm.platform = "cpu";
pdehm.mpiprocs = 16;
pdehm.hybrid = 1;               % 0 -> LDG, 1 -> HDG
pdehm.debugmode = 0;
pdehm.nd = 2;

% Set discretization parameters, physical parameters, and solver parameters
pdehm.porder = mesh.porder;             % match NS polynomial degree
pdehm.pgauss = 2*pdehm.porder;
pdehm.physicsparam = kappa0^2*3e-3;
pdehm.saveParaview = 1;
pdehm.tau = 1.0;              % DG stabilization parameter
pdehm.linearsolvertol = 1e-8; % GMRES tolerance
pdehm.ppdegree = 1;          % degree of polynomial preconditioner
pdehm.RBdim = 0;

% Create a separate mesh for the Helmholtz solve (don't modify the NS mesh)
meshhm = mesh; % copy ns mesh
meshhm.boundarycondition = [1;1;1;1];
meshhm.udg = zeros(size(meshhm.dgnodes,1), 1, size(meshhm.dgnodes,3));

div = divergence(sol, 1);
meshhm.vdg = limiting(div,0,3,1e3,0);

% call exasim to generate and run C++ code to solve the pdehm model
[solhm,pdehm,meshhm] = exasim(pdehm,meshhm);

% plot solution
figure(2); clf; scaplot(meshhm,meshhm.vdg); axis on; axis equal; axis tight;
