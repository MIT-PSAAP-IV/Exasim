% Put the Exasim MATLAB frontend on the path. For an installed Exasim use
% run('<prefix>/share/exasim/matlab/exasim_setup.m') instead.
run(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', 'frontends', 'Matlab', 'exasim_setup.m'));

% initialize pdehm structure and mesh structure
cd("hm");
[pdehm,~] = initializeexasim();
cd ..;

% Use the same mesh as the NS solver
pdehm.model = "ModelD";          % ModelC, ModelD, ModelW
pdehm.modelfile = "pdemodel_hm";    % name of a file defining the pdehm model

% Choose computing platform and set number of processors
pdehm.platform = "cpu";         % choose this option if NVIDIA GPUs are available
pdehm.mpiprocs = 16;             % number of MPI processors
pdehm.hybrid = 1;               % 0 -> LDG, 1 -> HDG
pdehm.debugmode = 0;
pdehm.nd = 2;

% Set discretization parameters, physical parameters, and solver parameters
pdehm.porder = mesh.porder;             % match NS polynomial degree
pdehm.pgauss = 2*pdehm.porder;
pdehm.physicsparam = kappa(1)^2*3e-3;
pdehm.tau = 1.0;              % DG stabilization parameter
pdehm.linearsolvertol = 1e-8; % GMRES tolerance
pdehm.ppdegree = 1;          % degree of polynomial preconditioner
pdehm.RBdim = 0;

mesh.boundarycondition = [1;1;1;1];
mesh.udg = zeros(size(mesh.dgnodes,1), 3, size(mesh.dgnodes,3));
div = divergence(sol, 1);
mesh.vdg = limiting(div,0,3,1e3,0);

% call exasim to generate and run C++ code to solve the pdehm model
[solhm,pdehm,mesh] = exasim(pdehm,mesh);
s = solhm(:,1,:);
s = s/max(s(:));
av = (s-S0).*(atan(gamma*(s-S0))/pi + 0.5) - atan(gamma)/pi + 0.5;
dist = tanh(mesh.dist*5);
av = lambda(1)*(av.*dist);

% plot solution
figure(2); clf; scaplot(mesh,av); axis on; axis equal; axis tight;
