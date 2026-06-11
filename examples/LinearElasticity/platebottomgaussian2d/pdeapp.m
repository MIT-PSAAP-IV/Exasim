% Put the Exasim MATLAB frontend on the path. For an installed Exasim use
% run('<prefix>/share/exasim/matlab/exasim_setup.m') instead.
run(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', 'frontends', 'Matlab', 'exasim_setup.m'));

% initialize pde structure and mesh structure
[pde, mesh] = initializeexasim();

% Define a PDE model: governing equations, initial solutions, and boundary conditions
pde.model = "ModelD";
pde.modelfile = "pdemodel";

% Choose computing platform and set number of processors
pde.platform = "cpu";
pde.mpiprocs = 1;
pde.hybrid = 1;
pde.debugmode = 0;

% Plate geometry: (-0.5, 0.5) x (0, 0.2)
plateLength = 1.0;
plateThickness = 0.2;

% Bottom displacement parameters: dy = a * exp(-(x-b)^2 / (2*c^2))
a = 0.08;
b = 0.00;
c = 0.15;

% Linear elasticity parameters
E = 1.0;
nu = 0.30;
mu = E / (2 * (1 + nu));
lambda = nu * E / ((1 + nu) * (1 - 2 * nu));

% Set discretization parameters, physical parameters, and solver parameters
pde.porder = 2;
pde.pgauss = 2 * pde.porder;
pde.physicsparam = [mu lambda a b c];
pde.tau = 2 * lambda;
pde.linearsolvertol = 1e-8;
pde.ppdegree = 12;
pde.RBdim = 0;
pde.GMRESrestart = 100;
pde.linearsolveriter = 200;
pde.preconditioner = 1;

% Boundary-layer quadrilateral mesh refined toward the plate bottom
mesh = mkmesh_platebottomgaussian(120, 48, 1.2);
mesh.boundarycondition = [2; 1; 1; 1];

figure(1); clf;
simpplot(mesh.p',mesh.t'); 
axis on; axis equal; axis tight
set(gca,'FontSize',20);
xlabel("x");
ylabel("y");
title("Initial mesh");
exportgraphics(gca, "initialmesh.png",'Resolution',300);

% Solve the linear elasticity problem
[sol, pde, mesh, master, dmd] = exasim(pde, mesh);

% Visualize the deformed mesh
mesh.elemtype = 1;
mesh.dgnodes = mesh.dgnodes + sol(:, 1:2, :);

figure(2); clf;
meshplot(mesh, 1);
axis on; axis equal; axis tight
set(gca,'FontSize',20);
xlabel("x");
ylabel("y");
title("Deformed mesh");
exportgraphics(gca, "deformedmesh.png",'Resolution',300);

fprintf("Plate length     : %.3f m\n", plateLength);
fprintf("Plate thickness  : %.3f m\n", plateThickness);
fprintf("Gaussian a, b, c : %.4f, %.4f, %.4f\n", a, b, c);

