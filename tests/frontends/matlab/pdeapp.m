% Frontend end-to-end test: Poisson 2D on the unit square (serial, HDG).
% Generates kernels with the MATLAB frontend, builds the solver against the
% installed Exasim via the external-model path, runs it, and gates the QoI
% (Domain_QoI1 = integral of (u - uexact)^2) against QOI_TOL.
% The test wrapper puts the frontend on the path via exasim_setup.m.

% initialize pde structure and mesh structure
[pde,mesh] = initializeexasim();

pde.model = "ModelD";
pde.modelfile = "pdemodel";
pde.platform = "cpu";
pde.mpiprocs = 1;
pde.hybrid = 1;
pde.nd = 2;
pde.gencode = 1;

pde.porder = 3;
pde.pgauss = 2*pde.porder;
pde.physicsparam = 1;
pde.tau = 1.0;
pde.linearsolvertol = 1e-8;
pde.ppdegree = 1;
pde.RBdim = 0;

[mesh.p,mesh.t] = squaremesh(16,16,1,1);
mesh.boundaryexpr = {@(p) abs(p(2,:))<1e-8, @(p) abs(p(1,:)-1)<1e-8, @(p) abs(p(2,:)-1)<1e-8, @(p) abs(p(1,:))<1e-8};
mesh.boundarycondition = [1;1;1;1];

% generate and run C++ code to solve the PDE model
[sol,pde,mesh] = exasim(pde,mesh);

% QoI gate
qoifile = char(pde.datapath + "/dataout/outqoi.txt");
lines = strsplit(strtrim(fileread(qoifile)), '\n');
last = strsplit(strtrim(lines{end}));
err2 = str2double(last{2});
tolenv = getenv('QOI_TOL');
if isempty(tolenv), tol = 1e-8; else, tol = str2double(tolenv); end
fprintf("L2 error^2 = %g (tol %g)\n", err2, tol);
assert(err2 < tol, "QoI gate failed: %g >= %g", err2, tol);
fprintf("FRONTEND TEST PASSED\n");
