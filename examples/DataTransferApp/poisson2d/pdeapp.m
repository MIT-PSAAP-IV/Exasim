% Data-transfer app example (Poisson 2D, HDG) -- MATLAB.
%
% Identical to the standard Poisson2D example except for one line:
%
%     pde.exportapp = fullfile(pwd, "poisson2d-bundle");
%
% When set, exasim() additionally packages a self-contained, relocatable
% "data-transfer app" bundle (datain/, kernels/, a generated pdemodel.txt, a
% relocatable CMakeLists.txt + main.cpp, run.sh, manifest). Copy it to any
% machine with an Exasim install and build + run it with no frontend:
%
%     EXASIM_ROOT=/path/to/exasim/install ./run.sh
%
% It is arch-independent -- retarget the build machine's variant with, e.g.,
%     EXASIM_ROOT=/path/to/install EXASIM_VARIANT=gpu ./run.sh
%
% See ../README.md for details.

% Put the Exasim MATLAB frontend on the path. For an installed Exasim use
% run('<prefix>/share/exasim/matlab/exasim_setup.m') instead.
run(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', 'frontends', 'Matlab', 'exasim_setup.m'));

[pde,mesh] = initializeexasim();

pde.model = "ModelD";
pde.modelfile = "pdemodel";

pde.platform = "cpu";
pde.mpiprocs = 1;            % serial; bundle stays variant "cpu"
pde.hybrid = 1;             % 0 -> LDG, 1 -> HDG
pde.nd = 2;

pde.porder = 3;
pde.pgauss = 2*pde.porder;
pde.physicsparam = 1;       % unit thermal conductivity
pde.tau = 1.0;             % DG stabilization parameter

% >>> The only line that distinguishes this from the plain Poisson2D example:
pde.exportapp = fullfile(pwd, "poisson2d-bundle");

[mesh.p,mesh.t] = squaremesh(16,16,1,1);
mesh.boundaryexpr = {@(p) abs(p(2,:))<1e-8, @(p) abs(p(1,:)-1)<1e-8, @(p) abs(p(2,:)-1)<1e-8, @(p) abs(p(1,:))<1e-8};
mesh.boundarycondition = [1;1;1;1];

% Generate + run the solver AND export the data-transfer app bundle.
[sol,pde,mesh] = exasim(pde,mesh);

fprintf("Done! Data-transfer app bundle: %s\n", pde.exportapp);
