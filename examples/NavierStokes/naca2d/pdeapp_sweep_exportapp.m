% Exported standalone app for a Reynolds-number sweep over the NACA0012 case.
%
% This example packages the generated solver, datain files, and sweep
% definition into naca2d-sweep-exportapp/. The exported app can then run the
% whole sweep without MATLAB by using its run.sh script.

% Put the Exasim MATLAB frontend on the path. For an installed Exasim use
% run('<prefix>/share/exasim/matlab/exasim_setup.m') instead.
run(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', 'frontends', 'Matlab', 'exasim_setup.m'));

porder = 3;                    % polynomial degree
gam = 1.4;                     % gas constant
Minf = 0.25;                   % freestream Mach number
tau = 0.6/Minf;                % stabilization parameter
alpha = 0*pi/180;              % angle of attack
rinf = 1.0;                    % freestream density
ruinf = cos(alpha);            % freestream horizontal velocity
rvinf = sin(alpha);            % freestream vertical velocity
pinf = 1/(gam*Minf^2);         % freestream pressure
rEinf = 0.5+pinf/(gam-1);      % freestream energy
Pr = 0.72;                     % Prandtl number
reynoldsNumbers = [500; 1000; 1500; 2000];

% initialize pde structure and mesh structure
[pde,~] = initializeexasim();

pde.model = "ModelD";          % ModelC, ModelD, ModelW
pde.modelfile = "pdemodel";    % name of a file defining the PDE model

% Choose computing platform and set number of processors
pde.platform = "cpu";
pde.mpiprocs = 4;
pde.hybrid = 1;
pde.debugmode = 0;
pde.porder = porder;
pde.pgauss = 2*porder;

basePhysicsParam = [gam reynoldsNumbers(1) Pr Minf rinf ruinf rvinf rEinf];
pde.physicsparam = basePhysicsParam;
pde.physicsparamsweep = repmat(basePhysicsParam, numel(reynoldsNumbers), 1);
pde.physicsparamsweep(:,2) = reynoldsNumbers;
pde.physicsparamwarmstart = 1;
pde.tau = tau;                 % DG stabilization parameter
pde.GMRESrestart = 100;
pde.GMRESortho = 1;
pde.linearsolvertol = 1e-6;    % GMRES tolerance
pde.linearsolveriter = 1000;
pde.preconditioner = 1;
pde.NLtol = 1e-8;
pde.ppdegree = 10;
pde.RBdim = 0;
pde.gencode = 1;

% Export the relocatable standalone app without running it in a scratch
% directory. Run naca2d-sweep-exportapp/run.sh to execute the sweep.
pde.exportapp = fullfile(pwd, "naca2d-sweep-exportapp");
pde.buildandrun = false;
pde.frontendprovider = true;
if exist(pde.exportapp, 'dir')
    rmdir(pde.exportapp, 's');
end

% naca mesh
mesh = mkmesh_naca0012(porder,1,2);

% call exasim to preprocess, generate code, and export the standalone app
exasim(pde,mesh);

fprintf("Exported NACA2D sweep app: %s\n", pde.exportapp);
fprintf("Run with: EXASIM_ROOT=%s %s\n", char(exasim_install_prefix()), fullfile(pde.exportapp, "run.sh"));
