% Reynolds-number sweep for the NACA0012 Navier-Stokes example.
%
% Each row in pde.physicsparamsweep is one concrete physicsparam vector. The
% frontend runs the cases sequentially and writes outputs to
% dataout/paramcase_0001, dataout/paramcase_0002, ...

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

% naca mesh
mesh = mkmesh_naca0012(porder,1,2);

% call exasim to generate and run C++ code once per Reynolds number
[sol,pde,mesh] = exasim(pde,mesh);

% plot the final sweep case when graphics are available
if iscell(sol)
    solplot = sol{end};
else
    solplot = sol;
end

mesh.porder = porder;
mesh.dgnodes = createdgnodes(mesh.p,mesh.t,mesh.f,mesh.curvedboundary,mesh.curvedboundaryexpr,porder);

figure(1); clf; scaplot(mesh,eulereval(solplot(:,1:4,:),'M',gam),[],2);
axis on; axis equal; axis tight;
axis([-0.5 2 -0.62 0.62])
set(gca,'fontsize', 16);
