% Add Exasim to Matlab search path
cdir = pwd(); ii = strfind(cdir, "Exasim");
run(cdir(1:(ii+5)) + "/install/setpath.m");

porder = 2;                     % polynomial degree
gam = 1.4;                      % gas constant
Minf = 0.09;                    % freestream Mach number
tau = 10;                        % stabilization parameter
alpha = 4*pi/180;               % angle of attack
beta = 0;                       % sideslip angle
rinf = 1.0;                     % freestream density
ruinf = cos(alpha)*cos(beta);   % freestream x momentum
rvinf = sin(alpha);             % freestream y momentum
rwinf = cos(alpha)*sin(beta);   % freestream z momentum
pinf = 1/(gam*Minf^2);          % freestream pressure
rEinf = 0.5 + pinf/(gam-1);     % freestream energy
Re = 300000;                     % Reynolds number
Pr = 0.72;                      % Prandtl number

% initialize pde structure and mesh structure
[pde,~] = initializeexasim();

pde.model = "ModelD";
pde.modelfile = "pdemodel";

% Choose computing platform and set number of processors
pde.platform = "cpu";
pde.mpiprocs = 16;
pde.hybrid = 1;
pde.debugmode = 0;
pde.nd = 3;
pde.porder = porder;
pde.pgauss = 2*porder;

pde.physicsparam = [gam Re Pr Minf rinf ruinf rvinf rwinf rEinf];
pde.tau = tau;
pde.GMRESrestart = 100;
pde.GMRESortho = 1;
pde.linearsolvertol = 1e-7;
pde.linearsolveriter = 60;
pde.preconditioner = 1;
pde.NLtol = 1e-7;
pde.NLiter = 1;
pde.ppdegree = 0;
pde.RBdim = 10;
pde.gencode = 1;

pde.torder = 2;
pde.nstage = 2;
pde.dt = [1e-4 2e-4 4e-4 8e-4 1.6e-3 0.003*ones(1,3600)];
pde.saveSolFreq = 6;
pde.saveSolBouFreq = 4;
pde.ibs = 1;

% Spanwise extrusion of the Eppler 387 C-grid.
mesh = mkmesh_eppler3d(porder, 1, -1, 16, 0.1);

% call exasim to generate and run C++ code to solve the PDE model
pde.exportapp = "eppler3d";
pde.frontendprovider = true;
pde.buildandrun = false;
[sol,pde,mesh] = exasim(pde,mesh);


% % plot final density field
% figure(1); clf;
% scaplot3d(mesh, sol(:,1,:,end), [], 1);
% colorbar; axis equal; axis tight;

