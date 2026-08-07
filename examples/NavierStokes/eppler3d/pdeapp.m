% Add Exasim to Matlab search path
cdir = pwd(); ii = strfind(cdir, "Exasim");
run(cdir(1:(ii+5)) + "/install/setpath.m");

porder = 2;                     % polynomial degree
gam = 1.4;                      % gas constant
Minf = 0.09;                    % freestream Mach number
tau = 8;                        % stabilization parameter
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
pde.mpiprocs = 32;
pde.hybrid = 1;
pde.debugmode = 0;
pde.nd = 3;
pde.porder = porder;
pde.pgauss = 2*porder;

pde.physicsparam = [gam Re Pr Minf rinf ruinf rvinf rwinf rEinf];
pde.tau = tau;
pde.GMRESrestart = 70;
pde.GMRESortho = 1;
pde.linearsolvertol = 1e-8;
pde.linearsolveriter = 70;
pde.preconditioner = 1;
pde.NLtol = 1e-8;
pde.NLiter = 1;
pde.ppdegree = 0;
pde.RBdim = 10;
pde.gencode = 1;

pde.torder = 3;
pde.nstage = 3;
pde.dt = 0.005*ones(1,2000);
pde.saveSolFreq = 4;
pde.saveSolBouFreq = 2;
pde.ibs = 1;

% Spanwise extrusion of the Eppler 387 C-grid.
% nz = 12;
% mesh = mkmesh_eppler3d(porder, 1, -2, nz, 0.1);
% if porder == 3
%   load sol2d.mat
% elseif porder == 2
%   load sol2dp2.mat
% end
% mesh.udg = extrudesol(sol2d(:,1:4,:), porder, nz);
% mesh.udg(:,5,:) = mesh.udg(:,4,:);
% mesh.udg(:,4,:) = 0;

TEC = 15;
sps = [TEC, 1, 1, 1, 1, TEC, 1, 1, 1, 1, TEC];
spr = [10, 10, 10, 10, 10, 10, 10]*70;
yref = [0.0025 0.008 0.02 0.036];
lw = 10;
ll = 10;
nxw = 21;
nflr = 11;
nflf = 11;
nfuf = 15;
nfur = 21;
nr   = 41;
mesh2d = clemesh_airfoil(xf, yf, nxw, nflr, nflf, nfuf, nfur, nr, sps, spr, yref, lw, ll, porder);
nz = 32;
mesh = mkmesh_eppler3d(porder, 1, -2, nz, 0.1, mesh2d);

% call exasim to generate and run C++ code to solve the PDE model
pde.exportapp = "eppler3dp2";
pde.frontendprovider = true;
pde.buildandrun = false;

[sol,pde,mesh,master,dmd] = exasim(pde,mesh);



% % plot final density field
% figure(1); clf;
% scaplot3d(mesh, sol(:,1,:,end), [], 1);
% colorbar; axis equal; axis tight;

